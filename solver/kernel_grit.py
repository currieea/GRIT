"""
KernelGRIT: Kernel extension of GRIT using RBF kernel + Woodbury correction.

hparam parameters
-----------------
param1 (int)   : n_subsample  — training samples to subsample (default: 2000)
param2 (float) : invariance_strength β  (default: 10000.0)
param3 (float) : RBF gamma bandwidth  (default: 0.01 for 512-dim CLIP features)
projection     : pair method — 'oracle' | 'conditional' | 'nearest' (default: 'oracle')
pretrained     : must be 'true'  (CLIP embeddings required)

Usage
-----
python main.py \\
    --solver KernelGRIT \\
    --pretrained true \\
    --dataset ColoredMNIST \\
    --projection oracle \\
    --param1 2000 \\
    --param2 10000 \\
    --param3 0.01 \\
    --no_wandb
"""

import random
import numpy as np
import torch

from datasets import *   # brings ColoredMNISTClipDataset (and others) into scope for eval()
from solver.invariant_l2_kernel import Direct_Invariant_L2_Kernel


class KernelGRIT:
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = hparam['device']

        assert str(hparam.get('pretrained', 'true')) == 'true', (
            "KernelGRIT requires --pretrained true (CLIP embeddings)."
        )

        self.dataset = eval(hparam['dataset'] + 'ClipDataset')(
            root_dir=hparam['root_dir'],
            download=False,
            split_scheme=hparam['split_scheme'],
        )

        self.n_subsample         = int(float(hparam.get('param1', 2000)))
        self.invariance_strength = float(hparam.get('param2', 10000.0))
        if self.invariance_strength <= 0:
            raise ValueError(
                "KernelGRIT requires param2 > 0. "
                "Non-positive values disable invariance regularization."
            )
        self.gamma               = float(hparam.get('param3', 0.01))
        self.C_svm               = 1.0
        self.projection          = hparam.get('projection', 'oracle')

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------
    def _get_pairs(self, X_tr=None, m_tr=None):
        if self.projection == 'oracle':
            return self._oracle_pairs(X_tr, m_tr)
        elif self.projection == 'conditional':
            return self._condition_matching()
        elif self.projection == 'nearest':
            return self._nearest_matching()
        else:
            raise ValueError(f"Unknown projection: {self.projection}")

    def _oracle_pairs(self, X_tr, m_tr):
        if X_tr is None or m_tr is None:
            raise ValueError("Oracle pairing requires sampled training features and metadata.")
        if not hasattr(self.dataset, "oracle_z") or not hasattr(self.dataset, "oracle_z_prime"):
            raise ValueError(
                "projection='oracle' requires endpoint oracle artifacts "
                "(oracle_z_array.pth and oracle_z_prime_array.pth)."
            )
        if self.dataset.oracle_z is None or self.dataset.oracle_z_prime is None:
            raise ValueError(
                "projection='oracle' in KernelGRIT does not support diff-only artifacts. "
                "Regenerate endpoint oracle artifacts via scripts/coloredMNIST_preprocess.py."
            )
        if "id" not in self.dataset._metadata_fields:
            raise ValueError(
                "projection='oracle' requires metadata field 'id' for diff alignment."
            )

        oracle_z = self.dataset.oracle_z
        oracle_z_prime = self.dataset.oracle_z_prime
        if not torch.is_tensor(oracle_z) or not torch.is_tensor(oracle_z_prime):
            raise ValueError("oracle endpoint artifacts must be torch tensors.")
        if oracle_z.shape != oracle_z_prime.shape or oracle_z.ndim != 2:
            raise ValueError("oracle endpoint artifacts must be matching 2D tensors [K, D].")

        id_col = self.dataset._metadata_fields.index("id")
        m_tr_tensor = m_tr if torch.is_tensor(m_tr) else torch.as_tensor(m_tr)
        diff_ids = m_tr_tensor[:, id_col].to(torch.long)
        assert diff_ids.max() < len(self.dataset.oracle_z), \
            "Sample IDs exceed diff tensor bounds — check ID field semantics"
        assert diff_ids.min() >= 0, \
            "Sample IDs are negative — check ID field semantics"

        if hasattr(self.dataset, "oracle_pair_ids") and self.dataset.oracle_pair_ids is not None:
            oracle_pair_ids = self.dataset.oracle_pair_ids.to(torch.long).cpu()
            expected_ids = torch.arange(len(oracle_pair_ids), dtype=torch.long)
            if not torch.equal(oracle_pair_ids, expected_ids):
                raise ValueError(
                    "oracle_pair_id_array must be 0-based contiguous row indices for KernelGRIT oracle mode."
                )

        z_subset = oracle_z[diff_ids.cpu()].to(torch.float32)
        z_prime_subset = oracle_z_prime[diff_ids.cpu()].to(torch.float32)
        if z_subset.shape[1] != X_tr.shape[1]:
            raise ValueError(
                "Oracle endpoint feature dimension does not match sampled training features: "
                f"{z_subset.shape[1]} vs {X_tr.shape[1]}."
            )

        Z = z_subset.numpy()
        Z_prime = z_prime_subset.numpy()
        print(f"[KernelGRIT] oracle endpoint matching: {len(Z)} pairs")
        return Z, Z_prime

    def _condition_matching(self):
        """Cross-domain same-label pairing (fixed global-index version of ecmp.py)."""
        train_idx = np.where(
            self.dataset._split_array.numpy() == self.dataset._split_dict['train']
        )[0]
        np.random.shuffle(train_idx)

        metadata = self.dataset._metadata_array[train_idx]
        y_col    = self.dataset._metadata_fields.index('y')
        dom_col  = self.dataset._metadata_fields.index(self.dataset.default_domain_fields[0])

        seen  = {}
        pairs = []   # (local_j, local_i)

        for i, row in enumerate(metadata):
            y_     = row[y_col].item()
            domain = row[dom_col].item()
            if y_ in seen:
                candidates = {
                    v: d
                    for d, vals in seen[y_].items()
                    if d != domain
                    for v in vals
                }
                if not candidates:
                    seen[y_].setdefault(domain, []).append(i)
                else:
                    j     = random.choice(list(candidates.keys()))
                    dom_j = candidates[j]
                    pairs.append((j, i))
                    seen[y_][dom_j].remove(j)
                    if not seen[y_][dom_j]:
                        del seen[y_][dom_j]
            else:
                seen[y_] = {domain: [i]}

        if not pairs:
            raise RuntimeError("KernelGRIT: no conditional cross-domain pairs found.")
        pairs    = np.array(pairs)                # (K, 2) — local indices
        global_j = train_idx[pairs[:, 0]]
        global_i = train_idx[pairs[:, 1]]
        Z        = self.dataset._x_array[global_j].numpy()
        Z_prime  = self.dataset._x_array[global_i].numpy()
        print(f"[KernelGRIT] conditional matching: {len(Z)} pairs")
        return Z, Z_prime

    def _nearest_matching(self):
        """Cross-domain nearest-neighbour pairing."""
        train_idx = np.where(
            self.dataset._split_array.numpy() == self.dataset._split_dict['train']
        )[0]
        np.random.shuffle(train_idx)

        meta  = self.dataset._metadata_array[train_idx]
        X_all = self.dataset._x_array[train_idx]
        y_col = self.dataset._metadata_fields.index('y')
        d_col = self.dataset._metadata_fields.index(self.dataset.default_domain_fields[0])

        pairs = []
        for y_val in meta[:, y_col].unique().tolist():
            mask      = (meta[:, y_col] == y_val)
            local_ids = torch.nonzero(mask, as_tuple=False).squeeze(1)
            if local_ids.numel() < 2:
                continue
            X_y      = X_all[local_ids]
            domains  = meta[local_ids, d_col]
            dist     = torch.cdist(X_y, X_y, p=2)
            same_dom = (domains.unsqueeze(1) == domains.unsqueeze(0))
            dist[same_dom] = float('inf')
            nn_j = torch.argmin(dist, dim=1)
            for i_loc, j_loc in enumerate(nn_j.tolist()):
                if dist[i_loc, j_loc].item() == float('inf'):
                    continue
                pairs.append((
                    train_idx[local_ids[i_loc].item()],
                    train_idx[local_ids[j_loc].item()],
                ))

        if not pairs:
            raise RuntimeError("KernelGRIT: no cross-domain pairs found.")
        P       = np.array(pairs)
        Z       = self.dataset._x_array[P[:, 0]].numpy()
        Z_prime = self.dataset._x_array[P[:, 1]].numpy()
        print(f"[KernelGRIT] nearest matching: {len(Z)} pairs")
        return Z, Z_prime

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self):
        split_arr  = self.dataset._split_array.numpy()
        train_mask = split_arr == self.dataset._split_dict['train']

        X_full    = self.dataset._x_array[train_mask].numpy()
        y_full    = self.dataset._y_array[train_mask].numpy()
        meta_full = self.dataset._metadata_array[train_mask]

        # Subsample
        n   = min(self.n_subsample, len(X_full))
        idx = np.random.permutation(len(X_full))[:n]
        X_tr  = X_full[idx]
        y_tr  = y_full[idx]
        m_tr  = meta_full[idx]

        # Build pairs
        Z, Z_prime = self._get_pairs(X_tr, m_tr)

        # Fit kernel SVM
        print(
            f"[KernelGRIT] fitting on N={n}, K={len(Z)} pairs, "
            f"gamma={self.gamma}, beta={self.invariance_strength}"
        )
        self.kernel_clf = Direct_Invariant_L2_Kernel(
            gamma=self.gamma,
            invariance_strength=self.invariance_strength,
            C=self.C_svm,
            normalize_pairs=True,
        )
        self.kernel_clf.fit(X_tr, y_tr, Z, Z_prime)
        print("[KernelGRIT] fit complete.")
        if getattr(self.kernel_clf, "corr_fro_rel", None) is not None:
            print(
                "[KernelGRIT] correction stats: "
                f"corr_fro_rel={self.kernel_clf.corr_fro_rel:.6f}, "
                f"corr_abs_mean={self.kernel_clf.corr_abs_mean:.6f}"
            )

        # Evaluate all splits
        self._eval_split(X_tr, y_tr, m_tr, 'train (subsampled)')
        for split_name, split_id in self.dataset._split_dict.items():
            if split_name == 'train':
                continue
            mask = split_arr == split_id
            if mask.sum() == 0:
                continue
            X_s    = self.dataset._x_array[mask].numpy()
            y_s    = self.dataset._y_array[mask].numpy()
            meta_s = self.dataset._metadata_array[mask]
            self._eval_split(X_s, y_s, meta_s, split_name)

    def _eval_split(self, X, y, metadata, split_name):
        y_pred  = torch.tensor(self.kernel_clf.predict(X), dtype=torch.long)
        y_true  = torch.tensor(y, dtype=torch.long)
        meta_t  = metadata if torch.is_tensor(metadata) else torch.tensor(metadata)
        results, results_str = self.dataset.eval(
            y_pred.cpu(), y_true.cpu(), meta_t.cpu()
        )
        print(f"[{split_name}] {results_str}")
        return results
