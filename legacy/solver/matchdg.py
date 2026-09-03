from solver import ERM
from tqdm.auto import tqdm

import torch
import wandb
import numpy as np
# Custom Models
from models import *
import math
import random


class MatchDG(ERM):  
    def __init__(self, hparam):
        super().__init__(hparam)
        if self.hparam['projection'] == 'oracle':
            self.diff = self.dataset.diff
        elif self.hparam['projection'] == 'conditional':
            self.diff = self._condition_matching()
        elif self.hparam['projection'] == 'nearest':
            self.diff = self._nearest_matching()
        else:
            raise ValueError("projection must be 'oracle', 'conditional' or 'nearest'")
        self.diff = self.diff.to(self.device)

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.    
            for x,y_true,metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true).mean() + self.hparam['param1'] * self.constraint()
        
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            logs = {"training_loss": total_loss / len(self.train_set)}
            if self.hparam['wandb']:
                wandb.log(logs, step=i)
            else:
                print(logs)
            self.evaluate(i)
    
    def _condition_matching(self):
        # Step 1: Get shuffled training indices
        train_idx = np.where(self.dataset.split_array == self.dataset._split_dict['train'])[0]
        if 'counterfactual' in self.dataset._split_list:
            train_idx = np.concatenate((train_idx, np.where(self.dataset.split_array == self.dataset._split_dict['counterfactual'])[0]))
        np.random.shuffle(train_idx)

        # Step 2: Extract domain and y values
        metadata = self.dataset.metadata_array[train_idx]
        """
        Return pairs of indices (i, j) where metadata[i] and metadata[j]
        have the same (col0, col1) values.
        """
        y_idx = self.dataset.metadata_fields.index('y')
        domain_idx = self.dataset.metadata_fields.index(self.dataset.default_domain_fields[0])
        
        seen = {} # dictionary to store seen (domain, y) pairs in the following structure: seen: {y: {domain: [index1, index2, ...]}}
        pairs = []

        for i, row in enumerate(metadata):
            y_ = row[y_idx].item()
            domain_ = row[domain_idx].item()

            if y_ in seen:
                # We found a match: pair up i with the stored row index
                potential_pairs = {}
                for key_domain, values in seen[y_].items():
                    if key_domain != domain_:
                        for v in values:
                            potential_pairs[v] = key_domain
                if len(potential_pairs) == 0:
                    # No potential pairs found, store the index
                    if domain_ not in seen[y_]:
                        seen[y_][domain_] = [i]
                    else:
                        seen[y_][domain_].append(i)
                else:
                    # Randomly select one of the potential pairs
                    # and pair it with the current index
                    j = random.choice(list(potential_pairs.keys()))
                    dom_j = potential_pairs[j]
                    pairs.append((j, i))
                    seen[y_][dom_j].remove(j)
                    if not seen[y_][dom_j]:
                        del seen[y_][dom_j]
            else:
                # Haven’t seen this label yet; initialize a dictionary for it.
                seen[y_] = {domain_: [i]}
        pairs = torch.tensor(pairs)
        diffs = self.dataset._x_array[pairs[:, 0]] - self.dataset._x_array[pairs[:, 1]]

        return diffs
    
    def _nearest_matching(self):
        # 1) collect & shuffle train indices
        train_idx = np.where(self.dataset.split_array == self.dataset._split_dict['train'])[0]
        if 'counterfactual' in self.dataset._split_list:
            train_idx = np.concatenate((train_idx, np.where(self.dataset.split_array == self.dataset._split_dict['counterfactual'])[0]))
        np.random.shuffle(train_idx)

        # 2) grab metadata & x‑vectors for those indices
        meta       = self.dataset.metadata_array[train_idx]
        X_all      = self.dataset._x_array[train_idx]
        y_col      = self.dataset.metadata_fields.index('y')
        d_col      = self.dataset.metadata_fields.index(self.dataset.default_domain_fields[0])

        pairs = []

        # 3) for each label y, compute within‑group distances & mask same‑domain
        for y_val in np.unique(meta[:, y_col]):
            mask_y    = (meta[:, y_col] == y_val)
            local_ids = np.nonzero(mask_y).squeeze()     # positions in train_idx with this y
            if len(local_ids) < 2:
                continue

            X_y       = X_all[local_ids]         # [M, D]
            domains   = meta[local_ids, d_col]    # shape [M]

            # all‑pairs L2 distances
            dist = torch.cdist(X_y, X_y, p=2)     # [M, M]

            # mask out same‑domain & self (set to ∞)
            same_dom  = domains.unsqueeze(1) == domains.unsqueeze(0)
            dist[same_dom] = float('inf')

            # for each i, pick nearest j
            nn_j = torch.argmin(dist, dim=1)      # [M]
            for i_loc, j_loc in enumerate(nn_j):
                if dist[i_loc, j_loc] == float('inf'):
                    # no cross‑domain neighbour found
                    continue
                idx_i = train_idx[ local_ids[i_loc] ]
                idx_j = train_idx[ local_ids[j_loc] ]
                pairs.append((idx_i, idx_j))

        # 4) pack into tensor & compute diffs
        if not pairs:
            return torch.empty(0, X_all.size(1))  # no matches

        P = torch.tensor(pairs, dtype=torch.long)    # [K, 2]
        diffs = self.dataset._x_array[P[:, 0]] - self.dataset._x_array[P[:, 1]]
        return diffs

    def _initialize_model(self):
        if self.hparam['featurizer'] == 'linear':
            if self.hparam['pretrained']:
                self.featurizer = Linear(input_shape=512, latent_dim=self.hparam['latent_dim']).to(self.device)                
            else: 
                self.featurizer = Linear(input_shape=math.prod(self.dataset.input_shape), latent_dim=self.hparam['latent_dim']).to(self.device)

        else:
            if 'MNIST' in self.dataset.dataset_name:
                self.featurizer = MNIST_CNN(input_shape=self.dataset.input_shape, latent_dim=self.hparam['latent_dim']).to(self.device)
            else:
                self.featurizer = ResNet50(input_shape=self.dataset.input_shape, latent_dim=self.hparam['latent_dim']).to(self.device)
        
        self.classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=self.dataset._n_classes)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier)).to(self.device)
    
    def constraint(self):
        return torch.norm(self.featurizer(self.diff)) ** 2 / self.diff.shape[0]
        