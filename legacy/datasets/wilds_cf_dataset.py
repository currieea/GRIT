
from wilds.datasets.wilds_dataset import WILDSDataset, WILDSSubset  # Ensure WILDSDataset and WILDSSubset are imported for type hinting
import numpy as np
import torch


class WILDSCFDataset(WILDSDataset):
    """
    Base class for counterfactual WILDS datasets. 
    """

    def get_subset(self, splits, frac=1.0, transform=None):
        """
        Args:
            - splits (str or list): The list of splits to include in the subset.
            - frac (float): What fraction of the split to randomly sample.
                            Used for fast development on a small dataset.
            - transform (function): Any data transformations to be applied to the input x.
        Output:
            - subset (WILDSSubset): A (potentially subsampled) subset of the WILDSDataset.
        """
        
        split_mask = torch.zeros(len(self), dtype=bool)
        if isinstance(splits, str):
            splits = [splits]
        for split in splits: 
            if split not in self.split_dict:
                raise ValueError(f"Split {split} not found in dataset's split_dict.")
            split_mask |= (self.split_array == self.split_dict[split])
        split_idx = np.where(split_mask)[0]

        if frac < 1.0:
            # Randomly sample a fraction of the split
            num_to_retain = int(np.round(float(len(split_idx)) * frac))
            split_idx = np.sort(np.random.permutation(split_idx)[:num_to_retain])

        return WILDSSubset(self, split_idx, transform)
    
    @property
    def train_transform(self):
        return None

    @property
    def eval_transform(self):
        """
        Default test transform. Override in subclasses if needed.
        """
        return None
    
    @property
    def key_metric(self):
        raise NotImplementedError