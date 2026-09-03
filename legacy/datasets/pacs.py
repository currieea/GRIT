import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from torchvision import transforms
from datasets import WILDSCFDataset


class PACSDataset(WILDSCFDataset):
    _dataset_name = 'pacs'
    _versions_dict = {
        '1.0': {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x19f5d1758a184e13aeaea05e0954422a/contents/blob/",
            "compressed_size": "171_612_540"
        }
    }
    def __init__(self, version: str = None, root_dir: str = "data", download: bool = False, split_scheme: str = "official"):
        self._version = version
        self._split_scheme = split_scheme
        self._original_resolution = (3, 224, 224)
        self.input_shape = (3, 224, 224)
        self._y_type = "long"
        self._y_size = 1
        self._data_dir = Path(self.initialize_data_dir(root_dir, download))
        metadata_filename = "metadata.csv" if split_scheme == 'official' else f"{split_scheme}.csv"
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'in_val': 3, 'in_test': 4}
        self._split_names = {
            'train': 'Train',
            'val': 'Validation (OOD/Trans)',
            'test': 'Test (OOD/Trans)',
            'in_val': 'Validation (ID/Cis)',
            'in_test': 'Test (ID/Cis)'
        }
        self._metadata_fields = ["domain", "y", "idx"]
        self._n_classes = 7
        df = self._load_metadata(metadata_filename)

        self.default_domain_fields = ['domain']
        self.counterfactual_fields = []
        self.default_optimizer = 'sgd'
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=self.default_domain_fields)
    def _load_metadata(self, metadata_filename: str):
        df = pd.read_csv(self._data_dir / metadata_filename)
        df['split_id'] = df['split'].apply(lambda x: self._split_dict[x])
        self._input_array = df['path'].values
        self._split_array = df["split_id"].values
        self._y_array = torch.from_numpy(df["y"].values).type(torch.LongTensor)
        self._metadata_array = torch.tensor(np.stack([
            df['domain_remapped'].values,
            df['y'].values,
            np.arange(df['y'].shape[0])
        ], axis=1))
        return df

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(metric, self._eval_grouper, y_pred, y_true, metadata)


    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._original_resolution[1], self._original_resolution[2])),
            transforms.RandomResizedCrop(self._original_resolution[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def eval_transform(self):
        return transforms.Compose([
            transforms.Resize((self._original_resolution[1], self._original_resolution[2])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def key_metric(self):
        return 'acc_avg'

class PACSClipDataset(PACSDataset):
    _dataset_name = 'pacs-clip'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None
        }
    }

    def __init__(self, version: str = None, root_dir: str = "data", download: bool = False, split_scheme: str = "official"):
        super().__init__(version, root_dir, download, split_scheme)
        self._x_array = torch.load(os.path.join(self.data_dir, 'x_array.pth'))


    def get_input(self, idx):
        return self._x_array[idx]

    @property
    def train_transform(self):
        return None

    @property
    def eval_transform(self):
        """
        Default test transform. Override in subclasses if needed.
        """
        return None