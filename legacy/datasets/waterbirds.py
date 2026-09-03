import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from datasets import WILDSCFDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from torchvision import transforms


class BaseWaterbirdsDataset(WILDSCFDataset):
    _split_list = ['train', 'counterfactual', 'test', 'val']
    _split_dict = {name: i for i, name in enumerate(_split_list)}
    _split_names = {
        'train': 'Train',
        'counterfactual': 'Counterfactual (ID)',
        'test': 'Test (OOD)',
        'val': 'Validation'
    }
    _metadata_fields = ['background', 'y', 'cf']
    _metadata_map = {
        'background': ['land', 'water', 'snow', 'desert'],
        'y': [' landbird', 'waterbird']
    }
    _y_size = 1
    _n_classes = 2
    _original_resolution = (3, 224, 224)
    default_domain_fields = ['background', 'y']
    default_optimizer = 'sgd'
    counterfactual_fields = ['cf']

    def load_metadata(self, metadata_df):
        self._y_array = torch.LongTensor(metadata_df['y'].values)
        cf_values = np.nan_to_num(metadata_df['cf'].values, nan=0)
        self._metadata_array = torch.stack([
            torch.LongTensor(metadata_df['place'].values),
            self._y_array,
            torch.LongTensor(cf_values)
        ], dim=1)
        self._split_array = metadata_df['split_id'].values
        self._input_array = metadata_df['img_filename'].values
        self._eval_grouper = CombinatorialGrouper(
            dataset=self, groupby_fields=self.default_domain_fields
        )

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        metric = Accuracy(prediction_fn=prediction_fn)
        results, results_str = self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )
        # Adjusted average accuracy (based on training distribution)
        results['adj_acc_avg'] = (
            results['acc_y:landbird_background:land'] * 3498 +
            results['acc_y:landbird_background:water'] * 184 +
            results['acc_y:waterbird_background:land'] * 56 +
            results['acc_y:waterbird_background:water'] * 1057
        ) / (3498 + 184 + 56 + 1057)
        results_str = f"Adjusted average acc: {results['adj_acc_avg']:.3f}\n" + '\n'.join(results_str.split('\n')[1:])
        return results, results_str

    @property
    def eval_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @property
    def train_transform(self):
        return transforms.Compose([
        transforms.RandomResizedCrop((224, 224),
        scale=(0.7, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class CounterfactualWaterbirdsDataset(BaseWaterbirdsDataset):
    _dataset_name = 'counterfactual-waterbirds'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None
        }
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._split_scheme = split_scheme
        self._data_dir = self.initialize_data_dir(root_dir, download)
        self.input_shape = (3, 224, 224)
        if not os.path.exists(self.data_dir):
            raise ValueError(f'{self.data_dir} does not exist yet. Please generate the dataset first.')

        metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        metadata_df['split'] = metadata_df['split_id'].apply(lambda x: self._split_list[x])
        self.load_metadata(metadata_df)

        super().__init__(root_dir, download, split_scheme)

    def get_input(self, idx):
        return Image.open(os.path.join(self.data_dir, self._input_array[idx])).convert('RGB')

    @property
    def key_metric(self):
        return 'acc_wg'

class CounterfactualWaterbirdsClipDataset(BaseWaterbirdsDataset):
    _dataset_name = 'waterbirds-cf-clip'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/',
            'compressed_size': None
        }
    }

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._split_scheme = split_scheme
        self._data_dir = self.initialize_data_dir(root_dir, download)

        if not os.path.exists(self.data_dir):
            raise ValueError(f'{self.data_dir} does not exist yet. Please generate the dataset first.')
        self.input_shape = 512
        self._x_array = torch.load(os.path.join(self.data_dir, 'x_array.pth'))
        self.diff = torch.load(os.path.join(self.data_dir, 'diff.pth'))

        metadata_df = pd.read_csv(os.path.join(self.data_dir, 'metadata.csv'))
        metadata_df['split'] = metadata_df['split_id'].apply(lambda x: self._split_list[x])
        self.load_metadata(metadata_df)

        super().__init__(root_dir, download, split_scheme)

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

    @property
    def key_metric(self):
        return 'acc_wg'
