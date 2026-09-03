import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
from datasets import WILDSCFDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from torchvision import transforms


class CounterfactualCelebADataset(WILDSCFDataset):
    _dataset_name = 'counterfactual-CelebA'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/', # to update
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        self._version = version
        self._data_dir = self.initialize_data_dir(root_dir, download)
        target_name = 'Blond_Hair'
        confounder_names = ['Male']

        # Read in attributes
        attrs_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        # Note: idx and filenames are off by one.
        self._input_array = attrs_df['image_id'].values
        self._original_resolution = (3, 178, 218)
        self.input_shape = (3, 224, 224)
        attrs_df = attrs_df.drop(labels='image_id', axis='columns')
        attr_names = attrs_df.columns.copy()
        def attr_idx(attr_name):
            return attr_names.get_loc(attr_name)

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        attrs_df = attrs_df.values
        attrs_df[attrs_df == -1] = 0

        # Get the y values
        target_idx = attr_idx(target_name)
        self._y_array = torch.LongTensor(attrs_df[:, target_idx])
        self._cf_array =  torch.LongTensor(attrs_df[:, attr_idx('cf')])
        self._y_size = 1
        self._n_classes = 2

        # Get metadata
        confounder_idx = [attr_idx(a) for a in confounder_names]
        confounders = attrs_df[:, confounder_idx]

        self._metadata_array = torch.cat([torch.LongTensor(confounders), self._y_array.reshape((-1, 1)), self._cf_array.reshape((-1, 1))], dim=1)
        confounder_names = [s.lower() for s in confounder_names]
        self._metadata_fields = confounder_names + ['y', 'cf']
        self._metadata_map = {
            'y': ['not blond', '    blond'] # Padding for str formatting
        }

        self.default_domain_fields = ['male', 'y']
        self.default_optimizer = 'sgd' # default optimizer for this dataset, used in solver/erm.py
        self.counterfactual_fields = ['cf']
        self._eval_grouper = CombinatorialGrouper(
            dataset=self,
            groupby_fields=(confounder_names + ['y']))
        self.group_count = torch.tensor([[71629, 22880], [66874, 1387]], dtype=torch.float32)
        # Extract splits
        self._split_scheme = split_scheme
        if self._split_scheme != 'official':
            raise ValueError(f'Split scheme {self._split_scheme} not recognized')
        self._split_list = ['train', 'counterfactual', 'test', 'val']
        self._split_dict = {'train': 0, 'counterfactual': 1, 'test': 2, 'val': 3}
        self._split_names = {'train': 'Train', 'counterfactual': 'Counterfactual (ID)',
                             'test': 'Test (OOD)', 'val': 'Validation'}
        split_df = pd.read_csv(
            os.path.join(self.data_dir, 'list_eval_partition.csv'))
        self._split_array = split_df['partition'].values

        super().__init__(root_dir, download, split_scheme)



    def get_input(self, idx):
       # Note: idx and filenames are off by one.
       img_filename = os.path.join(
           self.data_dir,
           'img_align_celeba',
           self._input_array[idx])
       x = Image.open(img_filename).convert('RGB')
       return x

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels 
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric,
            self._eval_grouper,
            y_pred, y_true, metadata)

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

    @property
    def eval_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    @property
    def key_metric(self):
        return 'acc_wg'
    
class CounterfactualCelebAClipDataset(CounterfactualCelebADataset):
    _dataset_name = 'counterfactual-celeba-clip'
    _versions_dict = {
        '1.0': {
            'download_url': 'https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/', # to update
            'compressed_size': None}}

    def __init__(self, version=None, root_dir='data', download=False, split_scheme='official'):
        super().__init__(version, root_dir, download, split_scheme)
        self.input_shape = 512
        self._x_array = torch.load(os.path.join(self.data_dir, 'x_array.pth')) 
        self.diff = torch.load(os.path.join(self.data_dir, 'diff.pth'))   
        

    def get_input(self, idx):
        """
        Returns x for a given idx.
        """
        x = self._x_array[idx]
        return x

    @property
    def train_transform(self):
        return None

    @property
    def eval_transform(self):
        """
        Default test transform. Override in subclasses if needed.
        """
        return None