from datasets import WILDSCFDataset
from typing import Any, Dict, Optional, Tuple, Union, Callable
from pathlib import Path
import os
import numpy as np
import copy
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from itertools import permutations, combinations_with_replacement
import random
from torch.utils.data import Dataset, TensorDataset
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper


# for debug LISA.
class LISAColoredMNISTDataset(WILDSCFDataset):
    _dataset_name = "LISAColoredMNIST"
    _versions_dict = {
        '1.0': {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x19f5d1758a184e13aeaea05e0954422a/contents/blob/",
            "compressed_size": "171_612_540"
        }
    }
    def __init__(
        self, version: str = None, root_dir: str = "data", 
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (2, 28, 28)
        self.input_shape = (2, 28, 28)
        self._y_type: str = "long"
        self._y_size = 1
        # Path of the dataset
        # self._data_dir = self.initialize_data_dir(root_dir, download)
        self._data_dir = Path(root_dir)
        self.invariant = 0.75   # 0.75 if original dataset
        self._n_classes = 2
        self._split_list = ['train', 'val', 'test', 'in_test']
        self._split_dict = {'train': 0, 'val': 1, 'test': 2, 'in_test': 3}
        self._split_names = {'train': 'Train', 'val': 'Val(OOD/Trans)', 'test': 'Test (OOD/Trans)', 'in_test': 'In-Test'}
        self._metadata_fields = ["color", "digit", "y", "id"]
        self._split_array, self._x_array, self._y_array, self._metadata_array = self._get_data()
        self.default_domain_fields = ['color']
        self.counterfactual_fields = ['id']
        self.default_optimizer = 'adam'
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=self.default_domain_fields)
        # super().__init__(root_dir, download, split_scheme)
    
    def _get_data(self):
        # split scheme is the test domain color digit correlation. We always test on test dataset.
        if self.split_scheme == "official":
            self._training_domains = ["0.2"]
            self._in_test_domain = self._training_domains
            self._test_domain = "0.9"
        elif self.split_scheme == "oracle":
            self._training_domains = ["0.9"]
            self._in_test_domain = self._training_domains
            self._test_domain = "0.9"
        else: 
            raise NotImplementedError
        
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)
        mnist_x_list = None
        mnist_y_list = None
        metaarray_list = None
        mnist_split_list = None

        for x, y in train_loader:
            images = x
            digits = y
        # train
            identity = torch.arange(0, 30000)
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[:30000] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(0.2*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[:30000], images[:30000]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            metaarray = torch.stack([colors, digits[:30000], labels, identity], dim=1)
            mnist_x_list = cimages
            mnist_y_list = labels
            metaarray_list = metaarray
            mnist_split_list = torch.zeros_like(labels)

        
        # val
        identity = torch.arange(30000,40000)
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits[30000:40000] < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; 
        colors = torch.logical_xor(labels, torch.bernoulli(0.5*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images[30000:40000], images[30000:40000]], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        metaarray = torch.stack([colors, digits[30000:40000], labels, identity], dim=1)
        
        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 1*torch.ones_like(labels)], dim=0)


        # test
        identity = torch.arange(40000, 50000)
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits[40000:50000] < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; flip the color with probability e
        colors = torch.logical_xor(labels, torch.bernoulli(0.9*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images[40000:50000], images[40000:50000]], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        metaarray = torch.stack([colors, digits[40000:50000], labels, identity], dim=1)

        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 2*torch.ones_like(labels)], dim=0)
        # in-test
        identity = torch.arange(50000, 60000)
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits[50000:60000] < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; flip the color with probability e
        colors = torch.logical_xor(labels, torch.bernoulli(0.2*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images[50000:60000], images[50000:60000]], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        metaarray = torch.stack([colors, digits[50000:60000], labels, identity], dim=1)

        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 3*torch.ones_like(labels)], dim=0)

        return mnist_split_list, mnist_x_list, mnist_y_list.long(), metaarray_list

         
    def get_input(self, idx):
        return self._x_array[idx]

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
    def key_metric(self):
        return 'acc_avg'


class LISAColoredMNISTClipDataset(LISAColoredMNISTDataset):
    _dataset_name = "LISAColoredMNIST-cf-clip"
    def __init__(self, version = None, root_dir = "data", download = False, split_scheme = "official"):
        super().__init__(version, root_dir, download, split_scheme)

        self.diff = torch.load(os.path.join(self._data_dir, 'diff.pth'))
    def _get_data(self):
        data_dir =  self._data_dir = self.initialize_data_dir(self._data_dir, download=False)
        return torch.load(os.path.join(data_dir, 'split_array.pth')), torch.load(os.path.join(data_dir, 'x_array.pth')), torch.load(os.path.join(data_dir, 'y_array.pth')), torch.load(os.path.join(data_dir, 'metadata_array.pth'))
