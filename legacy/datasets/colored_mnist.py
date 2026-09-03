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


class ColoredMNISTDataset(WILDSCFDataset):
    _dataset_name = "ColoredMNIST"
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
        self.num_train_samples = 50000
        self._n_classes = 2
        self._split_dict = {'train': 0, 'in_val': 1, 'val': 2, 'in_test': 3, 'test': 4}
        self._split_names = {'train': 'Train', 'in_val': 'Val (In-Domain)',  'val': 'Val(OOD/Trans)', 'in_test': 'Test (In-Domain)', 'test': 'Test (OOD/Trans)'}
        self._metadata_fields = ["split", "color", "digit", "y", "id"]
        self._split_array, self._x_array, self._y_array, self._metadata_array = self._get_data()
        self.default_domain_fields = ['domain']
        self.counterfactual_fields = ['id']
        self.default_optimizer = 'sgd'
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=self.default_domain_fields)
        # super().__init__(root_dir, download, split_scheme)
    
    def _get_data(self):
        # split scheme is the test domain color digit correlation. We always test on test dataset.
        if self.split_scheme == "official":
            self._training_domains = ["0.1", "0.2"]
            self._in_test_domain = self._training_domains
            self._test_domain = "0.9"
        elif self.split_scheme == "oracle":
            self._training_domains = ["0.9"]
            self._in_test_domain = self._training_domains
            self._test_domain = "0.9"
        else: 
            raise NotImplementedError
        
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)
        test_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=False, download=True, transform=transforms.ToTensor()), batch_size=10000, shuffle=False)
        mnist_x_list = None
        mnist_y_list = None
        metaarray_list = None
        mnist_split_list = None

        for x, y in train_loader:
            images = x
            digits = y
        
        # train
        for d, e in enumerate(self._training_domains):
            identity = torch.arange(d, self.num_train_samples, len(self._training_domains))
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[d:self.num_train_samples:2] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(float(e)*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[d:self.num_train_samples:2], images[d:self.num_train_samples:2]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            split = 0
            metaarray = torch.stack([split*torch.ones_like(labels), colors, digits[d:self.num_train_samples:2], labels, identity], dim=1)

            if mnist_x_list is None:
                mnist_x_list = cimages
                mnist_y_list = labels
                metaarray_list = metaarray
                mnist_split_list = split*torch.ones_like(labels)
            else:
                mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
                mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
                metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
                mnist_split_list = torch.cat([mnist_split_list, torch.zeros_like(labels)], dim=0)
        
        # in_val
        for d, e in enumerate(self._training_domains):
            identity = torch.arange(self.num_train_samples+d,len(images), len(self._training_domains))
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[self.num_train_samples+d::2] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(float(e)*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[self.num_train_samples+d::2], images[self.num_train_samples+d::2]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            split = 1
            metaarray = torch.stack([split*torch.ones_like(labels), colors, digits[self.num_train_samples+d::2], labels, identity], dim=1)
            if mnist_x_list is None:
                mnist_x_list = cimages
                mnist_y_list = labels
                metaarray_list = metaarray
                mnist_split_list = torch.ones_like(labels)
            else:
                mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
                mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
                metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
                mnist_split_list = torch.cat([mnist_split_list, torch.ones_like(labels)], dim=0)
        
        # val
        identity = torch.arange(self.num_train_samples, len(images))
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits[self.num_train_samples:] < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; 
        colors = torch.logical_xor(labels, torch.bernoulli(float(self._test_domain)*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images[self.num_train_samples:], images[self.num_train_samples:]], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        split = 2
        metaarray = torch.stack([split*torch.ones_like(labels), colors, digits[self.num_train_samples:], labels, identity], dim=1)
        metaarray = torch.stack([2*torch.ones_like(labels), 2*torch.ones_like(labels), digits[self.num_train_samples:], labels, identity, len(images)+torch.arange(len(identity))], dim=1)
        
        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 2*torch.ones_like(labels)], dim=0)

        for x, y in test_loader:
            images = x
            digits = y
        
        # in-domain test
        for d, e in enumerate(self._training_domains):
            identity = 60000+torch.arange(0, len(images), len(self._training_domains))
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[d::2] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(float(e)*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[d::2], images[d::2]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            metaarray = torch.stack([d*torch.ones_like(labels), 3*torch.ones_like(labels), digits[d::2], labels, identity, 10000+identity], dim=1)
            mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
            mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
            metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
            mnist_split_list = torch.cat([mnist_split_list, 3*torch.ones_like(labels)], dim=0)
        
        # out-of-domain test
        identity = torch.arange(len(images))
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; flip the color with probability e
        colors = torch.logical_xor(labels, torch.bernoulli(float(self._test_domain)*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images, images], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        metaarray = torch.stack([2*torch.ones_like(labels), 4*torch.ones_like(labels), digits, labels, identity, 80000+torch.arange(len(identity))], dim=1)
        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 4*torch.ones_like(labels)], dim=0)

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


class ColoredMNISTClipDataset(ColoredMNISTDataset):
    _dataset_name = "ColoredMNIST-cf-clip"
    def __init__(self, version = None, root_dir = "data", download = False, split_scheme = "official"):
        super().__init__(version, root_dir, download, split_scheme)

        self.diff = torch.load(os.path.join(self._data_dir, 'diff.pth'))
    def _get_data(self):
        data_dir =  self._data_dir = self.initialize_data_dir(self._data_dir, download=False)
        return torch.load(os.path.join(data_dir, 'split_array.pth')), torch.load(os.path.join(data_dir, 'x_array.pth')), torch.load(os.path.join(data_dir, 'y_array.pth')), torch.load(os.path.join(data_dir, 'metadata_array.pth'))
