from wilds.datasets.wilds_dataset import WILDSDataset
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
from torch.utils.data import Dataset 

class ColoredMNIST(WILDSDataset):
    _dataset_name = "ColoredMNIST"
    
    def __init__(
        self, version: str = None, root_dir: str = "data", 
        download: bool = False,
        split_scheme: str = "official"
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._split_scheme: str = split_scheme
        self._original_resolution = (2, 28, 28)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(root_dir)
        self.invariant = 0.75   # 0.75 if original dataset
        self.num_train_samples = 50000
        # The original dataset contains 7 categories. 
        # if self._split_scheme == 'official':
        #     metadata_filename = "metadata.csv"
        #     print('dcc')
        # else:
        #     raise Not
        self._n_classes = 2
        self._split_dict = {'train': 0, 'val': 1, 'in_test': 2, 'test': 3}
        self._split_names = {'train': 'Train', 'val': 'Val(OOD/Trans)', 'in_test': 'Test (In-Domain)', 'test': 'Test (OOD/Trans)'}
        self._metadata_fields = ["domain", "split", "digit", "y", "id", "idx"]
        self._split_array, self._data, self._y_array, self._metadata_array = self._get_data()

    
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

        for d, e in enumerate(self._training_domains):

            identity = torch.arange(len(images[d:self.num_train_samples:2]))
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[d:self.num_train_samples:2] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(float(e)*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[d:self.num_train_samples:2], images[d:self.num_train_samples:2]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            metaarray = torch.stack([d*torch.ones_like(labels), torch.zeros_like(labels), digits[d:self.num_train_samples:2], labels, identity, d*self.num_train_samples+identity], dim=1)
            if mnist_x_list is None:
                mnist_x_list = cimages
                mnist_y_list = labels
                metaarray_list = metaarray
                mnist_split_list = torch.zeros_like(labels)
            else:
                mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
                mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
                metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
                mnist_split_list = torch.cat([mnist_split_list, torch.zeros_like(labels)], dim=0)
        
        # val
        identity = torch.arange(len(images[self.num_train_samples:]))
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (digits[self.num_train_samples:] < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
        # Assign a color based on the label; 
        colors = torch.logical_xor(labels, torch.bernoulli(float(self._test_domain)*torch.ones_like(labels))).float()
        # Apply the color to the image by zeroing out the other color channel
        cimages = torch.cat([images[self.num_train_samples:], images[self.num_train_samples:]], dim=1)
        cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
        metaarray = torch.stack([2*torch.ones_like(labels), 1*torch.ones_like(labels), digits[self.num_train_samples:], labels, identity, identity], dim=1)
        
        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, torch.ones_like(labels)], dim=0)

        for x, y in test_loader:
            images = x
            digits = y
        
        # in-domain test
        for d, e in enumerate(self._training_domains):
            identity = torch.arange(len(images))[d::2]
            # Assign a binary label based on the digit; flip label with probability 0.25
            labels = (digits[d::2] < 5).float()
            labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).float()
            # Assign a color based on the label; flip the color with probability e
            colors = torch.logical_xor(labels, torch.bernoulli(float(e)*torch.ones_like(labels))).float()
            # Apply the color to the image by zeroing out the other color channel
            cimages = torch.cat([images[d::2], images[d::2]], dim=1)
            cimages[torch.arange(len(cimages)), (1-colors).long(), :, :] *= 0
            metaarray = torch.stack([d*torch.ones_like(labels), 2*torch.ones_like(labels), digits[d::2], labels, identity, d*10000+identity], dim=1)
            mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
            mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
            metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
            mnist_split_list = torch.cat([mnist_split_list, 2*torch.ones_like(labels)], dim=0)
        
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
        metaarray = torch.stack([2*torch.ones_like(labels), 3*torch.ones_like(labels), digits, labels, identity, identity], dim=1)
        mnist_x_list = torch.cat([mnist_x_list, cimages], dim=0)
        mnist_y_list = torch.cat([mnist_y_list, labels], dim=0)
        metaarray_list = torch.cat([metaarray_list, metaarray], dim=0)
        mnist_split_list = torch.cat([mnist_split_list, 3*torch.ones_like(labels)], dim=0)

        return mnist_split_list, mnist_x_list, mnist_y_list.long(), metaarray_list

    def get_input(self, idx):
        return self._data[idx]


class CFColoredMNIST(Dataset):
    _dataset_name = "CF-ColorMNIST"
    
    def __init__(
        self, 
        num_pairs: int,
        version: str = None, 
        root_dir: str = "data", 
        download: bool = False
    ):
        # Dataset information
        self._version: Optional[str] = version
        self._original_resolution = (2, 28, 28)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._data_dir: str = Path(root_dir)
        self._n_classes = 2
        self.invariant = 0.75   # 0.75 if original dataset
        self.num_pairs = int(num_pairs)
        self.train_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=self.num_pairs, shuffle=False)
        self._data, self._data_cf, self._labels = self._get_data()

        
    def _get_data(self):
        for x, y in self.train_loader:
            images = x
            digits = y
            break
        labels = (digits < 5).float()
        labels = torch.logical_xor(labels, torch.bernoulli(self.invariant*torch.ones_like(labels))).long()
        # Apply the color to the image by zeroing out the other color channel
        zero_channel = torch.zeros_like(images)
        image = torch.cat([images, zero_channel], dim=1) 
        image_cf = torch.cat([zero_channel, images], dim=1)
        return image, image_cf, labels
    
    def __getitem__(self, idx):
        return self._data[idx], self._data_cf[idx], self._labels[idx]

    def __len__(self):
        return self.num_pairs