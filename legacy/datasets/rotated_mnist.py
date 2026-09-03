from datasets import WILDSCFDataset
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
import os
import numpy as np
import copy
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from itertools import permutations, combinations_with_replacement
import random
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper


class RotatedMNISTDataset(WILDSCFDataset):
    _dataset_name = "RotatedMNIST"
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
        self._original_resolution = (1, 28, 28)
        self.input_shape = (1, 28, 28)
        self._y_type: str = "long"
        # Path of the dataset
        self._data_dir: str = Path(root_dir)
        self._n_classes = 10
        self._angle_to_domain = {"0": 0, "15": 1, "30": 2, "45": 3, "60": 4, "75": 5, "90": 6}
        self._split_list = ['train', 'in_test', 'test']
        self._split_dict = {'train': 0, 'in_test': 1, 'test': 2}
        self._split_names = {'train': 'Train', 'in_test': 'Test (In-Domain)', 'test': 'Test (OOD/Trans)'}
        self._metadata_fields = ["domain","angle", "y", "id", "idx"]
        self._split_array, self._x_array, self._y_array, self._metadata_array = self._get_data()
        self.default_domain_fields = ['domain']
        self.default_optimizer = 'adam'
        self.counterfactual_fields = ['id']
        self._eval_grouper = CombinatorialGrouper(dataset=self, groupby_fields=self.default_domain_fields)
        # super().__init__(root_dir, download, split_scheme)
    def _get_data(self):
        # split scheme is the test domain rotation angle. We always test on test dataset.
        if self.split_scheme == "official":
            self._training_domains = ["0", "45"]
            self._in_test_domain = self._training_domains
            self._test_domains = ["90"]
        elif self.split_scheme == "oracle":
            self._training_domains = ["90"]
            self._in_test_domain = self._training_domains
            self._test_domains = ["90"]
        elif self.split_scheme == "simple":
            self._training_domains = ["0", "15", "30", "45", "60"]
            self._in_test_domain = self._training_domains
            self._test_domains= ["75"]
        elif self.split_scheme == "matchdg_hard":
            self._training_domains = ["30", "45"]
            self._in_test_domain = self._training_domains
            self._test_domains = ["0", "90"]
        elif self.split_scheme == "matchdg_medium":
            self._training_domains = ["30", "45", "60"]
            self._in_test_domain = self._training_domains
            self._test_domains = ["0", "90"]
        else: 
            raise NotImplementedError
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)

        for x, y in train_loader:
            mnist_imgs = x
            mnist_labels = y
        mnist_x_list = []
        mnist_y_list = []
        metaarray_list = []
        mnist_split_list = []
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        idx = 0
        for angle in self._training_domains:
            for i in range(len(mnist_imgs)):
                if angle == '0':
                    mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
                else:
                    mnist_x_list.append(to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(angle))))
                d = self._angle_to_domain[angle]
                mnist_y_list.append(mnist_labels[i].item())
                metaarray_list.append(torch.tensor([d, int(angle), mnist_labels[i].item(), i, idx]))
                mnist_split_list.append(0)
                idx += 1

        test_loader = torch.utils.data.DataLoader(datasets.MNIST(self._data_dir, train=False, download=True, transform=transforms.ToTensor()), batch_size=10000, shuffle=False)
        for x, y in test_loader:
            mnist_imgs = x
            mnist_labels = y
        for i in range(len(mnist_imgs)):
            for angle in self._in_test_domain:
                if angle == '0':
                    mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
                else:
                    mnist_x_list.append(to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(angle))))
                d = self._angle_to_domain[angle]
                mnist_y_list.append(mnist_labels[i].item())
                metaarray_list.append(torch.tensor([d, int(angle), mnist_labels[i].item(), i, idx]))
                mnist_split_list.append(1)
                idx += 1

        
        for i in range(len(mnist_imgs)):
            for angle in self._test_domains:
                if angle == '0':
                    mnist_x_list.append(to_tensor(to_pil(mnist_imgs[i])))
                else:
                    mnist_x_list.append(to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(angle))))
                d = self._angle_to_domain[angle]
                mnist_y_list.append(mnist_labels[i].item())
                metaarray_list.append(torch.tensor([d, int(angle), mnist_labels[i].item(), i, idx]))
                mnist_split_list.append(2)
                idx += 1

        # Stack
        img_array = torch.cat(mnist_x_list)
        y_array = torch.tensor(mnist_y_list)
        meta_array = torch.stack(metaarray_list, dim=0)
        split_array = torch.tensor(mnist_split_list)
    

        return split_array, img_array.unsqueeze(1), y_array, meta_array

    def get_input(self, idx) -> str:
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


class RotatedMNISTClipDataset(RotatedMNISTDataset):
    _dataset_name = "RotatedMNIST-cf-clip"
    _versions_dict = {
        '1.0': {
            "download_url": "https://worksheets.codalab.org/rest/bundles/0x19f5d1758a184e13aeaea05e0954422a/contents/blob/",
            "compressed_size": "171_612_540"
        }
    }
    def __init__(self, version = None, root_dir = "data", download = False, split_scheme = "official"):
        super().__init__(version, root_dir, download, split_scheme)

        self.diff = torch.load(os.path.join(self._data_dir, 'diff.pth'))
    def _get_data(self):
        data_dir =  self._data_dir = self.initialize_data_dir(self._data_dir, download=False)
        return torch.load(os.path.join(data_dir, 'split_array.pth')), torch.load(os.path.join(data_dir, 'x_array.pth')), torch.load(os.path.join(data_dir, 'y_array.pth')), torch.load(os.path.join(data_dir, 'metadata_array.pth'))
