import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch import Tensor
from typing import List, Any
from abc import abstractmethod
import torch

import math
from torchvision import transforms


class ResNet50(nn.Module):
    def __init__(self, input_shape=(3,224,224), latent_dim=1024):
        super(ResNet50, self).__init__()
        self.model = resnet50(num_classes=1000, weights=ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, latent_dim)
    
    def forward(self, x):
        return self.model(x)


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, is_nonlinear=False):
        super(Classifier, self).__init__()
        self.model = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.model(x)


class Linear(nn.Module):
    def __init__(self, input_shape, latent_dim=512):
        super(Linear, self).__init__()
        self.model = nn.Linear(input_shape, latent_dim)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class Flattener(nn.Module):
    def __init__(self, hparam):
        super(Flattener, self).__init__()
        self.hparam = hparam
        self.out_shape = math.prod(self.hparam['input_shape'])
        
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):  
    def __init__(self, hparam):
        super().__init__()
        self.hparam = hparam
        self.out_shape = self.hparam['input_shape']

    def forward(self, x):
        return x