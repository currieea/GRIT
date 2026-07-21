import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor
from typing import List, Any
from abc import abstractmethod
import torch
from torch.nn import functional as F
import math
import clip 
from torchvision import transforms

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape, latent_dim=128):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.fc = nn.Linear(128, latent_dim)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.activation = nn.Identity() # for URM; does not affect other algorithms

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return self.fc(self.activation(x))
    
class ResNet18(nn.Module):
    def __init__(self, input_shape, latent_dim=1024):
        super(ResNet18, self).__init__()
        self.model = resnet18(num_classes=latent_dim)
        self.model.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
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


class Clip(nn.Module):
    def __init__(self, hparam):
        super(Clip, self).__init__()    
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.in_shape = hparam['image_shape']
        self.model, _ = clip.load("ViT-B/32")
        self.out_shape = 512


    def forward(self, x):
        if self.in_shape[0] == 2:
            x = torch.cat([x, torch.zeros_like(x[:,:1,:,:])], dim=1)
            # Upsample to 224x224 using bilinear interpolation
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Normalize the image
            x = self.normalize(x)

        # Forward pass through the CLIP model
        features = self.model.encode_image(x)

        return features.to(torch.float32)


class Flattener(nn.Module):
    def __init__(self, hparam):
        super(Flattener, self).__init__()
        self.hparam = hparam
        self.out_shape = math.prod(self.hparam['image_shape'])
        
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):  
    def __init__(self, hparam):
        super().__init__()
        self.hparam = hparam
        self.out_shape = self.hparam['image_shape']

    def forward(self, x):
        return x