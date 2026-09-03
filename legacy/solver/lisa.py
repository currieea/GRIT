import torch
import torch.autograd as autograd


# Third-Party Libraries
import wandb
from tqdm.auto import tqdm

# WILDS Library
from wilds.common.utils import split_into_groups
import copy
import torch
from torch.nn import Module
from copy import deepcopy

from solver import ERM
from utils import ParamDict

import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn

class LISA(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.selection_prob = self.hparam['param1']
        self.beta = torch.distributions.Beta(2, 2, validate_args=None)
        self.bernoulli = torch.distributions.Bernoulli(self.selection_prob)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                
                lmda = self.beta.sample([x.shape[0]]).to(self.device)
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)

                # LISA
 
                x_aug, y_aug, meta_aug = self.sample_lisa(metadata, intra=self.bernoulli.sample())              
                x_aug = x_aug.to(self.device)
                y_aug = y_aug.to(self.device)
                lmda_expanded_x = lmda.view([x.shape[0]] + [1] * (x.dim() - 1))
                lmda_expanded_y = lmda.view(y_true.shape[0],1)

                x = lmda_expanded_x * x + (1 - lmda_expanded_x) * x_aug
                y = lmda_expanded_y * F.one_hot(y_true, num_classes=self.dataset._n_classes) + (1 - lmda_expanded_y) * F.one_hot(y_aug, num_classes=self.dataset._n_classes)

                outputs = self.model(x)
               
                # loss = self.criterion(F.log_softmax(outputs, dim=1), y)
                loss = (- F.log_softmax(outputs, dim=-1) * y).mean()
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            logs = {"training_loss": total_loss / len(self.train_set)}
            if self.hparam['wandb']:
                wandb.log(logs, step=i)
            else:
                print(logs)
            self.evaluate(i)
    

    def sample_lisa(self, metadata, intra):
        all_metadata = self.dataset._metadata_array.to(self.device)

        n_sample = metadata.shape[0]
        y_col = self.dataset._metadata_fields.index('y')
        for d in self.domain_fields:
            if d != 'y':
                d_col = self.dataset._metadata_fields.index(d)
                break
        y = metadata[0, y_col]
        d = metadata[0, d_col]
        train_idx = torch.tensor(self.train_set.indices,device=self.device)
        train_mask = torch.zeros(len(self.dataset), dtype=torch.bool, device=self.device)
        train_mask[train_idx] = True
        if intra: # LISA-L: same y different d
            new_y = all_metadata[:, y_col] == y
            new_d = all_metadata[:, d_col] != d
        else:
            new_y = all_metadata[:, y_col] != y
            new_d = all_metadata[:, d_col] == d

        idx = torch.logical_and(torch.logical_and(new_y, new_d), train_mask).nonzero().squeeze()
        idx = idx[torch.randperm(len(idx))][:n_sample].to('cpu')
    
        return self.dataset[idx]       

   
    @property
    def loader_type(self):
        return 'group'

    
    @property
    def uniform_over_groups(self):
        return True

    @property
    def n_groups_per_batch(self):
        return 1
    
    @property
    def domain_fields(self):
        if 'y' in self.dataset.default_domain_fields:
            return self.dataset.default_domain_fields
        else:
            return self.dataset.default_domain_fields + ['y']