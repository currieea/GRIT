import copy
import torch

# Third-Party Libraries
import wandb
from tqdm.auto import tqdm

# WILDS Library

from wilds.common.utils import split_into_groups
from solver import ERM


# Custom Models
from models import *

# Custom Utilities
from utils import ParamDict


class Fish(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.meta_lr = hparam["param1"]
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            self.model.train()
            for x, y_true, metadata in tqdm(self.train_loader):
                param_dict = ParamDict(copy.deepcopy(self.model.state_dict()))
            
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                unique_groups, group_indices, _ = split_into_groups(g)
                for i_group in group_indices: # Each element of group_indices is a list of indices
                    # print(i_group)
                    group_loss = self.criterion(self.model(x[i_group]), y_true[i_group]).mean()   
                    total_loss += group_loss * len(i_group)
                    if group_loss.grad_fn is None:
                        # print('jump')
                        pass
                    else:
                        group_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                new_param_dict = param_dict + self.meta_lr * (ParamDict(self.model.state_dict()) - param_dict)
                self.model.load_state_dict(new_param_dict)

            logs = {"training_loss": total_loss / len(self.train_set)}
            if self.hparam['wandb']:
                wandb.log(logs, step=i)
            else:
                print(logs)
            self.evaluate(i)

   
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return self.dataset.default_domain_fields
    
    @property
    def uniform_over_groups(self):
        return True

    @property
    def n_groups_per_batch(self):
        return 2