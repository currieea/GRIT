import torch
import torch.nn as nn


# Third-Party Libraries
import wandb
from tqdm.auto import tqdm


# WILDS Library
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss


# Custom Utilities
from solver import ERM


class GroupDRO(ERM):
    """
    Group distributionally robust optimization.

    Original paper:
        @inproceedings{sagawa2019distributionally,
          title={Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization},
          author={Sagawa, Shiori and Koh, Pang Wei and Hashimoto, Tatsunori B and Liang, Percy},
          booktitle={International Conference on Learning Representations},
          year={2019}
        }    
    """
    def __init__(self, hparam):
        # initialize model
        super(GroupDRO, self).__init__(hparam)
        # step size
        self.group_weights_step_size = self.hparam['param1'] # config.group_dro_step_size
        # initialize adversarial weights
        self.group_weights = torch.zeros(self.grouper.n_groups, device=self.device)
        train_g = self.grouper.metadata_to_group(self.train_set.metadata_array)
        unique_groups, unique_counts = torch.unique(train_g, sorted=False, return_counts=True)
        counts = torch.zeros(self.grouper.n_groups, device=train_g.device)
        counts[unique_groups] = unique_counts.float()
        is_group_in_train = counts > 0
        self.group_weights[is_group_in_train] = 1
        self.group_weights = self.group_weights/self.group_weights.sum()
        self.loss = ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    

    def fit(self):
        for i in range(int(self.hparam['epochs'])):
            total_loss = 0.
            self.model.train()
            for x, y, meta in tqdm(self.train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                g = self.grouper.metadata_to_group(meta).to(self.device)
                meta = meta.to(self.device)
                y_pred = self.model(x)
                group_losses, _, _ = self.loss.compute_group_wise(y_pred, y, g, self.grouper.n_groups, return_dict=False)
                
                loss = group_losses @ self.group_weights
                
                self.optimizer.zero_grad()
                loss.backward()
                self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
                self.group_weights = (self.group_weights/(self.group_weights.sum()))
                self.optimizer.step()
                with torch.no_grad():
                    total_loss += loss.item() * x.size(0)
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