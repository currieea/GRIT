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

class SWAD(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.tol_rate = hparam['param1']
        self.optimum_patience = 3
        self.overfit_patience = 6
        self.iid_max_acc = 0.0
        self.swa_max_acc = 0.0

    def fit(self):
        loss_last = None
        avgmodel = None
        avg_loss = 0
        n_iter = 0
        s_avgmodel = None
        s_avg_loss = 0
        s_iter = 0

        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x) 
                loss = self.criterion(outputs, y_true).mean()        
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # swad

                with torch.no_grad():
                    val_loss_single_iter = 0
                    if 'in_val' in self.eval_loader_dict.keys():
                        for x, y_true, _ in self.eval_loader_dict['in_val']:
                            x = x.to(self.device)
                            y_true = y_true.to(self.device)
                            metadata = metadata.to(self.device)
                            outputs = self.model(x)
                            val_loss_single_iter += self.criterion(outputs, y_true).sum()
                        val_loss_single_iter /= len(self.eval_set_dict['in_val'])
                    elif 'in_test' in self.eval_loader_dict.keys():
                        for x, y_true, _ in self.eval_loader_dict['in_test']:
                            x = x.to(self.device)
                            y_true = y_true.to(self.device)
                            metadata = metadata.to(self.device)
                            outputs = self.model(x)
                            val_loss_single_iter += self.criterion(outputs, y_true).sum()
                        val_loss_single_iter /= len(self.eval_set_dict['in_test'])
                    elif 'train' in self.eval_loader_dict.keys() and 'waterbirds' in self.dataset.name.lower():
                        for x, y_true, _ in self.eval_loader_dict['train']:
                            x = x.to(self.device)
                            y_true = y_true.to(self.device)
                            metadata = metadata.to(self.device)
                            outputs = self.model(x)
                            val_loss_single_iter += self.criterion(outputs, y_true).sum()
                        val_loss_single_iter /= len(self.eval_set_dict['train'])
                if val_loss_single_iter < avg_loss * self.tol_rate:
                    # merge back
                    if s_avgmodel is not None:
                        # if the current loss is smaller than the average loss, update the swa model.
                        avg_loss = (avg_loss * n_iter + s_avg_loss) / (n_iter + 1)
                        avgmodel = (avgmodel * n_iter + s_avgmodel * s_iter) / (n_iter + s_iter)
                        s_avgmodel = None
                        s_iter = 0
                
                if loss_last is None or val_loss_single_iter < loss_last and n_iter < self.optimum_patience:
                    # a new potential i.
                    loss_last = val_loss_single_iter
                    avg_loss = loss_last
                    avgmodel = ParamDict(self.model.state_dict())
                    n_iter = 1
                
                elif val_loss_single_iter > avg_loss * self.tol_rate and s_iter < self.overfit_patience:
                    if s_avgmodel is None:
                        s_avg_loss = val_loss_single_iter
                        s_avgmodel = ParamDict(self.model.state_dict())
                        s_iter += 1
                    else:
                        s_avg_loss = (s_avg_loss * s_iter + val_loss_single_iter) / (s_iter + 1)
                        s_avgmodel = s_iter * s_avgmodel + ParamDict(self.model.state_dict())
                        s_iter += 1
                    if s_iter == self.overfit_patience:
                        self.model.load_state_dict(avgmodel.to(self.device))
                        break

                elif val_loss_single_iter >= loss_last:
                    # if the loss is not improving, increase the n_iter.
                    loss_last = val_loss_single_iter
                    avg_loss = (avg_loss * n_iter + val_loss_single_iter) / (n_iter + 1)
                    avgmodel = n_iter * avgmodel + ParamDict(self.model.state_dict())
                    n_iter += 1
            else:
                self.model.load_state_dict(avgmodel.to(self.device))
            # here. Run some debugger to check the loss.

            logs = {"training_loss": total_loss / len(self.train_set)}
            if self.hparam['wandb']:
                wandb.log(logs, step=i)
            else:
                print(logs)
            self.evaluate(i)

