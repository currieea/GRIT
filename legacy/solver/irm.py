
import torch
import torch.autograd as autograd


# Third-Party Libraries
import wandb
from tqdm.auto import tqdm

# WILDS Library
from wilds.common.utils import split_into_groups
 
from solver import ERM

class IRM(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.penalty_anneal_iters = self.hparam["param2"]
        self.penalty_weight = self.hparam["param1"]
        self.update_count = 0.
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                outputs = self.model(x)
                penalty = 0.
                loss = self.criterion(outputs*self.scale, y_true)

                penalty = self.irm_penalty(loss)
                if self.update_count >= self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight
                else:
                    penalty_weight = self.update_count / self.penalty_anneal_iters
                avg_loss = loss.mean()
                obj = avg_loss + penalty_weight * penalty
                with torch.no_grad():
                    total_celoss += avg_loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (avg_loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_count += 1
            logs = {"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}
            if self.hparam['wandb']:
                wandb.log(logs, step=i)
            else:
                print(logs)
            self.evaluate(i)


    def irm_penalty(self, loss):
        loss_1 = loss[:len(loss)//2].mean()
        loss_2 = loss[len(loss)//2:].mean()
        grad_1 = autograd.grad(loss_1, [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [self.scale], create_graph=True)[0]
        return (torch.sum(grad_1 ** 2) + torch.sum(grad_2 ** 2)) / 2

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


class REx(IRM):
    def irm_penalty(self, loss):
        mean = loss.mean()
        penalty = ((loss - mean) ** 2).mean()
        return penalty