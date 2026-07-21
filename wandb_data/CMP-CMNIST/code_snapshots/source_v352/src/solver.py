import torch
import torch.autograd as autograd
from torch.utils.data import DataLoader

from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.utils import split_into_groups
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss


import wandb
import einops
import copy
from tqdm.auto import tqdm

from src.datasets import ColoredMNIST, CFColoredMNIST
from src.models import *
from src.utils import ParamDict


class ERM(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        # initialize model
        if self.hparam['mode'] == 0:
            self._preprocessor = Flattener(self.hparam).to(self.device)
            self._featurizer = Linear(input_shape=self._preprocessor.out_shape, latent_dim=self.hparam['latent_dim'])
            self._classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=self.hparam['num_classes'])
        elif self.hparam['mode'] == 1:
            self._preprocessor = Identity(self.hparam).to(self.device)
            self._featurizer = MNIST_CNN(input_shape=self._preprocessor.out_shape, latent_dim=self.hparam['latent_dim'])
            self._classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=self.hparam['num_classes'])
        elif self.hparam['mode'] == 2:
            self._preprocessor = Clip(self.hparam).to(self.device)
            self._featurizer = Linear(input_shape=self._preprocessor.out_shape, latent_dim=self.hparam['latent_dim'])
            self._classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=self.hparam['num_classes'])
        
        self._model = torch.nn.Sequential(self._featurizer, self._classifier).to(self.device)
        self.model = torch.nn.Sequential(self._preprocessor, self._featurizer, self._classifier).to(self.device)
        self.featurizer = torch.nn.Sequential(self._preprocessor, self._featurizer).to(self.device)
        
        # initialize dataset
        self.dataset = ColoredMNIST(root_dir=self.hparam["root_dir"])
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=True, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.val_set = self.dataset.get_subset(split='val')
        self.val_loader = get_eval_loader(loader='standard', dataset=self.val_set, batch_size=self.hparam["batch_size"])
        self.in_test_set = self.dataset.get_subset(split='in_test')
        self.in_test_loader = get_eval_loader(loader='standard', dataset=self.in_test_set, batch_size=self.hparam["batch_size"])
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

        self.step = 0

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self._model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                self.step += 1
                x = x.to(self.device)  
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true)
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"training_loss": total_loss / len(self.train_set)}, step=self.step)
                    else:
                        print(total_loss / len(self.train_set))
                    self.evaluate(self.step)

    
    def evaluate(self, step):
        self._model.eval()
        val_corr = 0.
        for x,y_true,metadata in self.val_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            val_corr += torch.sum(torch.eq(y_pred, y_true))
        
        in_corr = 0.
        for x,y_true,metadata in self.in_test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            in_corr += torch.sum(torch.eq(y_pred, y_true))

        corr = 0.
        for x,y_true,metadata in self.test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            corr += torch.sum(torch.eq(y_pred, y_true))
        if self.hparam['wandb']:
            wandb.log({"val_acc": val_corr / len(self.val_set), "in_test_acc": in_corr / len(self.in_test_set), "test_acc": corr / len(self.test_set)}, step=step)
        else:
            print({"in_test_acc": in_corr / len(self.in_test_set), "test_acc": corr / len(self.test_set)})

    @property
    def loader_type(self):
        return 'standard'

    @property
    def domain_fields(self):
        return ['domain']

    @property
    def n_groups_per_batch(self):
        return 1

  
class IRM(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hparam['lr'], weight_decay=hparam['param3'])
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.penalty_anneal_iters = self.hparam["param2"]
        self.penalty_weight = self.hparam["param1"]
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self._model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            total_sample = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                self.step += 1
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                outputs = self.model(x)
                penalty = 0.
                loss = self.criterion(outputs*self.scale, y_true)

                penalty = self.irm_penalty(loss)
                if self.step > self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight
                else:
                    penalty_weight = self.step / self.penalty_anneal_iters
                avg_loss = loss.mean()
                obj = avg_loss + penalty_weight * penalty
                with torch.no_grad():
                    total_celoss += avg_loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (avg_loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                    total_sample += len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"CELoss": total_celoss.item() / total_sample, "penalty": total_penalty.item() / total_sample, "training_loss": total_loss / total_sample}, step=self.step)
                    else:
                        print(total_loss / total_sample)
                    self.evaluate(self.step)


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
        return ['domain']
    
    @property
    def n_groups_per_batch(self):
        return 2

    def form_group(self, group_indices):
        return group_indices
        

class REx(IRM):
    def irm_penalty(self, loss):
        mean = loss.mean()
        penalty = ((loss - mean) ** 2).mean()
        return penalty


class CMP(ERM):
    """
    param 1: number of fine tune sample.
    param 2: penalty for the alignment
    param 3: penalty for the fine-tune set weight.
    """
    def __init__(self, hparam):
        super().__init__(hparam)
        self.cf_dataset = CFColoredMNIST(root_dir=self.hparam["root_dir"], num_pairs=self.hparam["param2"])
        self.cf_dataloader = iter(DataLoader(self.cf_dataset, batch_size=int(self.hparam["fewshot_batch_size"]), shuffle=True))

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            total_loss = 0.
            erm_loss = 0.
            cf_loss = 0.
            self._model.train()
            for x, y_true, _ in tqdm(self.train_loader):
                self.step += 1
                batch_size = x.shape[0]
                try:
                    img, img_cf, labels = next(self.cf_dataloader)
                except StopIteration:
                    self.cf_dataloader = iter(DataLoader(self.cf_dataset, batch_size=int(self.hparam["fewshot_batch_size"]), shuffle=True))
                    img, img_cf, labels = next(self.cf_dataloader)
            
                x = torch.cat((x, img, img_cf)).to(self.device)
                y_true = torch.cat((y_true, labels, labels)).to(self.device)
                features = self.featurizer(x)
                outputs = self._classifier(features)
                feature_per_domain = features[batch_size:].reshape(len(self.dataset._training_domains), -1, self.hparam['latent_dim'])
                feature_mean = feature_per_domain.mean(dim=0, keepdim=True)
                loss_2 = ((feature_per_domain-feature_mean) ** 2).sum() * 2 / feature_per_domain.shape[1]
                loss = self.criterion(outputs[:batch_size], y_true[:batch_size])    
                obj = loss + self.hparam['param1'] * loss_2
                with torch.no_grad():
                    total_loss += obj.item() * len(y_true)
                    erm_loss += loss.item() * len(y_true)
                    cf_loss += loss_2.item() * len(y_true)
                self.optimizer.zero_grad()
                obj.backward()
                self.optimizer.step()
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"train_loss": total_loss / len(self.train_set), "erm_loss": erm_loss / len(self.train_set), "cf_loss": cf_loss / len(self.train_set)}, step=self.step)
                    else:
                        print({"train_loss": total_loss / len(self.train_set)})
                    self.evaluate(self.step)

    @property
    def loader_type(self):
        return 'standard'

    @property
    def domain_fields(self):
        return ['domain']

    @property
    def n_groups_per_batch(self):
        return 1


class Fish(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.meta_lr = hparam["param1"]
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['domain']
    
    @property
    def n_groups_per_batch(self):
        return 2

    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            total_loss = 0.
            self._model.train()
            for x, y_true, metadata in tqdm(self.train_loader):
                self.step += 1
                param_dict = ParamDict(copy.deepcopy(self._model.state_dict()))
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                unique_groups, group_indices, _ = split_into_groups(g)
                for i_group in group_indices: # Each element of group_indices is a list of indices
                    # print(i_group)
                    group_loss = self.criterion(self.model(x[i_group]), y_true[i_group])
                    total_loss += group_loss * len(i_group)
                    if group_loss.grad_fn is None:
                        # print('jump')
                        pass
                    else:
                        group_loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                param_dict = param_dict + self.meta_lr * (ParamDict(self._model.state_dict()) - param_dict)
                self._model.load_state_dict(copy.deepcopy(param_dict))
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"training_loss": total_loss / len(self.train_set)}, step=self.step)
                    else:
                        print(total_loss / len(self.train_set))
                    self.evaluate(self.step)


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
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=self.hparam['lr'], weight_decay=hparam['param2'])
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_loader = get_train_loader('group', self.train_set, batch_size=self.hparam['batch_size'], num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=self.grouper, distinct_groups=False, n_groups_per_batch=self.n_groups_per_batch)
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
            self._model.train()
            for x, y, meta in tqdm(self.train_loader):
                self.step += 1
                x = x.to(self.device)
                y = y.to(self.device)
                g = self.grouper.metadata_to_group(meta).to(self.device)
                meta = meta.to(self.device)
                y_pred = self.model(x)
                group_losses, _, _ = self.loss.compute_group_wise(y_pred, y, g, self.grouper.n_groups, return_dict=False)
                
                loss = group_losses @ self.group_weights
                with torch.no_grad():
                    total_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.group_weights = self.group_weights * torch.exp(self.group_weights_step_size*group_losses.data)
                self.group_weights = (self.group_weights/(self.group_weights.sum()))
                self.optimizer.step()
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"training_loss": total_loss / len(self.train_set)}, step=self.step)
                    else:
                        print(total_loss / len(self.train_set))
                    self.evaluate(self.step)


    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['domain']
    
    @property
    def n_groups_per_batch(self):
        return 2



class ECMP(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.cf_dataset = self.dataset.get_subset('counterfactual', transform=self.train_transform)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['cf',])
        cf_pair, _, _ = next(iter(get_train_loader('group', self.cf_dataset, batch_size=len(self.cf_dataset), num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=self.grouper, distinct_groups=False, n_groups_per_batch=len(self.cf_dataset)//2)))
        cf_z = self._preprocessor(cf_pair.to(self.device))
        cf_diff = cf_z[0::2] - cf_z[1::2]
        
        U, _,_ = self.truncated_svd(cf_diff.T, int(self.hparam['param2']))
        self.projection = torch.eye(cf_z.shape[1], device=self.device) - (U @ U.T)

    @staticmethod
    def truncated_svd(X: torch.Tensor, r: int):
        # Perform full SVD
        U, S, Vh = torch.linalg.svd(X, full_matrices=True)
        
        # Truncate to rank r
        U[:, r:] = 0
        S[r:] = 0 
        Vh[r:, :] = 0
        
        # # Reconstruct the truncated approximation
        # X_approx = U_r @ torch.diag(S_r) @ Vh_r
        
        return U, S, Vh
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self._model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                self.step += 1
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self._classifier(self._featurizer(self._preprocessor(x) @ self.projection))
                loss = self.criterion(outputs, y_true)
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.step % 20 == 0:
                    if self.hparam['wandb']:
                        wandb.log({"training_loss": total_loss / len(self.train_set)}, step=self.step)
                    else:
                        print(total_loss / len(self.train_set))
                    self.evaluate(self.step)