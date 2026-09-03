import numpy as np
import torch
import torch.nn as nn

import math

# Third-Party Libraries
import wandb
from tqdm.auto import tqdm


# WILDS Library
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper


# Custom Datasets
from datasets import *

# Custom Models
from models import *


class ERM(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']

        if self.hparam['pretrained'] == 'true':
            self.dataset = eval(self.hparam['dataset'] + 'ClipDataset')(root_dir=self.hparam['root_dir'], download=False, split_scheme=self.hparam['split_scheme'])
        elif self.hparam['pretrained'] == 'false':
            self.dataset = eval(self.hparam['dataset'] + 'Dataset')(root_dir=self.hparam['root_dir'], download=False, split_scheme=self.hparam['split_scheme'])                
        else:
            raise ValueError("pretrained must be 'true' or 'false'")
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        if 'counterfactual' in self.dataset._split_dict.keys():
            self.train_set = self.dataset.get_subset(['train', 'counterfactual'], transform=self.dataset.train_transform)
        else:
            self.train_set = self.dataset.get_subset('train', transform=self.dataset.train_transform)
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], num_workers=4, pin_memory=True, uniform_over_groups=self.uniform_over_groups, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.eval_set_dict = {}
        self.eval_loader_dict = {}
        for subset in self.dataset._split_dict.keys():
            if subset != 'train' and subset != 'counterfactual':
                dataset = self.dataset.get_subset(subset, transform=self.dataset.eval_transform)
                if len(dataset) > 0:
                    self.eval_set_dict[subset] = dataset
                    self.eval_loader_dict[subset] = get_eval_loader(loader='standard', dataset=dataset, batch_size=self.hparam["batch_size"])
            
        self._initialize_model() # initialize the model
        
        if self.dataset.default_optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparam['lr'], momentum=0.9, weight_decay=self.hparam['weight_decay']) # Use momentum and weight decay for SGD
        elif self.dataset.default_optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'], weight_decay=self.hparam['weight_decay'])
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.best_id_log = {'in_test': {self.dataset.key_metric: -1}}
        self.best_val_log = {'val': {self.dataset.key_metric: -1}}
        self.best_oracle_log = {'test': {self.dataset.key_metric: -1}}


    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x)
                if self.hparam['upweighting'] == 'true':
                    weight = self.dataset.group_weight[metadata[:,0], metadata[:,1]] 
                    loss = torch.sum(self.criterion(outputs, y_true) * weight) / weight.sum()
                elif self.hparam['upweighting'] == 'false':
                    loss = self.criterion(outputs, y_true).mean()
                
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
    
    def evaluate(self, step):
        self.model.eval()
        logs = {}
        for subset, loader in self.eval_loader_dict.items():
            y_pred = None
            y_true = None
            metadata = None
            for x, label, meta_batch in loader:
                x = x.to(self.device)
                label = label.to(self.device)
                meta_batch = meta_batch.to(self.device)
                outputs = self.model(x)
                prediction = torch.argmax(outputs, dim=-1)
                if y_pred is None:
                    y_pred = prediction
                    y_true = label
                    metadata = meta_batch
                else:
                    y_pred = torch.cat([y_pred, prediction])
                    y_true = torch.cat([y_true, label])
                    metadata = torch.cat([metadata, meta_batch])
            metric = self.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
            logs[subset] = metric[0]
            
        self.report(logs, step) # Log evaluation metrics to wandb if enabled

        
    @property
    def loader_type(self):
        return 'standard'

    @property
    def uniform_over_groups(self):
        return False

    @property
    def domain_fields(self):
        return None

    @property
    def n_groups_per_batch(self):
        return None # Default to None for standard loader, can be overridden by subclasses if using group-based loaders


    def report(self, logs, step):
        if 'in_test' in logs:
            if self.best_id_log['in_test'][self.dataset.key_metric] < logs['in_test'][self.dataset.key_metric]:
                self.best_id_log = logs
        if 'val' in logs:
            if self.best_val_log['val'][self.dataset.key_metric] < logs['val'][self.dataset.key_metric]:
                self.best_val_log = logs
        if self.best_oracle_log['test'][self.dataset.key_metric] < logs['test'][self.dataset.key_metric]:
            self.best_oracle_log = logs
        if self.hparam['wandb']:
            wandb.log(logs, step=step)
        else:
            print(logs)
        if step == self.hparam['epochs'] - 1:
            # last epoch
            if self.hparam['wandb']:
                if 'in_test' in logs:
                    wandb.summary['in_test_val_best_in_test'] = self.best_id_log['in_test'][self.dataset.key_metric]
                    wandb.summary['in_test_val_best_test'] = self.best_id_log['test'][self.dataset.key_metric]
                    wandb.summary['test_val_best_in_test'] = self.best_oracle_log['in_test'][self.dataset.key_metric]
                wandb.summary['test_val_best_test'] = self.best_oracle_log['test'][self.dataset.key_metric]

            else:
                print("best_epoch_in_acc: ")
                print(self.best_id_log)
        
    def _initialize_model(self):
        if self.hparam['featurizer'] == 'linear':
            if self.hparam['pretrained']:
                self.model = Classifier(in_features=512, out_features=self.dataset._n_classes)
            else: 
                self.model = Classifier(in_features=math.prod(self.dataset.input_shape), out_features=self.dataset._n_classes)
        else:
            if 'MNIST' in self.dataset.dataset_name:
                self.featurizer = MNIST_CNN(input_shape=self.dataset.input_shape, latent_dim=self.hparam['latent_dim']).to(self.device)
            else:
                self.featurizer = ResNet50(input_shape=self.dataset.input_shape, latent_dim=self.hparam['latent_dim']).to(self.device)
                
            self.classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=self.dataset._n_classes).to(self.device)
            self.model = nn.Sequential(self.featurizer, self.classifier) 
        self.model = torch.nn.DataParallel(self.model).to(self.device)