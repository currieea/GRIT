import argparse
import random
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn
import wandb
from solver import *

"""
The main file function:
1. Load the hyperparameter dict.
2. Initialize logger
3. Initialize data (preprocess, data splits, etc.)
4. Initialize clients. 
5. Initialize Server.
6. Register clients at the server.
7. Start the server.
"""
def main(args):
    hparam = vars(args)
    wandb_project = "CMP"
    # setup WanDB
    if not args.no_wandb:
        wandb.init(project=wandb_project,
                    entity='inouye-lab',
                    config=hparam)
        wandb.run.log_code()
    hparam['wandb'] = not args.no_wandb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparam['device'] = device
    seed = hparam['seed']
    set_seed(seed)
    solver = eval(hparam['solver'])(hparam)
    solver.fit()

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Waterbirds experiments')
    parser.add_argument('--no_wandb', default=False, action="store_true")
    parser.add_argument('--root_dir', default="/local/scratch/a/bai116/datasets/", action="store_true")
    parser.add_argument('--seed', default=1001, type=int)
    parser.add_argument('--dataset', type=str, default='ColoredMNIST')
    parser.add_argument('--latent_dim', default=512, type=int) # Not active for erm and ecmp.
    parser.add_argument('--pretrained', default="false", type=str, help='use pretrained model')
    parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for optimizer, default is 0.0. Set to 0.01 for regularization on some datasets')
    parser.add_argument('--split_scheme', default='official', type=str)
    parser.add_argument('--solver', default='ERM')
    parser.add_argument('--param1', default=100, type=float)
    parser.add_argument('--param2', default=0, type=float)
    parser.add_argument('--param3', default=0, type=float)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--upweighting', default='false', type=str)
    parser.add_argument('--featurizer', default="linear", type=str)
    parser.add_argument('--projection', default='oracle', type=str)
    args = parser.parse_args()
    main(args)
