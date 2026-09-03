import numpy as np
from PIL import Image
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import copy

# Third-Party Libraries
import wandb
from tqdm.auto import tqdm
from torchvision import transforms

# WILDS Library
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.utils import split_into_groups
from wilds.common.grouper import CombinatorialGrouper


# Custom Datasets
from datasets import CounterfactualCelebADataset

# Custom Models
from models import *

# Custom Utilities
from utils import ParamDict

class TransformedWildsDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y, metadata = self.dataset[idx]
        x = self.transform(x)
        return x, y, metadata

    def __len__(self):
        return len(self.dataset)


train_transform = transforms.Compose([
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224, 224)),
        lambda image: image.convert("RGB"),  # _convert_image_to_rgb
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])


new_dir = "/local/scratch/a/bai116/datasets/counterfactual-CelebA-clip_v1.0/"
root_dir = "/local/scratch/a/bai116/datasets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparam = {"input_shape": (3, 224, 224)}
preprocessor = Clip(hparam).to(device)
dataset = CounterfactualCelebADataset(root_dir=root_dir, download=False)
new_dataset = TransformedWildsDataset(dataset, transform=train_transform)
dataloader = DataLoader(new_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
cf_dataset = dataset.get_subset('counterfactual', transform=train_transform)
grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=['cf',])
with torch.no_grad():
    cf_pair, _, _ = next(iter(get_train_loader('group', cf_dataset, batch_size=len(cf_dataset), num_workers=4, pin_memory=True, uniform_over_groups=None, grouper=grouper, distinct_groups=False, n_groups_per_batch=len(cf_dataset)//2)))
    cf_z = preprocessor(cf_pair.to(device))
    cf_diff = cf_z[0::2] - cf_z[1::2]
    torch.save(cf_diff.to('cpu'), new_dir + "diff.pth")  


# new_x = []
# for x, _, _ in tqdm(dataloader, desc="Preprocessing", total=len(dataloader)):
#     x = x.to(device)
#     with torch.no_grad():
#         clip_x = preprocessor(x)
#         new_x.append(clip_x.detach().to('cpu'))

# new_x = torch.cat(new_x, dim=0)
# torch.save(new_x, new_dir + "x_array.pth")
# torch.save(dataset._y_array, new_dir + "y_array.pth")
# torch.save(dataset._metadata_array, new_dir + "metadata_array.pth")
        
