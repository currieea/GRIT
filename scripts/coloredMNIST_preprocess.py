import torch
import torchvision
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Third-Party Libraries
from torchvision import transforms
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_train_loader

# WILDS Library
from wilds.common.grouper import CombinatorialGrouper

# Custom Datasets
from datasets import ColoredMNISTDataset

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


train_transform = transforms.Compose(
    [
        transforms.Resize(224, interpolation=Image.BICUBIC),
        transforms.CenterCrop((224, 224)),
        lambda image: image.convert("RGB"),  # _convert_image_to_rgb
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ]
)


# CHANGE TO THIS
new_dir = "./data/ColoredMNIST-cf-clip_v1.0/"
root_dir = "./data/"
os.makedirs(new_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparam = {"input_shape": (2, 28, 28)}
preprocessor = Clip(hparam).to(device)
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root_dir, train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=250,
    shuffle=False,
)
oracle_z = []
oracle_z_prime = []
oracle_pair_ids = []
processed = 0
for x, y in tqdm(train_loader, total=200):
    remaining = 50000 - processed
    if remaining <= 0:
        break
    if x.shape[0] > remaining:
        x = x[:remaining]

    with torch.no_grad():
        images = x
        empty_tensor = torch.zeros_like(images)
        r_images = torch.cat((images, empty_tensor), dim=1)
        g_images = torch.cat((empty_tensor, images), dim=1)

        r_latent = preprocessor(r_images.to(device))
        g_latent = preprocessor(g_images.to(device))
        batch_size = x.shape[0]
        oracle_z.append(r_latent.to("cpu"))
        oracle_z_prime.append(g_latent.to("cpu"))
        oracle_pair_ids.append(
            torch.arange(processed, processed + batch_size, dtype=torch.long)
        )
        processed += batch_size

oracle_z = torch.cat(oracle_z)
oracle_z_prime = torch.cat(oracle_z_prime)
oracle_pair_ids = torch.cat(oracle_pair_ids)
assert oracle_z.shape == oracle_z_prime.shape
assert torch.equal(oracle_pair_ids, torch.arange(len(oracle_pair_ids), dtype=torch.long))

diff_tensor = oracle_z - oracle_z_prime
torch.save(diff_tensor, new_dir + "diff.pth")
torch.save(oracle_z, new_dir + "oracle_z_array.pth")
torch.save(oracle_z_prime, new_dir + "oracle_z_prime_array.pth")
torch.save(oracle_pair_ids, new_dir + "oracle_pair_id_array.pth")

dataset = ColoredMNISTDataset(root_dir=root_dir, download=True)
# new_dataset = TransformedWildsDataset(dataset, transform=None)

dataloader = DataLoader(
    dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True
)

new_x = []
for x, _, _ in tqdm(dataloader, desc="Preprocessing", total=len(dataloader)):
    x = x.to(device)
    with torch.no_grad():
        clip_x = preprocessor(x)
        new_x.append(clip_x.detach().to("cpu"))

new_x = torch.cat(new_x, dim=0)
torch.save(dataset._split_array, new_dir + "split_array.pth")
torch.save(new_x, new_dir + "x_array.pth")
torch.save(dataset._y_array, new_dir + "y_array.pth")
torch.save(dataset._metadata_array, new_dir + "metadata_array.pth")
