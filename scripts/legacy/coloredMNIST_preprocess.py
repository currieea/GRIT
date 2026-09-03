from PIL import Image
import torch

from torch.utils.data import Dataset, DataLoader

# Third-Party Libraries
from torchvision import transforms

# WILDS Library
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.data_loaders import get_train_loader

# Custom Datasets
from datasets import ColoredMNISTDataset
import torchvision
# Custom Models
from models import *
from tqdm.auto import tqdm
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


new_dir = "/local/scratch/a/bai116/datasets/ColoredMNIST-cf-clip_v1.0/"
root_dir = "/local/scratch/a/bai116/datasets/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparam = {"input_shape": (2, 28, 28)}
preprocessor = Clip(hparam).to(device)
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root_dir, train=True, download=True, transform=transforms.ToTensor()), batch_size=250, shuffle=False)
diff_tensor = []
count = 0.
# for x, y in tqdm(train_loader, total=200):
#     with torch.no_grad():
#         images = x
#         empty_tensor = torch.zeros_like(images)
#         r_images = torch.cat((images, empty_tensor), dim=1)
#         g_images = torch.cat((empty_tensor, images), dim=1)
        
#         r_latent = preprocessor(r_images.to(device))
#         g_latent = preprocessor(g_images.to(device))
#         cf_diff = r_latent - g_latent
#         diff_tensor.append(cf_diff.to('cpu'))
#         count += x.shape[0]
#     if count >= 50000:
#         break
# diff_tensor = torch.cat(diff_tensor)
# torch.save(diff_tensor, new_dir + "diff.pth")  

dataset = ColoredMNISTDataset(root_dir=root_dir, download=False)
# new_dataset = TransformedWildsDataset(dataset, transform=None)

dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

new_x = []
for x, _, _ in tqdm(dataloader, desc="Preprocessing", total=len(dataloader)):
    x = x.to(device)
    with torch.no_grad():
        clip_x = preprocessor(x)
        new_x.append(clip_x.detach().to('cpu'))

new_x = torch.cat(new_x, dim=0)
torch.save(dataset._split_array, new_dir + "split_array.pth")
torch.save(new_x, new_dir + "x_array.pth")
torch.save(dataset._y_array, new_dir + "y_array.pth")
torch.save(dataset._metadata_array, new_dir + "metadata_array.pth")
