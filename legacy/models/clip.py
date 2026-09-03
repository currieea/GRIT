from torchvision import transforms
import torch
import clip
from torch.nn import functional as F


class Clip(torch.nn.Module):
    def __init__(self, hparam):
        super(Clip, self).__init__()    
        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
        self.in_shape = hparam['input_shape']
        self.model, _ = clip.load("ViT-B/32")
        self.out_shape = 512

    def forward(self, x):
        if self.in_shape[0] == 2:
            x = torch.cat([x, torch.zeros_like(x[:,:1,:,:])], dim=1)
            # Upsample to 224x224 using bilinear interpolation
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Normalize the image
            x = self.normalize(x)
        elif self.in_shape[0] == 1:
            x = torch.cat([x, x, x], dim=1)
            # Upsample to 224x224 using bilinear interpolation
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

            # Normalize the image
            x = self.normalize(x)


        # Forward pass through the CLIP model
        features = self.model.encode_image(x)

        return features.to(torch.float32)