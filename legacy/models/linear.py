import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, input_shape, latent_dim=512):
        super(Linear, self).__init__()
        self.model = nn.Linear(input_shape, latent_dim)
    
    def forward(self, x):
        return self.model(x.view(x.size(0), -1))