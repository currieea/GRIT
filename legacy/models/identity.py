import torch.nn as nn

class Identity(nn.Module):  
    def __init__(self, hparam):
        super().__init__()
        self.hparam = hparam
        self.out_shape = self.hparam['input_shape']

    def forward(self, x):
        return x