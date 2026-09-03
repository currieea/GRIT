import torch


class Classifier(torch.nn.Module):
    def __init__(self, in_features, out_features, is_nonlinear=False):
        super(Classifier, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, out_features)
            )
    
    def forward(self, x):
        return self.model(x)