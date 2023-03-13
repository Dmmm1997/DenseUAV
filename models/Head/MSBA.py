import torch
import torch.nn as nn

class MSBA(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        
        self.opt = opt

    def forward(self, image):
        features = self.backbne.forward_features
        return features