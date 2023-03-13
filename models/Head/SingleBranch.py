import torch
import torch.nn as nn
from .utils import ClassBlock

class SingleBranch(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)

    def forward(self, features):
        global_feature = features[:,0]
        cls, feature = self.classifier(global_feature)
        return [cls,feature]
    

class SingleBranchCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)

    def forward(self, features):
        global_feature = self.pool(features).reshape(features.shape[0],-1)
        cls, feature = self.classifier(global_feature)
        return [cls,feature]
    

class SingleBranchSwin(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)

    def forward(self, features):
        global_feature = self.pool(features.transpose(2,1)).reshape(features.shape[0],-1)
        cls, feature = self.classifier(global_feature)
        return [cls,feature]


    

