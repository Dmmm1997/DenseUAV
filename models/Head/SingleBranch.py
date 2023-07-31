import torch.nn as nn
from .utils import ClassBlock, Pooling
import torch.nn.functional as F
import torch

class SingleBranch(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.head_pool = opt.head_pool
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        global_feature = features[:, 0]
        local_feature = features[:, 1:]
        if self.head_pool == "global":
            feature = global_feature
        elif self.head_pool == "avg":
            local_feature = local_feature.transpose(1, 2)
            feature = torch.mean(local_feature, 2).squeeze()
        elif self.head_pool == "max":
            local_feature = local_feature.transpose(1, 2)
            feature = torch.max(local_feature, 2)[0].squeeze()
        elif self.head_pool == "avg+max":
            local_feature = local_feature.transpose(1, 2)
            avg_feature = torch.mean(local_feature, 2).squeeze()
            max_feature = torch.max(local_feature, 2)[0].squeeze()
            feature = avg_feature+max_feature
        else:
            raise TypeError("head_pool 不在支持的列表中！！！")

        cls, feature = self.classifier(feature)
        return [cls, feature]


class SingleBranchCNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        global_feature = self.pool(features).reshape(features.shape[0], -1)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]


class SingleBranchSwin(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)

    def forward(self, features):
        global_feature = self.pool(features.transpose(
            2, 1)).reshape(features.shape[0], -1)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]
