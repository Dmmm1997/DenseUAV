import torch
import torch.nn as nn
import timm
from .SingleBranch import SingleBranch, SingleBranchCNN, SingleBranchSwin
from .MSBA import MSBA
from .FSRA import FSRA
from .LPN import LPN


def make_head(opt):
    return Head(opt)


class Head(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.head = self.init_head(opt)
        self.opt = opt

    def init_head(self, opt):
        head = opt.head
        if head == "SingleBranch":
            head_model = SingleBranch(opt)
        elif head == "SingleBranchCNN":
            head_model = SingleBranchCNN(opt)
        elif head == "SingleBranchSwin":
            head_model = SingleBranchSwin(opt)
        elif head == "MSBA":
            head_model = MSBA(opt)
        elif head == "FSRA":
            head_model = FSRA(opt)
        elif head == "LPN":
            head_model = LPN(opt)
        else:
            raise NameError("{} not in the head list!!!".format(head))
        return head_model

    def forward(self, features):
        features = self.head(features)
        return features
