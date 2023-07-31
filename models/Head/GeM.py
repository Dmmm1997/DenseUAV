import torch.nn as nn
from .utils import ClassBlock, Pooling, vector2image



class GeM(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)
        self.pool = Pooling(opt.h//16*opt.w//16, "gem")

    def forward(self, features):# (N,(H*W+1),C)
        local_feature = features[:, 1:]
        local_feature = local_feature.transpose(1,2).contiguous()
        # local_feature = vector2image(local_feature,dim = 2)
        global_feature = self.pool(local_feature)
        cls, feature = self.classifier(global_feature)
        return [cls, feature]