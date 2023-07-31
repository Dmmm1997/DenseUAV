from torch import nn
import torch
from torch.nn import functional as F
import numpy as np


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    # classname = m.__class__.__name__
    # if classname.find('Linear') != -1:
    #     nn.init.normal_(m.weight, std=0.001)
    #     if m.bias:
    #         nn.init.constant_(m.bias, 0.0)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f=False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        feature_ = self.add_block(x)
        cls_ = self.classifier(feature_)
        return cls_, feature_


class Gem_heat(nn.Module):
    def __init__(self, dim=768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x = torch.transpose(x, 1, -1)
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x, p)
        # x = torch.transpose(x, 1, -1)
        # x = F.avg_pool1d(x, x.size(-1))
        x = x.view(x.size(0), x.size(1))
        # x = x.pow(1. / p)
        return x


class GeM(nn.Module):
    # channel-wise GeM zhedong zheng
    def __init__(self, dim=2048, p=1, eps=1e-6):
        super(GeM,  self).__init__()
        self.p = nn.Parameter(torch.ones(dim)*p)  # initial p
        self.eps = eps

    def forward(self, x):
        x = torch.transpose(x, 1, -1)
        x = (x+self.eps).pow(self.p)
        x = torch.transpose(x, 1, -1)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.view(x.size(0), x.size(1)).contiguous()
        x = x.pow(1./self.p)
        return x



class Pooling(nn.Module):
    def __init__(self, dim, pool="avg"):
        super(Pooling, self).__init__()
        self.pool = pool
        if pool == 'avg+max':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'avg':
            self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            self.maxpool2 = nn.AdaptiveMaxPool2d((1, 1))
        elif pool == 'gem':
            self.gem2 = Gem_heat(dim=dim)

    def forward(self, x):
        if self.pool == 'avg+max':
            x1 = self.avgpool2(x)
            x2 = self.maxpool2(x)
            x = torch.cat((x1, x2), dim=1)
        elif self.pool == 'avg':
            x = self.avgpool2(x)
        elif self.pool == 'max':
            x = self.maxpool2(x)
        elif self.pool == 'gem':
            x = self.gem2(x)
        return x


def vector2image(x, dim=1):  # (B,N,C)
    B, N, C = x.shape
    if dim == 1:
        return x.reshape(B, int(np.sqrt(N)), int(np.sqrt(N)), C)
    if dim == 2:
        return x.reshape(B, N, int(np.sqrt(C)), int(np.sqrt(C)))
    else:
        raise TypeError("dim is not correct!!")
