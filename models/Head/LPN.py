import torch
import torch.nn as nn
import numpy as np
from .utils import ClassBlock
from torch.nn import functional as F
import math 


class LPN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.block = opt.block
        self.global_classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)
        for i in range(opt.block):
            name = 'classifier_lpn_' + str(i+1)
            setattr(self, name, ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck))
        self.opt = opt

    def forward(self, features):
        cls_token = features[:, 0]
        image_tokens = features[:, 1:]
        # 全局特征
        global_cls, global_feature = self.global_classifier(cls_token)
        # LPN特征
        image_tokens = image_tokens.reshape(image_tokens.size(0),int(np.sqrt(image_tokens.size(1))),int(np.sqrt(image_tokens.size(1))),image_tokens.size(2))
        image_tokens = image_tokens.permute(0,3,1,2)
        LPN_result = self.get_part_pool(image_tokens).squeeze(-1)
        LPN_cls_features = self.part_classifier(LPN_result)
        LPN_cls = []
        LPN_features = []
        for f in LPN_cls_features:
            LPN_cls.append(f[0])
            LPN_features.append(f[1])
        total_cls = [global_cls]+LPN_cls
        total_features = [global_feature]+LPN_features
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls,total_features]
    

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                xmin = c_h - i * per_h
                xmax = c_h + i * per_h
                ymin = c_w - i * per_w
                ymax = c_w + i * per_w
                x_curr = x[:, :, xmin:xmax, ymin:ymax]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, x, cls_name='classifier_lpn_'):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        return y
    

class LPN_CNN(nn.Module):
    def __init__(self, opt):
        super().__init__()
        
        self.block = opt.block
        # self.global_classifier = ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck)
        for i in range(opt.block):
            name = 'classifier_lpn_' + str(i+1)
            setattr(self, name, ClassBlock(opt.in_planes, opt.nclasses, opt.droprate, num_bottleneck= opt.num_bottleneck))
        self.opt = opt

    def forward(self, features):
        image_tokens = features
        # # 全局特征
        # global_cls, global_feature = self.global_classifier(cls_token)
        # LPN特征
        # image_tokens = image_tokens.reshape(image_tokens.size(0),int(np.sqrt(image_tokens.size(1))),int(np.sqrt(image_tokens.size(1))),image_tokens.size(2))
        # image_tokens = image_tokens.permute(0,3,1,2)
        LPN_result = self.get_part_pool(image_tokens).squeeze(-1)
        LPN_cls_features = self.part_classifier(LPN_result)
        LPN_cls = []
        LPN_features = []
        for f in LPN_cls_features:
            LPN_cls.append(f[0])
            LPN_features.append(f[1])
        total_cls = LPN_cls
        total_features = LPN_features
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls,total_features]
    

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                xmin = c_h - i * per_h
                xmax = c_h + i * per_h
                ymin = c_w - i * per_w
                ymax = c_w + i * per_w
                x_curr = x[:, :, xmin:xmax, ymin:ymax]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, x, cls_name='classifier_lpn_'):
        part = {}
        predict = {}
        for i in range(self.block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(self.block):
            y.append(predict[i])
        return y