import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import timm


def make_backbone(opt):
    backbone_model = Backbone(opt)
    return backbone_model


class Backbone(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.img_size = (opt.h,opt.w)
        self.backbone,self.output_channel = self.init_backbone(opt.backbone)
        

    def init_backbone(self, backbone):
        if backbone=="resnet50":
            backbone_model = timm.create_model('resnet50', pretrained=True)
            output_channel = 2048
        elif backbone=="ViTS-224":
            backbone_model = timm.create_model("vit_small_patch16_224", pretrained=True, img_size=self.img_size)
            output_channel = 384
        elif backbone=="ViTS-384":
            backbone_model = timm.create_model("vit_small_patch16_384", pretrained=True)
            output_channel = 384
        elif backbone=="DeitS-224":
            backbone_model = timm.create_model("deit_small_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="DeitB-224":
            backbone_model = timm.create_model("deit_base_distilled_patch16_224", pretrained=True)
            output_channel = 384
        elif backbone=="Pvtv2b2":
            backbone_model = timm.create_model("pvt_v2_b2", pretrained=True)
            output_channel = 512
        elif backbone=="ViTB-224":
            backbone_model = timm.create_model("vit_base_patch16_224", pretrained=True)
            output_channel = 768
        elif backbone=="SwinB-224":
            backbone_model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2S-256":
            backbone_model = timm.create_model("swinv2_small_window8_256", pretrained=True)
            output_channel = 768
        elif backbone=="Swinv2T-256":
            backbone_model = timm.create_model("swinv2_tiny_window16_256", pretrained=True)
            output_channel = 768
        elif backbone=="Convnext-T":
            backbone_model = timm.create_model("convnext_tiny", pretrained=True)
            output_channel = 768
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel

    def forward(self, image):
        features = self.backbone.forward_features(image)
        return features



    
