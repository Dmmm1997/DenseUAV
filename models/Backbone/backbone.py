import torch.nn as nn
import timm
from .RKNet import RKNet
from .cvt import get_cvt_models
import torch

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
        elif backbone=="RKNet":
            backbone_model = RKNet()
            output_channel = 2048
        elif backbone=="senet":
            backbone_model = timm.create_model('legacy_seresnet50', pretrained=True)
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
        elif backbone=="EfficientNet-B2":
            backbone_model = timm.create_model("efficientnet_b2", pretrained=True)
            output_channel = 1408
        elif backbone=="EfficientNet-B3":
            backbone_model = timm.create_model("efficientnet_b3", pretrained=True)
            output_channel = 1536
        elif backbone=="EfficientNet-B5":
            backbone_model = timm.create_model("tf_efficientnet_b5", pretrained=True)
            output_channel = 2048
        elif backbone=="EfficientNet-B6":
            backbone_model = timm.create_model("tf_efficientnet_b6", pretrained=True)
            output_channel = 2304
        elif backbone=="vgg16":
            backbone_model = timm.create_model("vgg16", pretrained=True)
            output_channel = 512
        elif backbone=="cvt13":
            backbone_model, channels = get_cvt_models(model_size="cvt13")
            output_channel = channels[-1]
            checkpoint_weight = "/home/dmmm/VscodeProject/FPI/pretrain_model/CvT-13-384x384-IN-22k.pth"
            backbone_model = self.load_checkpoints(checkpoint_weight, backbone_model)
        else:
            raise NameError("{} not in the backbone list!!!".format(backbone))
        return backbone_model,output_channel
    
    def load_checkpoints(self, checkpoint_path, model):
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        filter_ckpt = {k: v for k, v in ckpt.items() if "pos_embed" not in k}
        missing_keys, unexpected_keys = model.load_state_dict(filter_ckpt, strict=False)
        print("Load pretrained backbone checkpoint from:", checkpoint_path)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        return model

    def forward(self, image):
        features = self.backbone.forward_features(image)
        return features