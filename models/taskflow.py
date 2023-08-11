import torch.nn as nn
from .Backbone.backbone import make_backbone
from .Head.head import make_head
import os
import torch


class Model(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = make_backbone(opt)
        opt.in_planes = self.backbone.output_channel
        self.head = make_head(opt)
        self.opt = opt

    def forward(self, drone_image, satellite_image):
        if drone_image is None:
            drone_res = None
        else:
            drone_features = self.backbone(drone_image)
            drone_res = self.head(drone_features)
        if satellite_image is None:
            satellite_res = None
        else:
            satellite_features = self.backbone(satellite_image)
            satellite_res = self.head(satellite_features)
        
        return drone_res,satellite_res
    
    def load_params(self, load_from):
        pretran_model = torch.load(load_from)
        model2_dict = self.state_dict()
        state_dict = {k: v for k, v in pretran_model.items() if k in model2_dict.keys() and v.size() == model2_dict[k].size()}
        model2_dict.update(state_dict)
        self.load_state_dict(model2_dict)


def make_model(opt):
    model = Model(opt)
    if os.path.exists(opt.load_from):
        model.load_params(opt.load_from)
    return model
