# -*- coding: utf-8 -*-
import sys
sys.path.append("../../")
import yaml
import argparse

from tool.utils import load_network, calc_flops_params

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='resnet',
                    type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth',
                    type=str, help='save model path')
parser.add_argument('--test_h', default=224, type=int, help='height')
parser.add_argument('--test_w', default=224, type=int, help='width')
opt = parser.parse_args()

config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
for cfg, value in config.items():
    setattr(opt, cfg, value)

model = load_network(opt).cuda()
model = model.eval()

# thop计算MACs
macs, params = calc_flops_params(
    model, (1, 3, opt.test_h, opt.test_w), (1, 3, opt.test_h, opt.test_w))
print("model MACs={}, Params={}".format(macs, params))
