# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from tool.utils import load_network
from tqdm import tqdm
import warnings
from datasets.Dataloader_University import DataLoader_Inference
warnings.filterwarnings("ignore")
from datasets.queryDataset import Dataset_query,Query_transforms
# Options
# --------
University="计量"
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--root',default='/media/dmmm/4T-3/DataSets/DenseCV_Data/inference_data/satelliteHub({})'.format(University),type=str, help='./test_data')
parser.add_argument('--savename', default='features{}.mat'.format(University), type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth', type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--num_worker',default=4, type=int,help='1:drone->satellite   2:satellite->drone')

opt = parser.parse_args()
###load config###
# load the training config
config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
        config = yaml.load(stream)
for cfg,value in config.items():
    setattr(opt,cfg,value)

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

print('We use the scale: %s'%opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids)>0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image_datasets = DataLoader_Inference(root=opt.root,transforms=data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=opt.batchsize,
                                         shuffle=False, num_workers=opt.num_worker)
use_gpu = torch.cuda.is_available()



def extract_feature(model,dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in tqdm(dataloaders):
        img, label = data
        n, c, h, w = img.size()
        count += n
        # if opt.LPN:
        #     # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
        #     ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        # else:
        #     ff = torch.FloatTensor(n, 2048).zero_().cuda()
        input_img = Variable(img.cuda())
        outputs, _ = model(input_img, None)
        outputs = outputs[1]
        ff=outputs
        # norm feature
        if len(ff.shape)==3:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features



# Load Collected data Trained model
model = load_network(opt)
model = model.eval()
if use_gpu:
    model = model.cuda()
# Extract feature
since = time.time()


images = image_datasets.imgs

labels = image_datasets.labels

if __name__ == "__main__":
    with torch.no_grad():
        features = extract_feature(model,dataloaders)

    # Save to Matlab for check
    result = {'features':features.numpy(),'labels':labels}
    scipy.io.savemat(opt.savename,result)
