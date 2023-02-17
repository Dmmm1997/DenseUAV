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
warnings.filterwarnings("ignore")
from datasets.queryDataset import Dataset_query,Query_transforms

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--test_dir',default='/home/dmmm/University-Release/test',type=str, help='./test_data')
parser.add_argument('--name', default='from_transreid_256_4B_small_lr005_03_just_backbone_share', type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth', type=str, help='save model path')
parser.add_argument('--pool', default='max', type=str, help='avg|max')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--mode',default='2', type=int,help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--num_worker',default=0, type=int,help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--box_vis', default=False, type=int, help='')
parser.add_argument('--deit', default=True, type=int, help='')


opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./checkpoints',opt.name,'opts.yaml')
with open(config_path, 'r') as stream:
        config = yaml.load(stream)

opt.stride = config['stride']
opt.views = config['views']
opt.transformer = config['transformer']
opt.pool = config['pool']
opt.views = config['views']
opt.LPN = config['LPN']
opt.block = config['block']
opt.nclasses = config['nclasses']
opt.droprate = config['droprate']
opt.share = config['share']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else: 
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')
name = opt.name
test_dir = opt.test_dir

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
data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_query_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        # Query_transforms(pad=10,size=opt.w),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



data_dir = test_dir

if opt.multi:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query','multi-query']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=4) for x in ['gallery','query','multi-query']}
else:
    image_datasets_query = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_query_transforms) for x in ['query_satellite','query_drone']}

    image_datasets_gallery = {x: datasets.ImageFolder(os.path.join(data_dir,x) ,data_transforms) for x in ['gallery_satellite','gallery_drone']}

    image_datasets = {**image_datasets_query, **image_datasets_gallery}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=opt.num_worker) for x in ['gallery_satellite', 'gallery_drone','query_satellite','query_drone']}
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

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
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None)
                    elif view_index ==3:
                        _, outputs = model(None, input_img)
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                if opt.box_vis:
                    from utils import check_box
                    check_box(input_img,outputs)
                if i==0:
                    ff = outputs
                else:
                    ff += outputs
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

def get_id(img_path):
    camera_id = []
    labels = []
    paths = []
    for path, v in img_path:
        folder_name = os.path.basename(os.path.dirname(path))
        labels.append(int(folder_name))
        paths.append(path)
    return labels, paths

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
print("这是%s的结果"%opt.checkpoint)
# model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

if opt.mode==1:
    query_name = 'query_satellite'
    gallery_name = 'gallery_drone'
elif opt.mode==2:
    query_name = 'query_drone'
    gallery_name = 'gallery_satellite'
else:
    raise Exception("opt.mode is not required")

#gallery_name = 'gallery_street'
#query_name = 'query_street'

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))
print(query_name.split("_")[-1],"->",gallery_name.split("_")[-1])

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p[0]+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p[0]+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    with torch.no_grad():
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)

    # For street-view image, we use the avg feature as the final feature.
    '''
    if which_query == 2:
        new_query_label = np.unique(query_label)
        new_query_feature = torch.FloatTensor(len(new_query_label) ,512).zero_()
        for i, query_index in enumerate(new_query_label):
            new_query_feature[i,:] = torch.sum(query_feature[query_label == query_index, :], dim=0)
        query_feature = new_query_feature
        fnorm = torch.norm(query_feature, p=2, dim=1, keepdim=True)
        query_feature = query_feature.div(fnorm.expand_as(query_feature))
        query_label   = new_query_label
    elif which_gallery == 2:
        new_gallery_label = np.unique(gallery_label)
        new_gallery_feature = torch.FloatTensor(len(new_gallery_label), 512).zero_()
        for i, gallery_index in enumerate(new_gallery_label):
            new_gallery_feature[i,:] = torch.sum(gallery_feature[gallery_label == gallery_index, :], dim=0)
        gallery_feature = new_gallery_feature
        fnorm = torch.norm(gallery_feature, p=2, dim=1, keepdim=True)
        gallery_feature = gallery_feature.div(fnorm.expand_as(gallery_feature))
        gallery_label   = new_gallery_label
    '''
    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_path':gallery_path,'query_f':query_feature.numpy(),'query_label':query_label, 'query_path':query_path}
    scipy.io.savemat('pytorch_result.mat',result)

    print(opt.name)
    result = './checkpoints/%s/result.txt'%opt.name
    os.system('python evaluate_gpu.py | tee -a %s'%result)

