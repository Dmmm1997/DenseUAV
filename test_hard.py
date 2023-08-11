# -*- coding: utf-8 -*-

from __future__ import print_function, division
from datasets.queryDataset import Dataset_query, Query_transforms, Dataset_gallery, test_collate_fn

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

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument(
    '--test_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/test', type=str, help='./test_data')
parser.add_argument('--name', default='resnet',
                    type=str, help='save model path')
parser.add_argument('--checkpoint', default='net_119.pth',
                    type=str, help='save model path')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=224, type=int, help='height')
parser.add_argument('--w', default=224, type=int, help='width')
parser.add_argument('--ms', default='1', type=str,
                    help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
parser.add_argument('--mode', default='hard', type=str,
                    help='1:drone->satellite   2:satellite->drone')
parser.add_argument('--num_worker', default=8, type=int,
                    help='1:drone->satellite   2:satellite->drone')

parser.add_argument('--split_feature', default=1, type=int, help='')

opt = parser.parse_args()
print(opt.name)
###load config###
# load the training config
config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
for cfg, value in config.items():
    if cfg not in opt:
        setattr(opt, cfg, value)

str_ids = opt.gpu_ids.split(',')
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True


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

image_datasets_query = Dataset_query(os.path.join(data_dir, "query_drone"), data_query_transforms)

# image_datasets_gallery = Dataset_gallery(os.path.join(opt.test_dir, "total_info_ms_i10m.txt"), data_transforms)

image_datasets_gallery = Dataset_gallery(os.path.join(opt.test_dir, "total_info_ss_i10m.txt"), data_transforms)


dataloaders_query = torch.utils.data.DataLoader(image_datasets_query, batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_worker, collate_fn=test_collate_fn)

split_nums = len(image_datasets_gallery)//opt.split_feature

list_split = [split_nums]*opt.split_feature

list_split[-1] = len(image_datasets_gallery)-(opt.split_feature-1)*split_nums

gallery_datasets_list = torch.utils.data.random_split(image_datasets_gallery, list_split)

dataloaders_gallery = {ind: torch.utils.data.DataLoader(gallery_datasets_list[ind], batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_worker, collate_fn=test_collate_fn) for ind in range(opt.split_feature)}

use_gpu = torch.cuda.is_available()

def extract_feature(model, dataloaders, view_index=1):
    features = torch.FloatTensor()
    infos_list = np.zeros((0,2),dtype=np.float32)
    path_list = []
    for data in tqdm(dataloaders):
        img, infos, path = data
        path_list.extend(path)
        # infos_list.extend(infos)
        infos_list = np.concatenate((infos_list,infos),0)
        # if opt.LPN:
        #     # ff = torch.FloatTensor(n,2048,6).zero_().cuda()
        #     ff = torch.FloatTensor(n,512,opt.block).zero_().cuda()
        # else:
        #     ff = torch.FloatTensor(n, 2048).zero_().cuda()

        input_img = Variable(img.cuda())
        if view_index == 1:
            outputs, _ = model(input_img, None)
        elif view_index == 3:
            _, outputs = model(None, input_img)
        outputs = outputs[1]
        ff = outputs
        # # norm feature
        if len(ff.shape) == 3:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * \
                np.sqrt(opt.block)
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.data.cpu()), 0)
    return features,infos_list,path_list


model = load_network(opt)
print("这是%s的结果" % opt.checkpoint)
# model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

if __name__ == "__main__":
    with torch.no_grad():
        query_feature, query_infos, query_path = extract_feature(
            model, dataloaders_query, 1)
        gallery_features = torch.FloatTensor()
        gallery_infos = np.zeros((0,2),dtype=np.float32)
        gallery_paths = []
        for i in range(opt.split_feature):
            gallery_feature,gallery_info,gallery_path = extract_feature(
                model, dataloaders_gallery[i], 3)
            gallery_infos = np.concatenate((gallery_infos,gallery_info),0)
            gallery_features = torch.cat((gallery_features,gallery_feature),0)
            gallery_paths.extend(gallery_path)


    # For street-view image, we use the avg feature as the final feature.

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    with open('inference_time.txt', 'w') as F:
        F.write('Test complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    # Save to Matlab for check
    
    result = {'gallery_f': gallery_features.numpy(),  'gallery_infos': gallery_infos.astype(np.float32), 'query_path':query_path,
                'query_f': query_feature.numpy(),  'query_infos': query_infos.astype(np.float32), 'gallery_path':gallery_paths}
    scipy.io.savemat('pytorch_result_{}_ss.mat'.format(opt.mode), result)

    # print(opt.name)
    # result = 'result.txt'
    # os.system('python evaluate_gpu.py | tee -a %s'%result)




