import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import math

#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
# parser.add_argument('--query_index', default=10, type=int, help='test_image_index')
parser.add_argument(
    '--root_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/', type=str, help='./test_data')
opts = parser.parse_args()



######################################################################

result = scipy.io.loadmat('pytorch_result_hard_ss.mat')

query_feature = torch.FloatTensor(result['query_f'])
query_info = result['query_infos']
query_path = result['query_path']
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_info = result['gallery_infos']
gallery_path = result['gallery_path']

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# sort the images and return topK index
def sort_img(qf, gf):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    return index[:10]




def latlog2meter(lata, loga, latb, logb):
    # log 纬度 lat 经度 
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS =6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance



def evaluate_MA(indexOfTop1, queryIndex):
    # get position information including latitude and longitude
    queryPosInfo = query_info[queryIndex]
    galleryTopKPosInfo = gallery_info[indexOfTop1]
    # get real distance
    distance_meter = latlog2meter(queryPosInfo[1],queryPosInfo[0],galleryTopKPosInfo[1],galleryTopKPosInfo[0])
    return distance_meter



indexOfTopK_list = []
for i in range(len(query_info)):
    indexOfTopK = sort_img(
        query_feature[i],gallery_feature)
    indexOfTopK_list.append(indexOfTopK)

MA_dict = {}
for meter in tqdm(range(1,101,1)):
    MA_K = 0
    for i in range(len(query_info)):
        MA_meter = evaluate_MA(indexOfTopK_list[i][0],i)
        if MA_meter<meter:
            MA_K+=1
    MA_K = MA_K/len(query_info)
    MA_dict[meter]=MA_K
        
with open("MA@K(1,100)_ss.json", 'w') as F:
    json.dump(MA_dict, F, indent=4)
