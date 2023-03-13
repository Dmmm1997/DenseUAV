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
parser.add_argument('--mode', default="1", type=str,
                    help='1:drone->satellite 2:satellite->drone')
opts = parser.parse_args()

opts.config = os.path.join(opts.root_dir, "Dense_GPS_ALL.txt")
opts.test_dir = os.path.join(opts.root_dir, "test")
configDict = {}
with open(opts.config, "r") as F:
    context = F.readlines()
    for line in context:
        splitLineList = line.split(" ")
        configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
                                                       float(splitLineList[2].split("N")[-1])]

if opts.mode == "1":
    gallery_name = 'gallery_satellite'
    query_name = 'query_drone'
else:
    gallery_name = 'gallery_drone'
    query_name = 'query_satellite'

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in [
    gallery_name, query_name]}


#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated


######################################################################
if opts.mode == "1":
    result = scipy.io.loadmat('pytorch_result_1.mat')
else:
    result = scipy.io.loadmat('pytorch_result_2.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# sort the images and return topK index
def sort_img(qf, ql, gf, gl, K):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl == -1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index[:K]


def getLatitudeAndLongitude(imgPath):
    if isinstance(imgPath, list):
        posInfo = [configDict[p.split("/")[-2]] for p in imgPath]
    else:
        posInfo = configDict[imgPath.split("/")[-2]]
    return posInfo



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
    query_path, _ = image_datasets[query_name].imgs[queryIndex]
    galleryTopKPath = image_datasets[gallery_name].imgs[indexOfTop1][0]
    # get position information including latitude and longitude
    queryPosInfo = getLatitudeAndLongitude(query_path)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
    # get real distance
    distance_meter = latlog2meter(queryPosInfo[1],queryPosInfo[0],galleryTopKPosInfo[1],galleryTopKPosInfo[0])
    return distance_meter



indexOfTopK_list = []
for i in range(len(query_label)):
    indexOfTopK = sort_img(
        query_feature[i], query_label[i], gallery_feature, gallery_label, 100)
    indexOfTopK_list.append(indexOfTopK)

MA_dict = {}
for meter in tqdm(range(1,101,1)):
    MA_K = 0
    for i in range(len(query_label)):
        MA_meter = evaluate_MA(indexOfTopK_list[i][0],i)
        if MA_meter<meter:
            MA_K+=1
    MA_K = MA_K/len(query_label)
    MA_dict[meter]=MA_K
        
with open("MA@K(1,100)", 'w') as F:
    json.dump(MA_dict, F, indent=4)
