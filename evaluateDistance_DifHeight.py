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
parser.add_argument('--K', default=[1, 3, 5, 10], type=str, help='./test_data')
parser.add_argument('--M', default=5e3, type=str, help='./test_data')
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


def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def evaluateSingle(distance, K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*opts.M)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


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


def evaluate_SDM(indexOfTopK, queryIndex, K):
    query_path, _ = image_datasets[query_name].imgs[queryIndex]
    galleryTopKPath = [image_datasets[gallery_name].imgs[i][0]
                       for i in indexOfTopK[:K]]
    height = os.path.basename(query_path).split(".")[0]
    # get position information including latitude and longitude
    queryPosInfo = getLatitudeAndLongitude(query_path)
    galleryTopKPosInfo = getLatitudeAndLongitude(galleryTopKPath)
    # compute Euclidean distance of query and gallery
    distance = euclideanDistance(queryPosInfo, galleryTopKPosInfo)
    # compute single query evaluate result
    P = evaluateSingle(distance, K)
    return P, height


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

SDM_dict = {}
SDM_80m_dict = {}
SDM_90m_dict = {}
SDM_100m_dict = {}
for K in tqdm(range(1, 101, 1)):
    metric = 0
    metric_80m = []
    metric_90m = []
    metric_100m = []
    for i in range(len(query_label)):
        P_, height = evaluate_SDM(indexOfTopK_list[i], i, K)
        if "80" in height:
            metric_80m.append(P_)
        elif "90" in height:
            metric_90m.append(P_)
        elif "100" in height:
            metric_100m.append(P_)
        metric += P_
    metric = metric / len(query_label)
    SDM_80m_dict[K] = np.mean(metric_80m)
    SDM_90m_dict[K] = np.mean(metric_90m)
    SDM_100m_dict[K] = np.mean(metric_100m)
    if K in opts.K:
        print("metric{} = {:.2f}%".format(K, metric * 100))
    SDM_dict[K] = metric

MA_dict = {}
for meter in tqdm(range(1,101,1)):
    MA_K = 0
    for i in range(len(query_label)):
        MA_meter = evaluate_MA(indexOfTopK_list[i][0],i)
        if MA_meter<meter:
            MA_K+=1
    MA_K = MA_K/len(query_label)
    MA_dict[meter]=MA_K
        

with open("SDM@K(1,100).json", 'w') as F:
    json.dump(SDM_dict, F, indent=4)

with open("80m_SDM@K(1,100).json", 'w') as F:
    json.dump(SDM_80m_dict, F, indent=4)

with open("90m_SDM@K(1,100).json", 'w') as F:
    json.dump(SDM_90m_dict, F, indent=4)

with open("100m_SDM@K(1,100).json", 'w') as F:
    json.dump(SDM_100m_dict, F, indent=4)

with open("MA@K(1,100)", 'w') as F:
    json.dump(MA_dict, F, indent=4)
