from __future__ import with_statement
import argparse
import scipy.io
import torch
import numpy as np
from datasets.queryDataset import CenterCrop
import glob
from torchvision import transforms
from PIL import Image
import yaml
from tool.utils import load_network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tool.get_property import find_GPS_image
import cv2

#######################################################################
# Evaluate
University="计量"
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--imageDir', default="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/实际测试图像({})/test02".format(University), type=str,
                    help='test_image_index')
parser.add_argument('--satelliteMat', default="features{}.mat".format(University), type=str, help='./test_data')
parser.add_argument('--MapDir', default="../../maps/{}.tif".format(University), type=str, help='./test_data')
parser.add_argument('--galleryPath', default="/media/dmmm/4T-3/DataSets/DenseCV_Data/inference_data/satelliteHub({})".format(University),
                    type=str, help='./test_data')
opts = parser.parse_args()
# TN30.325763471673625 TE120.37341802729506 BN30.320529681696023 BE120.38174250997761 jinrong

mapPosInfodir = "/home/dmmm/PycharmProjects/DenseCV/demo/maps/pos.txt"
with open(mapPosInfodir,"r") as F:
    listLine = F.readlines()
    for line in listLine:
        name,TN,TE,BN,BE = line.split(" ")
        if name==University:
            startE = eval(TE.split("TE")[-1])
            startN = eval(TN.split("TN")[-1])
            endE = eval(BE.split("BE")[-1])
            endN = eval(BN.split("BN")[-1])

AllImage = cv2.imread(opts.MapDir)
h, w, c = AllImage.shape


def generateDictOfGalleryPosInfo():
    satellite_configDict = {}
    with open(os.path.join(opts.galleryPath, "PosInfo.txt"), "r") as F:
        context = F.readlines()
        for line in context:
            splitLineList = line.split(" ")
            satellite_configDict[splitLineList[0]] = [float(splitLineList[1].split("N")[-1]),
                                                      float(splitLineList[2].split("E")[-1])]
    return satellite_configDict


def sort_(list):
    list_ = [int(i.split(".JPG")[-2].split("DJI_")[-1]) for i in list]
    arg = np.argsort(list_)
    newlist = np.array(list)[arg]
    return newlist


#######################################################################
# sort the images and return topK index
def getBestImage(qf, gf, gl):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze().cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    # query_index = np.argwhere(gl == ql)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    # junk_index = np.argwhere(gl == -1)

    # mask = np.in1d(index, junk_index, invert=True)
    # index = index[mask]
    return gl[index[0]]


data_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

query_paths = glob.glob(opts.imageDir + "/*.JPG")
# sorted(query_paths,key=lambda x : int(x.split(".JPG")[-2].split("DJI_")[-1]))
query_paths = sort_(query_paths)


#####################################################################
def extract_feature(img, model):
    count = 0
    n, c, h, w = img.size()
    count += n
    input_img = Variable(img.cuda())
    outputs, _ = model(input_img, None)
    ff = outputs
    # norm feature
    if len(ff.shape) == 3:
        # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
        # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(opts.block)
        ff = ff.div(fnorm.expand_as(ff))
        ff = ff.view(ff.size(0), -1)
    else:
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

    features = ff.data
    return features


def getPosInfo(imgPath):
    GPS_info = find_GPS_image(imgPath)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    E = y[3]
    N = y[1]
    return [N, E]


def imshowByIndex(index):
    galleryPath = os.path.join(opts.galleryPath, index + ".tif")
    image = cv2.imread(galleryPath)
    cv2.imshow("gallery", image)


######################################################################
# load network
config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opts.stride = config['stride']
opts.views = config['views']
opts.transformer = config['transformer']
opts.pool = config['pool']
opts.views = config['views']
opts.LPN = config['LPN']
opts.block = config['block']
opts.nclasses = config['nclasses']
opts.droprate = config['droprate']
opts.share = config['share']
opts.checkpoint = "net_119.pth"
torch.cuda.set_device("cuda:0")
cudnn.benchmark = True

model = load_network(opts)
model = model.eval()
model = model.cuda()

######################################################################
result = scipy.io.loadmat(opts.satelliteMat)
gallery_feature = torch.FloatTensor(result['features'])
gallery_label = result['labels']
gallery_feature = gallery_feature.cuda()

satellitePosInfoDict = generateDictOfGalleryPosInfo()  # 字典中的数组第一位为N 第二位为E
firstPos = getPosInfo(query_paths[0])
# gmap = gmplot.GoogleMapPlotter(firstPos[0], firstPos[1], 19)

queryPosDict = {"N": [], "E": []}
galleryPosDict = {"N": [], "E": []}
for query in query_paths:
    queryPosInfo = getPosInfo(query)
    queryPosDict["N"].append(float(queryPosInfo[0]))
    queryPosDict["E"].append(float(queryPosInfo[1]))
    img = Image.open(query)
    input = data_transforms(img)
    input = torch.unsqueeze(input, 0)
    with torch.no_grad():
        feature = extract_feature(input, model)
        bestIndex = getBestImage(feature, gallery_feature, gallery_label)
        imshowByIndex(bestIndex)
        bestMatchedPosInfo = satellitePosInfoDict[bestIndex]
        galleryPosDict["N"].append(float(bestMatchedPosInfo[0]))
        galleryPosDict["E"].append(float(bestMatchedPosInfo[1]))
        print("query--N:{} E:{} gallery--N:{} E:{}".format(queryPosInfo[0], queryPosInfo[1], bestMatchedPosInfo[0],
                                                           bestMatchedPosInfo[1]))
        # cv2.waitKey(0)

result = {"query": queryPosDict, "gallery": galleryPosDict}
scipy.io.savemat('global_{}.mat'.format(University), result)

index = 1
for N, E in zip(queryPosDict["N"], queryPosDict["E"]):
    X = int((E - startE) / (endE - startE) * w)
    Y = int((N - startN) / (endN - startN) * h)
    if index>=10:
        cv2.circle(AllImage, (X, Y), 50, color=(255, 0, 0), thickness=8)
        cv2.putText(AllImage, str(index), (X - 40, Y + 25), cv2.FONT_HERSHEY_COMPLEX, 2.2, color=(255, 0, 0),
                    thickness=3)
    else:
        cv2.circle(AllImage, (X, Y), 50, color=(255, 0, 0), thickness=8)
        cv2.putText(AllImage, str(index), (X - 30, Y + 30), cv2.FONT_HERSHEY_COMPLEX, 3, color=(255, 0, 0),
                    thickness=3)
    index += 1

index = 1
for N, E in zip(galleryPosDict["N"], galleryPosDict["E"]):
    X = int((E - startE) / (endE - startE) * w)
    Y = int((N - startN) / (endN - startN) * h)
    if index>=10:
        cv2.circle(AllImage, (X, Y), 50, color=(0, 0, 255), thickness=8)
        cv2.putText(AllImage, str(index), (X - 40, Y + 25), cv2.FONT_HERSHEY_COMPLEX, 2.2, color=(0, 0, 255),
                    thickness=3)
    else:
        cv2.circle(AllImage, (X, Y), 50, color=(0, 0, 255), thickness=8)
        cv2.putText(AllImage, str(index), (X - 30, Y + 30), cv2.FONT_HERSHEY_COMPLEX, 3, color=(0, 0, 255),
                    thickness=3)
    index += 1

# AllImage = cv2.resize(AllImage,(0,0),fx=0.25,fy=0.25)
cv2.imwrite("global_{}.tif".format(University), AllImage)
# gmap.plot(queryPosDict["N"], queryPosDict["E"], color="red")
# gmap.plot(galleryPosDict["N"], galleryPosDict["E"], color="blue")
#
# gmap.draw("user001_map.html")
