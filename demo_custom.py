import scipy.io
import argparse
import torch
import numpy as np
from datasets.queryDataset import CenterCrop
from torchvision import transforms
from PIL import Image
import yaml
from tool.utils_server import load_network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tool.get_property import find_GPS_image
# import matplotlib.pyplot as plt
import cv2
import json

University="计量"
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--img', default="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/实际测试图像({})/test02/DJI_0297.JPG".format(University), type=str, help='image path for visualization')
parser.add_argument("--galleryFeature",default="features{}.mat".format(University), type=str, help='galleryFeature')
parser.add_argument('--galleryPath', default="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/satelliteHub({})".format(University),
                    type=str, help='./test_data')
parser.add_argument('--MapDir', default="../../maps/{}.tif".format(University), type=str, help='./test_data')
parser.add_argument('--K', default=10, type=int, help='./test_data')
# parser.add_argument("--mode",default="1", type=int, help='1:drone->satellite 2:satellite->drone')
opts = parser.parse_args()

# 30.3257654243082, 120.37341533989152 30.320441588110295, 120.38193743012363
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


#####################################################################
# #Show result
# def imshow(path, title=None):
#     """Imshow for Tensor."""
#     im = plt.imread(path)
#     plt.imshow(im)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.1)  # pause a bit so that plots are updated


def getPosInfo(imgPath):
    GPS_info = find_GPS_image(imgPath)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    E = y[3]
    N = y[1]
    return [N, E]

#######################################################################
# sort the images and return topK index
def getTopKImage(qf, gf, gl,K):
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
    return gl[index[:K]]


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

def generateDictOfGalleryPosInfo():
    satellite_configDict = {}
    with open(os.path.join(opts.galleryPath, "PosInfo.txt"), "r") as F:
        context = F.readlines()
        for line in context:
            splitLineList = line.split(" ")
            satellite_configDict[splitLineList[0]] = [float(splitLineList[1].split("N")[-1]),
                                                      float(splitLineList[2].split("E")[-1])]
    return satellite_configDict


def imshowByIndex(index):
    galleryPath = os.path.join(opts.galleryPath, index + ".tif")
    image = cv2.imread(galleryPath)
    cv2.imshow("gallery", image)

data_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
result = scipy.io.loadmat(opts.galleryFeature)
gallery_feature = torch.FloatTensor(result['features'])
gallery_label = result['labels']
gallery_feature = gallery_feature.cuda()

satellitePosInfoDict = generateDictOfGalleryPosInfo()

img = Image.open(opts.img)
input = data_transforms(img)
input = torch.unsqueeze(input, 0)
# query Pos info
queryPosInfo = getPosInfo(opts.img)
Q_N = float(queryPosInfo[0])
Q_E = float(queryPosInfo[1])

with torch.no_grad():
    feature = extract_feature(input, model)
    indexSorted = getTopKImage(feature, gallery_feature, gallery_label,opts.K)


selectedGalleryPath = [os.path.join(opts.galleryPath,"{}.tif".format(index)) for index in indexSorted]
dict_paths = {"K":opts.K,"query":opts.img,"gallery":selectedGalleryPath}
with open("test.json", "w", encoding='utf-8') as f:
    # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
    # f.write(json.dumps(dict_paths, indent=4))
    json.dump(dict_paths, f, indent=4)  # 传入文件描述符，和dumps一样的结果


galleryPosDict = {"N": [], "E": []}
for index in indexSorted:
    bestMatchedPosInfo = satellitePosInfoDict[index]
    galleryPosDict["N"].append(float(bestMatchedPosInfo[0]))
    galleryPosDict["E"].append(float(bestMatchedPosInfo[1]))

## query visualization
X = int((Q_E - startE) / (endE - startE) * w)
Y = int((Q_N - startN) / (endN - startN) * h)
cv2.circle(AllImage, (X,Y), 30, color=(0, 0, 255), thickness=10)

## gallery visualization
index = 1
for N, E in zip(galleryPosDict["N"], galleryPosDict["E"]):
    X = int((E - startE) / (endE - startE) * w)
    Y = int((N - startN) / (endN - startN) * h)
    if index>=10:
        cv2.circle(AllImage, (X, Y), 30, color=(255, 0, 0), thickness=6)
        cv2.putText(AllImage, str(index), (X - 20, Y + 15), cv2.FONT_HERSHEY_COMPLEX, 1.2, color=(255, 0, 0),
                    thickness=2)
    else:
        cv2.circle(AllImage, (X, Y), 30, color=(255, 0, 0), thickness=6)
        cv2.putText(AllImage, str(index), (X - 20, Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2, color=(255, 0, 0),
                    thickness=2)
    index += 1

AllImage = cv2.resize(AllImage,(0,0),fx=0.25,fy=0.25)
cv2.imwrite("topKLocationBySingleImage.tif",AllImage)

# os.system("python visualization.py")

###报qt的错误
# try:
#     fig = plt.figure(figsize=(12, 4))
#     ax = plt.subplot(1, opts.K+1, 1)
#     ax.axis('off')
#     imshow(opts.img, 'query')
#     for i,path in enumerate(selectedGalleryPath):
#         ax = plt.subplot(1, 11, i + 2)
#         ax.axis('off')
#         imshow(path)
# except:
#     print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
#
# fig.savefig("show.png",dpi=600)

