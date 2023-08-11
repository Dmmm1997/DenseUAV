'''
在基于初始位置在邻近区域内搜索
'''
import scipy.io
import argparse
import os
import cv2
import numpy as np
import glob
from tool.get_property import find_GPS_image
from PIL import Image
from torchvision import transforms
import torch
from torch.autograd import Variable
from datasets.queryDataset import CenterCrop
import yaml
import torch.backends.cudnn as cudnn
from tool.utils import load_network

# Jiliang TN30.32576454772815 TE120.35248328688216 BN30.318951172904185 BE120.3631189084238
# Jingmao TN30.325763471673625 TE120.37341802729506 BN30.320529681696023 BE120.38174250997761
# Xianke TN30.326352560545967 TE120.3665695697911 BN30.317768074272823 BE120.37753817499194
# 30.323690034336742, 120.36964312478378 30.32072410729269, 120.37478722220634 小现科
University = "计量"
def getopts():
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--video', default="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/实际测试图像({})/test02".format(University),
                        type=str,
                        help='test_image_index')
    parser.add_argument('--satelliteMat', default="features{}.mat".format(University), type=str, help='./test_data')
    parser.add_argument('--Map', default="../../maps/{}.tif".format(University), type=str, help='./test_data')
    parser.add_argument('--neiborDistance', default=2e-3, type=float, help='./test_data')
    parser.add_argument('--saveName', default="neibor{}.tif".format(University), type=str, help='./test_data')
    parser.add_argument('--galleryPath', default="/media/dmmm/CE31-3598/DataSets/DenseCV_Data/satelliteHub({})".format(University),
                        type=str, help='./test_data')
    opts = parser.parse_args()
    return opts


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

data_transforms = transforms.Compose([
    CenterCrop(),
    transforms.Resize((256, 256), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def sort_(list):
    list_ = [int(i.split(".JPG")[-2].split("DJI_")[-1]) for i in list]
    arg = np.argsort(list_)
    newlist = np.array(list)[arg]
    return newlist


def getPosInfo(imgPath):
    GPS_info = find_GPS_image(imgPath)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    E = y[3]
    N = y[1]
    return [N, E]

class GalleryProcess:
    def __init__(self, opts):
        # 初始gps信息 (N,E)
        # self.init_Gps = opts.initGps
        self.galleryPath = opts.galleryPath
        # 获取读取gallery的前向传播后的特征 以及标签
        galleryFL = scipy.io.loadmat(opts.satelliteMat)
        self.galleryFeature, self.galleryLabel = galleryFL["features"], galleryFL["labels"]
        # 获取gallery卫星图对应的位置信息
        galleryPosInfoDict = self.generateDictOfGalleryPosInfo()
        self.galleryPosInfoDict = galleryPosInfoDict
        self.galleryPosKey = list(galleryPosInfoDict.keys())
        self.galleryPosValue = list(galleryPosInfoDict.values())
        self.neiborDistance = opts.neiborDistance


    # 获取邻近域内的图像
    def getNeiborFeature(self):
        D = np.array(self.galleryPosValue)-self.init_Gps
        # distance = np.matmul(D,D.T)
        # mask = np.eye(distance.shape[0],dtype=np.bool8)
        # distance = np.sqrt(distance[mask]).reshape(-1)
        distance = np.sqrt(np.power(D, 2).sum(1)).reshape(-1)
        neiborIndex = np.array(self.galleryPosKey)[distance<self.neiborDistance]
        return self.cutFeatureByNeibor(neiborIndex)

    def cutFeatureByNeibor(self,neiborIndex):
        neiborIndex_ = np.array(neiborIndex,dtype=np.int32)
        galleryFeature = self.galleryFeature[neiborIndex_]
        galleryLabel = self.galleryLabel[neiborIndex_]
        return galleryFeature,galleryLabel


    #######################################################################
    # sort the images and return topK index
    def getBestImage(self, qf, gf, gl):
        query = qf.view(-1, 1)
        query = query.cpu().numpy()
        # print(query.shape)
        score = np.matmul(gf, query)
        score = score.squeeze()
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

    def generateDictOfGalleryPosInfo(self):
        satellite_configDict = {}
        with open(os.path.join(self.galleryPath, "PosInfo.txt"), "r") as F:
            context = F.readlines()
            for line in context:
                splitLineList = line.split(" ")
                satellite_configDict[splitLineList[0]] = [float(splitLineList[1].split("N")[-1]),
                                                          float(splitLineList[2].split("E")[-1])]
        return satellite_configDict

    def updataGPS(self,gps):
        self.init_Gps = gps



#####################################################################
def extract_feature(img, model,opts):
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

def loadNetwork(opts):
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
    return model

def main():
    opts = getopts()

    AllImage = cv2.imread(opts.Map)

    gallery = GalleryProcess(opts)

    h, w, c = AllImage.shape

    retrivalRadio = int(opts.neiborDistance/(endE - startE) * w)

    model = loadNetwork(opts)

    # 判断是视频还是文件夹
    if os.path.isdir(opts.video):
        droneImages = glob.glob(os.path.join(opts.video,"*.JPG"))
        query_paths = sort_(droneImages)
        initGPS = getPosInfo(query_paths[0])
        gallery.updataGPS(initGPS)
        for index,imgPath in enumerate(query_paths):
            # 获取真实位置信息
            queryPosInfo = getPosInfo(imgPath)
            # 读取图片
            img = Image.open(imgPath)
            input = data_transforms(img)
            input = torch.unsqueeze(input, 0)
            # 提取query的特征
            with torch.no_grad():
                feature = extract_feature(input, model,opts)
            # 在邻近卫星图中检索
            neiborFeature,neiborLabel = gallery.getNeiborFeature()
            bestIndex = gallery.getBestImage(feature, neiborFeature, neiborLabel)
            # 获取卫星图对应的位置信息
            bestMatchedPosInfo = gallery.galleryPosInfoDict[bestIndex]
            gallery_N = float(bestMatchedPosInfo[0])
            gallery_E = float(bestMatchedPosInfo[1])
            # 绘制
            ## query
            X_Q = int((queryPosInfo[1] - startE) / (endE - startE) * w)
            Y_Q = int((queryPosInfo[0] - startN) / (endN - startN) * h)
            X_G = int((gallery_E - startE) / (endE - startE) * w)
            Y_G = int((gallery_N - startN) / (endN - startN) * h)

            cv2.circle(AllImage,(X_G,Y_G),retrivalRadio,color=(0,255,0),thickness=3)
            if index >= 10:
                cv2.circle(AllImage, (X_Q, Y_Q), 50, color=(255, 0, 0), thickness=6)
                cv2.putText(AllImage, str(index), (X_Q - 40, Y_Q + 25), cv2.FONT_HERSHEY_COMPLEX, 2.2, color=(255, 0, 0),
                            thickness=3)
                cv2.circle(AllImage, (X_G, Y_G), 50, color=(0, 0, 255), thickness=6)
                cv2.putText(AllImage, str(index), (X_G - 40, Y_G + 25), cv2.FONT_HERSHEY_COMPLEX, 2.2, color=(0, 0, 255),
                            thickness=3)
            else:
                cv2.circle(AllImage, (X_Q, Y_Q), 50, color=(255, 0, 0), thickness=6)
                cv2.putText(AllImage, str(index), (X_Q - 30, Y_Q + 40), cv2.FONT_HERSHEY_COMPLEX, 3, color=(255, 0, 0),
                            thickness=3)
                cv2.circle(AllImage, (X_G, Y_G), 50, color=(0, 0, 255), thickness=6)
                cv2.putText(AllImage, str(index), (X_G - 30, Y_G + 30), cv2.FONT_HERSHEY_COMPLEX, 3, color=(0, 0, 255),
                            thickness=3)
            img = cv2.resize(AllImage,(1080,720))
            cv2.imshow("1",img)
            cv2.waitKey(0)

            gallery.updataGPS([gallery_N,gallery_E])

        cv2.imwrite(opts.saveName,AllImage)

    else:
        # 获取视频流
        capture = cv2.VideoCapture(opts.video)
        # 遍历视频流
        while True:
            ret,img = capture.read()
            if not ret:
                print("读取视频结束!")
                break





if __name__ == '__main__':
    main()