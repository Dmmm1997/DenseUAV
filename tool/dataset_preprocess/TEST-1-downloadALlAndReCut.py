import cv2
import numpy as np
from tqdm import tqdm
import os
import shutil

# TN30.325763481658377 TE120.3524959718277 BN30.318953289529816 BE120.36312306473656 jiliang
# TN30.32631193830858 TE120.36678791233928 BN30.317870065161575 BE120.37745052386553 xianke
# TN30.325763471673625 TE120.37341802729506 BN30.320441588110295 BE120.38193743012363
mapPosInfodir = "/home/dmmm/PycharmProjects/DenseCV/demo/maps/pos.txt"
# get_picture(startE, startN, endE, endN, 20, "./计量.tif", server="Google")

University = "Jiliang"

with open(mapPosInfodir,"r") as F:
    listLine = F.readlines()
    for line in listLine:
        name,TN,TE,BN,BE = line.split(" ")
        if name==University:
            startE = eval(TE.split("TE")[-1])
            startN = eval(TN.split("TN")[-1])
            endE = eval(BE.split("BE")[-1])
            endN = eval(BN.split("BN")[-1])

# 图像的根文件夹
dir_path = r"/media/dmmm/CE31-3598/DataSets/DenseCV_Data/satelliteHub({})".format(University)
if os.path.exists(dir_path):
    shutil.rmtree(dir_path)
os.mkdir(dir_path)
# 经纬度信息存放文件夹
infoFile = "/media/dmmm/4T-31/DataSets/DenseCV_Data/高度数据集/oridata/train/old_tif/PosInfo.txt".format(University)
file = open(infoFile, "w")

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number

oriImage = cv2.imread("/home/dmmm/PycharmProjects/DenseCV/demo/maps/{}.tif".format(University))
h,w,c = oriImage.shape

cropedSizeList = [640,768,896]
marginrate = 4
index = 0
for cropedSize in cropedSizeList:
    margin = cropedSize//marginrate #间距为图像尺寸的1/marginrate
    TopLeftEast = startE + (endE - startE)*cropedSize/2/w
    TopLeftNorth = startN + (endN - startN)*cropedSize/2/h
    BottomRightEast = endE - (endE - startE)*cropedSize/2/w
    BottomRightNorth = endN - (endN - startN)*cropedSize/2/h

    # YY = list(range(cropedSize//2,h-cropedSize//2+margin,margin))
    # XX = list(range(cropedSize//2,w-cropedSize//2+margin,margin))
    # Y_Size = len(YY)#y轴上总共记录图像数
    # X_Size = len(XX)#x轴上总共记录图像数
    X_Size = (w-cropedSize)//margin
    Y_Size = (h-cropedSize)//margin
    YY = np.linspace(cropedSize//2,h-cropedSize//2,Y_Size,dtype=np.int16)
    XX = np.linspace(cropedSize//2,w-cropedSize//2,X_Size,dtype=np.int16)

    pbar = tqdm(total=Y_Size*X_Size)

    PosInfoN = np.linspace(TopLeftNorth,BottomRightNorth,Y_Size)
    PosInfoE = np.linspace(TopLeftEast,BottomRightEast,X_Size)
    for n,y in zip(PosInfoN,YY):
        for e,x in zip(PosInfoE,XX):
            topLX = x - cropedSize//2
            topLY = y - cropedSize//2
            BottomRX = x + cropedSize//2
            BottomRY = y + cropedSize//2
            cropImage = oriImage[topLY:BottomRY,topLX:BottomRX,:]

            index_ = sixNumber(index)

            cv2.imwrite(os.path.join(dir_path,index_+".tif"),cropImage)

            file.write("{} N{:.10f} E{:.10f}\n".format(index_,n,e))

            index+=1
            pbar.update()