import cv2
import os
import glob
from collections import defaultdict
from get_property import find_GPS_image
from utils import get_fileNames
import sys
from tqdm import tqdm
'''
通过Uav图像找出satellite图像，用于训练部分
'''
root_dir = "/media/dmmm/CE31-3598/DataSets/DenseCV_Data/高度数据集/oridata/test"
PlaceNameList = []

for root, PlaceName, files in os.walk(root_dir):
    PlaceNameList = PlaceName
    break

# PlaceNameList = os.listdir(root_dir)
# y = 10x-500
satellite_size = 768
correspond_size = {'80':640,'90':768,'100':896}

place_info_dict = defaultdict(list)
with open(os.path.join(root_dir,"PosInfo.txt"),"r") as F:
    context = F.readlines()
    for line in context:
        name = line.split(" ")[0]
        TN = float(line.split((" "))[1].split("TN")[-1])
        TE = float(line.split((" "))[2].split("TE")[-1])
        BN = float(line.split((" "))[3].split("BN")[-1])
        BE = float(line.split((" "))[4].split("BE")[-1])
        place_info_dict[name] = [TN,TE,BN,BE]


# 遍历每个地方的文件加，首先获取到大地图，然后进行切割
for place in PlaceNameList:
    place_root = os.path.join(root_dir,place)
    try:
        cur_TN,cur_TE,cur_BN,cur_BE = place_info_dict[place]
    except:
        print("{}没有在PosInfo.txt中定义！！".format(place))
        continue
    satellite_tif = os.path.join(root_dir,place + ".tif")

    BigSatellite = cv2.imread(satellite_tif)
    h,w,c = BigSatellite.shape
    JPG_List = get_fileNames(place_root,endwith=".JPG")
    for JPG in tqdm(JPG_List):
        satelliteTif = JPG.replace(".JPG","_satellite.tif")
        GPS_info = find_GPS_image(JPG)
        gps_dict_formate = list(GPS_info.values())[0]
        y = list(gps_dict_formate.values())
        E,N = y[3],y[1]
        satellite_size = correspond_size[JPG.split("/")[-2]]
        centerX = (E-cur_TE)/(cur_BE-cur_TE)*w # 计算当前无人机位置对应大图中的位置
        centerY = (N-cur_TN)/(cur_BN-cur_TN)*h

        if centerY<satellite_size/2 or centerX<satellite_size/2 or centerX>w-satellite_size/2 or centerY>h-satellite_size/2:
            print("切取区域超出图像范围")
            sys.exit(0)
        TLX = int(centerX-satellite_size/2)
        TLY = int(centerY-satellite_size/2)
        BRX = int(centerX+satellite_size/2)
        BRY = int(centerY+satellite_size/2)

        cropImage = BigSatellite[TLY:BRY,TLX:BRX,:] # 切出指定位置的内容

        cv2.imwrite(satelliteTif,cropImage)