import cv2
from utils import get_fileNames
import os
from get_property import find_GPS_image
from tqdm import tqdm
import numpy as np
import functools
import sys
from multiprocessing import Pool


UAV_target_size = (1440,1080)
Satellite_target_size = (512,512)

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number

def compare_personal(x,y):
    x1 = int(os.path.split(x)[1].split(".JPG")[0])
    y1 = int(os.path.split(y)[1].split(".JPG")[0])
    return x1-y1

# from center crop image
def center_crop_and_resize(img,target_size=None):
    h,w,c = img.shape
    min_edge = min((h,w))
    if min_edge==h:
        edge_lenth = int((w-min_edge)/2)
        new_image = img[:,edge_lenth:w-edge_lenth,:]
    else:
        edge_lenth = int((h - min_edge) / 2)
        new_image = img[edge_lenth:h-edge_lenth, :, :]
    assert new_image.shape[0]==new_image.shape[1],"the shape is not correct"
    # LINEAR Interpolation
    if target_size:
        new_image = cv2.resize(new_image,target_size)

    return new_image

def resize(img,target_size=None):
    # LINEAR Interpolation
    return cv2.resize(img,target_size)


def getFileNameList(fullnamelist):
    list_return = []
    for i in fullnamelist:
        _,filename = os.path.split(i)
        list_return.append(filename)
    return list_return

def process(info):
    index, [drone_80,drone_90,drone_100] = info
    satellite_80 = drone_80.replace(".JPG","_satellite.tif")
    satellite_90 = drone_90.replace(".JPG","_satellite.tif")
    satellite_100 = drone_100.replace(".JPG","_satellite.tif")
    if not (os.path.exists(satellite_80) and os.path.exists(satellite_90) and os.path.exists(satellite_100)):
        print("没有对应的satellite图像存在，请查看{}".format(satellite_80+" "+satellite_90+" "+satellite_100))
        sys.exit(0)
    # ----new added----
    satellite_80_old = drone_80.replace(".JPG","_satellite_old.tif")
    satellite_90_old = drone_90.replace(".JPG","_satellite_old.tif")
    satellite_100_old = drone_100.replace(".JPG","_satellite_old.tif")
    if not (os.path.exists(satellite_80_old) and os.path.exists(satellite_90_old) and os.path.exists(satellite_100_old)):
        print("没有对应的satellite图像存在，请查看{}".format(satellite_80_old+" "+satellite_90_old+" "+satellite_100_old))
        sys.exit(0)

    name = sixNumber(index)
    droneCurDir = os.path.join(dronePath, name)
    SatelliteCurDir = os.path.join(satellitePath, name)
    os.makedirs(droneCurDir,exist_ok=True)
    os.makedirs(SatelliteCurDir,exist_ok=True)

    # # load drone and satellite image
    # drone80_img = cv2.imread(drone_80)
    # drone90_img = cv2.imread(drone_90)
    # drone100_img = cv2.imread(drone_100)
    # satellite80_img = cv2.imread(satellite_80)
    # satellite90_img = cv2.imread(satellite_90)
    # satellite100_img = cv2.imread(satellite_100)
    # # ---new added---
    # satellite80_img_old = cv2.imread(satellite_80_old)
    # satellite90_img_old = cv2.imread(satellite_90_old)
    # satellite100_img_old = cv2.imread(satellite_100_old)

    # # process image including crop and resize
    # processed_drone80_img = resize(drone80_img,target_size=UAV_target_size)
    # processed_drone90_img = resize(drone90_img,target_size=UAV_target_size)
    # processed_drone100_img = resize(drone100_img,target_size=UAV_target_size)
    # processed_satellite80_img = resize(satellite80_img,target_size=Satellite_target_size)
    # processed_satellite90_img = resize(satellite90_img,target_size=Satellite_target_size)
    # processed_satellite100_img = resize(satellite100_img,target_size=Satellite_target_size)
    # # ---new added---
    # processed_satellite80_img_old = resize(satellite80_img_old,target_size=Satellite_target_size)
    # processed_satellite90_img_old = resize(satellite90_img_old,target_size=Satellite_target_size)
    # processed_satellite100_img_old = resize(satellite100_img_old,target_size=Satellite_target_size)

    # cv2.imwrite(os.path.join(droneCurDir, "H80.JPG"), processed_drone80_img)
    # cv2.imwrite(os.path.join(droneCurDir, "H90.JPG"), processed_drone90_img)
    # cv2.imwrite(os.path.join(droneCurDir, "H100.JPG"), processed_drone100_img)
    satelliteImgPath80 = os.path.join(SatelliteCurDir,"H80.tif")
    satelliteImgPath90 = os.path.join(SatelliteCurDir,"H90.tif")
    satelliteImgPath100 = os.path.join(SatelliteCurDir,"H100.tif")
    # cv2.imwrite(satelliteImgPath80, processed_satellite80_img)
    # cv2.imwrite(satelliteImgPath90, processed_satellite90_img)
    # cv2.imwrite(satelliteImgPath100, processed_satellite100_img)
    # # # ---new added---
    # satelliteImgPath80_old = os.path.join(SatelliteCurDir,"H80_old.tif")
    # satelliteImgPath90_old = os.path.join(SatelliteCurDir,"H90_old.tif")
    # satelliteImgPath100_old = os.path.join(SatelliteCurDir,"H100_old.tif")
    # cv2.imwrite(satelliteImgPath80_old, processed_satellite80_img_old)
    # cv2.imwrite(satelliteImgPath90_old, processed_satellite90_img_old)
    # cv2.imwrite(satelliteImgPath100_old, processed_satellite100_img_old)

    # write the GPS information
    GPS_info = find_GPS_image(drone_80)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    information = "{} {}{} {}{} {}\n".format(satelliteImgPath80,y[2],y[3],y[0],y[1],height)
    # ---new added---
    # information_old = "{} {}{} {}{} {}\n".format(satelliteImgPath80_old,y[2],y[3],y[0],y[1],height)
    # return information,information_old
    return [information]

heightList = ["80","90","100"]
index = 2256 # 测试集需要根据训练集的总长设置index
dir_path = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/"
mode = "test"
oriPath = os.path.join(dir_path,"oridata", mode, "University_UAV_Images")
dirList = os.listdir(oriPath)
root_dir = os.path.join(dir_path, "data_2021")
mode_dir_path = os.path.join(root_dir, mode)
os.makedirs(mode_dir_path,exist_ok=True)

# GPS txt file
txt_path = os.path.join(root_dir, "Dense_GPS_{}.txt".format(mode))
file = open(txt_path, 'w')
dronePath = os.path.join(mode_dir_path, "drone")
os.makedirs(dronePath,exist_ok=True)
satellitePath = os.path.join(mode_dir_path, "satellite")
os.makedirs(satellitePath,exist_ok=True)

p = Pool(8)
for p_idx, place in enumerate(dirList):
    if not os.path.isdir(os.path.join(oriPath,place)):
        continue

    Drone_JPG_paths_80 = get_fileNames(os.path.join(oriPath, place, "80"),endwith=".JPG")
    Drone_JPG_paths_90 = get_fileNames(os.path.join(oriPath, place, "90"), endwith=".JPG")
    Drone_JPG_paths_100 = get_fileNames(os.path.join(oriPath, place, "100"), endwith=".JPG")
    Drone_JPG_paths_80 = sorted(Drone_JPG_paths_80,key=functools.cmp_to_key(compare_personal))
    Drone_JPG_paths_90 = sorted(Drone_JPG_paths_90,key=functools.cmp_to_key(compare_personal))
    Drone_JPG_paths_100 = sorted(Drone_JPG_paths_100,key=functools.cmp_to_key(compare_personal))
    f_80 = getFileNameList(Drone_JPG_paths_80)
    f_90 = getFileNameList(Drone_JPG_paths_90)
    f_100 = getFileNameList(Drone_JPG_paths_100)
    assert f_80==f_90 and f_90==f_100, "数据存在不对应请检查{}".format(place)
    print("Set index for the every iter")
    indexed_iters = []
    for ind, (drone_80,drone_90,drone_100) in tqdm(enumerate(zip(Drone_JPG_paths_80,Drone_JPG_paths_90,Drone_JPG_paths_100))):
        indexed_iters.append([index+ind, [drone_80,drone_90,drone_100]])
    index+=len(indexed_iters)
    for idx, res in enumerate(p.imap(process,indexed_iters)):
        if idx%50==0:
            print("-{}- {}/{} {}".format(p_idx, idx, len(indexed_iters), place))
        for info in res:
            file.write(info)

file.close()

p.close()