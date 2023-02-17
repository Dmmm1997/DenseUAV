import cv2
from utils import get_fileNames
import os
from get_property import find_GPS_image
from tqdm import tqdm
import numpy as np
import functools
import sys
from multiprocessing import Pool


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

    name = sixNumber(index)
    droneCurDir = os.path.join(dronePath, name)
    SatelliteCurDir = os.path.join(satellitePath, name)
    os.makedirs(droneCurDir,exist_ok=True)
    os.makedirs(SatelliteCurDir,exist_ok=True)

    # load drone and satellite image
    drone80_img = cv2.imread(drone_80)
    drone90_img = cv2.imread(drone_90)
    drone100_img = cv2.imread(drone_100)
    satellite80_img = cv2.imread(satellite_80)
    satellite90_img = cv2.imread(satellite_90)
    satellite100_img = cv2.imread(satellite_100)

    # process image including crop and resize
    processed_drone80_img = center_crop_and_resize(drone80_img,target_size=(768,768))
    processed_drone90_img = center_crop_and_resize(drone90_img,target_size=(768,768))
    processed_drone100_img = center_crop_and_resize(drone100_img,target_size=(768,768))
    processed_satellite80_img = center_crop_and_resize(satellite80_img,target_size=(512,512))
    processed_satellite90_img = center_crop_and_resize(satellite90_img,target_size=(512,512))
    processed_satellite100_img = center_crop_and_resize(satellite100_img,target_size=(512,512))

    cv2.imwrite(os.path.join(droneCurDir, "H80.JPG"), processed_drone80_img)
    cv2.imwrite(os.path.join(droneCurDir, "H90.JPG"), processed_drone90_img)
    cv2.imwrite(os.path.join(droneCurDir, "H100.JPG"), processed_drone100_img)
    satelliteImgPath80 = os.path.join(SatelliteCurDir,"H80.tif")
    satelliteImgPath90 = os.path.join(SatelliteCurDir,"H90.tif")
    satelliteImgPath100 = os.path.join(SatelliteCurDir,"H100.tif")
    cv2.imwrite(satelliteImgPath80, processed_satellite80_img)
    cv2.imwrite(satelliteImgPath90, processed_satellite90_img)
    cv2.imwrite(satelliteImgPath100, processed_satellite100_img)

    # write the GPS information
    GPS_info = find_GPS_image(drone_80)
    x = list(GPS_info.values())
    gps_dict_formate = x[0]
    y = list(gps_dict_formate.values())
    height = eval(y[5])
    information = "{} {}{} {}{} {}\n".format(satelliteImgPath80,y[2],y[3],y[0],y[1],height)
    # print(information)
    file.write(information)

heightList = ["80","90","100"]
dir_path = r"/media/dmmm/CE31-3598/DataSets/DenseCV_Data/高度数据集/"
oriPath = os.path.join(dir_path,"oridata","test")
dirList = os.listdir(oriPath)

new_dir_path = os.path.join(dir_path, "newdata","test")
if not os.path.exists(new_dir_path):
    os.makedirs(new_dir_path)
# GPS txt file
txt_path = os.path.join(dir_path, "Dense_GPS_test.txt")
file = open(txt_path, 'w')
dronePath = os.path.join(new_dir_path, "drone")
if not os.path.exists(dronePath):
    os.mkdir(dronePath)
satellitePath = os.path.join(new_dir_path, "satellite")
if not os.path.exists(satellitePath):
    os.mkdir(satellitePath)

p = Pool(10)
index = 0
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
        indexed_iters.append([ind, [drone_80,drone_90,drone_100]])

    
    for idx, res in enumerate(p.imap(process,indexed_iters)):
        if idx%10==0:
            print("-{}- {}/{} {}".format(p_idx,len(indexed_iters), place))
        

file.close()

p.close()