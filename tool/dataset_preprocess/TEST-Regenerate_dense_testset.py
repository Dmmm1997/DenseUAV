import numpy as np
from tqdm import tqdm
import os
import shutil
import glob
import cv2
from multiprocessing import Pool,Manager
import copy
from utils import Distance

# 对测试卫星图像进行密集切分，old版本,生成一个GPS_ALL.txt文件

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number


type = "new"

root_dir = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/oridata/test/{}_tif".format(type)

source_loc_info = os.path.join(root_dir, "PosInfo.txt")

output_dir = "/home/dmmm/Dataset/DenseUAV/data_2022/test/hard_gallery_satellite_ss_interval10m"
os.makedirs(output_dir,exist_ok=True)

with open(source_loc_info, "r") as F:
    context = F.readlines()

correspond_size = [640,768,896]
correspond_size = correspond_size[1:2]
gap=77
output_size = [256,256]
total_info = []



# p = Pool(10)

for line in context:
    info = line.strip().split(" ")
    university_name, TN, TE, BN, BE = info
    TE = eval(TE.split("TE")[-1])
    TN = eval(TN.split("TN")[-1])
    BE = eval(BE.split("BE")[-1])
    BN = eval(BN.split("BN")[-1])
    print("acutal_height:{}--width:{}".format(Distance(TN,TE,BN,TE),Distance(TN,TE,TN,BE)))

    source_map_tif_path = os.path.join(root_dir,"{}.tif".format(university_name))
    source_map_image = cv2.imread(source_map_tif_path)

    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['total_info'] = []

    h,w,c = source_map_image.shape
    print("image_height:{}--width:{}".format(h,w))
    x,y = np.meshgrid(list(range(0,w-896,gap)),list(range(0,h-896,gap)))
    x,y = x.reshape(-1),y.reshape(-1)
    inds = np.array(list(range(0,len(x),1)))

    def process(infos):
        ind = infos[0]
        position = infos[1:]
        
        info_list = []
        for size in correspond_size:
            East = TE + (position[0]+size/2)/w*(BE-TE)
            North = TN - (position[1]+size/2)/h*(TN-BN)
            image = source_map_image[position[1]:position[1]+size, position[0]:position[0]+size, :]
            image = cv2.resize(image, output_size)
            filepath = os.path.join(output_dir, "{}_{}_{}_{}.jpg".format(university_name,sixNumber(ind),size,type))
            cv2.imwrite(filepath, image)
            pos_info = [filepath, East, North, size]
            info_list.append(pos_info)

        return info_list

    p = Pool(20)
    for ind,res in enumerate(p.imap(process,zip(inds,x,y))):
        if ind % 100 == 0:
            print("{}/{} process the image!!".format(ind,len(inds)))
        total_info.extend(res)
    p.close()

info_path = "/home/dmmm/Dataset/DenseUAV/data_2022/test/{}_info.txt".format(type)

F = open(info_path, "w")
for info in total_info:
    F.write("{} {} {} {}\n".format(*info))

F.close()

# cat file1.txt file2.txt > merged.txt 合并文件