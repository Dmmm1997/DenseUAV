import numpy as np
from tqdm import tqdm
import os
import shutil
import glob
import cv2
from multiprocessing import Pool
import copy

type = "new"

info_file = "/home/dmmm/Dataset/DenseUAV/data_2022/test/{}_info.txt".format(type)

root_dir = "/home/dmmm/Dataset/DenseUAV/data_2022/test/{}_tif".format(type)

source_loc_info = os.path.join(root_dir, "PosInfo.txt")

with open(source_loc_info, "r") as F:
    context_map = F.readlines()

with open(info_file,"r") as F:
    context = F.readlines()

line = context[100]
infos = line.strip().split(" ")
filename = infos[0]
query_image = cv2.imread(filename)

E = float(infos[1])
N = float(infos[2])

for line in context_map:
    info = line.strip().split(" ")
    university_name, TN, TE, BN, BE = info
    TE = eval(TE.split("TE")[-1])
    TN = eval(TN.split("TN")[-1])
    BE = eval(BE.split("BE")[-1])
    BN = eval(BN.split("BN")[-1])

    source_map_tif_path = os.path.join(root_dir,"{}.tif".format(university_name))
    source_map_image = cv2.imread(source_map_tif_path)
    h,w = source_map_image.shape[:2]

    if TE<=E<=BE and BN<=N<=TN:
        x = (E-TE)/(BE-TE)*w
        y = (N-TN)/(BN-TN)*h
        break

cv2.circle(source_map_image,(int(x),int(y)),40,(255,0,0),20)
cv2.imwrite("map.jpg",source_map_image)
cv2.imwrite("query.jpg",query_image)




    

    



    


