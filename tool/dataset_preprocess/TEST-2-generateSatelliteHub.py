# from google_interface import get_picture
import time
from google_interface_2 import get_picture
import os
from tqdm import tqdm
import numpy as np

'''
测试阶段用于产生卫星图像库
'''

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number


if __name__ == '__main__':# 30.32331706,120.37025917
    startE = 120.3524959718277       #start左上角
    startN = 30.325763481658377
    endE = 120.36312306473656      #end右下角30.32128231,120.37425118,
    endN = 30.318953289529816
    start_time = time.time()
    margin = 0.0001
    #图像的根文件夹
    dir_path = r"/media/dmmm/CE31-3598/DataSets/DenseCV_Data/satelliteHub(现科dense)"
    os.mkdir(dir_path)
    #经纬度信息存放文件夹
    infoFile = "/media/dmmm/CE31-3598/DataSets/DenseCV_Data/satelliteHub(现科dense)/PosInfo.txt"
    file = open(infoFile,"w")
    #获取文件夹中所有的jpg图像的路径
    # jpg_paths = get_fileNames(dir_path,endwith=".JPG")
    index = 0
    NorthList = np.linspace(startN, endN, int((startN - endN) / (margin)))
    EastList = np.linspace(startE,endE,int((endE-startE)/(margin)))
    pbar = tqdm(total=len(NorthList)*len(EastList))
    for North in NorthList:
        for East in EastList:
            east_left = East - margin
            east_right = East + margin
            north_top = North + margin
            north_bottom = North - margin

            # 设置新的存放的路径
            index_ = sixNumber(index)
            satellite_tif_path = os.path.join(dir_path,"{}.tif".format(index_))
            get_picture(east_left, north_top, east_right, north_bottom, 19, satellite_tif_path, server="Google")
            file.write("{} N{:.10f} E{:.10f}\n".format(index_,North,East))
            index+=1
            pbar.update()

file.close()
pbar.close()