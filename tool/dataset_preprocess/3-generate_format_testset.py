import os
import shutil
from tqdm import tqdm
import glob
from multiprocessing import Pool


def processData(datalist,targetDir):
    os.makedirs(targetDir,exist_ok=True)
    for dir in tqdm(datalist):
        name = os.path.basename(dir)
        target_dir = os.path.join(targetDir,name)
        shutil.copytree(dir,target_dir)
    

def main():
    rootDir = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/data_2021/"
    # # 测试集
    # testDir = os.path.join(rootDir,"test")
    # ClassForTestDrone = glob.glob(os.path.join(testDir, "drone/*"))
    # ClassForTestSatellite = glob.glob(os.path.join(testDir, "satellite/*"))

    # query_drone = os.path.join(testDir,"query_drone")
    # gallery_drone = os.path.join(testDir,"gallery_drone")
    # query_satellite = os.path.join(testDir,"query_satellite")
    # gallery_satellite = os.path.join(testDir,"gallery_satellite")

    # # 训练集
    # trainDir = os.path.join(rootDir, "train")
    # ClassForTrainDrone = glob.glob(os.path.join(trainDir, "drone/*"))
    # ClassForTrainSatellite = glob.glob(os.path.join(trainDir, "satellite/*"))

    # # process test data
    # processData(ClassForTestDrone,query_drone)
    # processData(ClassForTestSatellite,query_satellite)
    # # gallery需要把测试集和训练集合在一起
    # processData(ClassForTrainDrone+ClassForTestDrone,gallery_drone)
    # processData(ClassForTrainSatellite+ClassForTestSatellite,gallery_satellite)

    # 生成最终的Dense_GPS_ALL.txt
    mode_list = ["train","test"]
    total_lines = []
    for mode in mode_list:
        mode_txt = os.path.join(rootDir,"Dense_GPS_{}.txt".format(mode))
        with open(mode_txt,"r") as F:
            lines = F.readlines()
        total_lines.extend(lines)
    ALL_filename = os.path.join(rootDir,"Dense_GPS_ALL.txt")
    with open(ALL_filename,"w") as F:
        for info in total_lines:
            F.write(info)
    print("success write {}".format(ALL_filename))


if __name__ == '__main__':
    main()