import os
import shutil
import glob
import tqdm


def mkdirAndCopyfile(ori_path,to_path):
    if not os.path.exists(to_path):
        os.mkdir(to_path)
    basename,name = ori_path.split("/")[-2:]
    new_basename = os.path.join(to_path,basename)
    if os.path.exists(new_basename):
       shutil.rmtree(new_basename)
    os.mkdir(new_basename)
    shutil.copyfile(ori_path,os.path.join(new_basename,name))



if __name__ == '__main__':
    root = "/home/dmmm/Dataset/DenseUAV/data_2022/test"
    root_80 = os.path.join(root,"queryDrone80")
    root_90 = os.path.join(root,"queryDrone90")
    root_100 = os.path.join(root,"queryDrone100")
    root_Drone_all = os.path.join(root,"query_drone")
    Drone_80_List = glob.glob(os.path.join(root_Drone_all,"*","H80.JPG"))
    Drone_90_List = glob.glob(os.path.join(root_Drone_all,"*","H90.JPG"))
    Drone_100_List = glob.glob(os.path.join(root_Drone_all,"*","H100.JPG"))
    tq = tqdm.tqdm(len(Drone_80_List))
    for H80,H90,H100 in zip(Drone_80_List,Drone_90_List,Drone_100_List):
        mkdirAndCopyfile(H80,root_80)
        mkdirAndCopyfile(H90,root_90)
        mkdirAndCopyfile(H100,root_100)
        tq.update()


