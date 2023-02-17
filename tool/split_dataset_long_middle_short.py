import os
import shutil
import glob
base_dir = "/home/dmmm/University-Release"

class MakeDataset():
    def __init__(self,name):
        self.name = name
        self.dir = os.path.join(base_dir, name)
        self.ori_dir = os.path.join(base_dir,"test")
        self.target_path = os.path.join(self.dir,"query_drone")
        # 移动query_drone的图片
        self.copy_pictures()
        # 移动其他需要的文件
        self.copy_other()


    def copy_pictures(self):
        class_list = os.listdir(os.path.join(self.ori_dir,"query_drone"))
        for i in class_list:
            #新建分类对应的文件夹
            self.mkdir(os.path.join(self.target_path,i))

            #取出每个Long Middle Short对应的图片
            path = os.path.join(self.ori_dir,"query_drone",i)
            tar_path = os.path.join(self.target_path,i)
            img_list = os.listdir(path)
            img_list.sort()
            long_num = len(img_list)//3
            middle_num = len(img_list)//3*2
            short_num = len(img_list)
            if self.name =="Long":
                list = img_list[:long_num]
            elif self.name == "Middle":
                list = img_list[long_num:middle_num]
            elif self.name == "Short":
                list = img_list[middle_num:short_num]
            else:
                raise ValueError("输入的name参数有误，必须为Long、Middle或Short")

            #复制图片到指定路径
            for j in list:
                path_j = os.path.join(path,j)
                path_t = os.path.join(tar_path,j)
                shutil.copyfile(path_j,path_t)




    def mkdir(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def copy_other(self):
        filename_list = ["gallery_drone","gallery_satellite","query_satellite"]
        for i in filename_list:
            source_path = os.path.join(self.ori_dir,i)
            target_path = os.path.join(self.dir,i)
            shutil.copytree(source_path, target_path)




if __name__ == '__main__':
    MakeDataset("Long")
    MakeDataset("Middle")
    MakeDataset("Short")