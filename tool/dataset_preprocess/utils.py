import glob
import os

def get_fileNames(rootdir,endwith=".JPG"):
    fs = []
    for root, dirs, files in os.walk(rootdir,topdown = True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == endwith:
                fs.append(os.path.join(root,name))
    return fs
