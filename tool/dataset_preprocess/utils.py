import glob
import os
import math

def get_fileNames(rootdir,endwith=".JPG"):
    fs = []
    for root, dirs, files in os.walk(rootdir,topdown = True):
        for name in files:
            _, ending = os.path.splitext(name)
            if ending == endwith:
                fs.append(os.path.join(root,name))
    return fs


def Distance(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS = 6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a)
                                  * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance