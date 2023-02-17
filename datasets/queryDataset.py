import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from math import cos, sin, pi

class Dataset_query(Dataset):
    def __init__(self,filename,transformer,basedir):
        super(Dataset_query, self).__init__()
        self.filename = filename
        self.transformer = transformer
        self.basedir = basedir

    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self)


class Query_transforms(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, pad=20,size=256):
        self.pad=pad
        self.size = size

    def __call__(self, img):
        img_=np.array(img).copy()
        img_part = img_[:,0:self.pad,:]
        img_flip = cv2.flip(img_part, 1)  # 镜像
        image = np.concatenate((img_flip, img_),axis=1)
        image = image[:,0:self.size,:]
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        return image


class CenterCrop(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self):
        pass

    def __call__(self, img):
        img_=np.array(img).copy()
        h, w, c = img_.shape
        min_edge = min((h, w))
        if min_edge == h:
            edge_lenth = int((w - min_edge) / 2)
            new_image = img_[:, edge_lenth:w - edge_lenth, :]
        else:
            edge_lenth = int((h - min_edge) / 2)
            new_image = img_[edge_lenth:h - edge_lenth, :, :]
        assert new_image.shape[0] == new_image.shape[1], "the shape is not correct"
        cv2.imshow("query",cv2.resize(new_image,(512,512)))
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image



class RotateAndCrop(object):
    def __init__(self,rate=0.5):
        pass

    def __call__(self, img):
        img_=np.array(img).copy()

        def getPosByAngle(img, angle):
            h, w, c = img.shape
            x_center = y_center = h // 2
            r = h // 2
            angle_lt = angle - 45
            angle_rt = angle + 45
            angle_lb = angle - 135
            angle_rb = angle + 135
            angleList = [angle_lt, angle_rt, angle_lb, angle_rb]
            pointsList = []
            for angle in angleList:
                x1 = x_center + r * cos(angle * pi / 180)
                y1 = y_center + r * sin(angle * pi / 180)
                pointsList.append([x1, y1])
            pointsOri = np.float32(pointsList)
            pointsListAfter = np.float32([[0, 0], [512, 0], [0, 512], [512, 512]])
            M = cv2.getPerspectiveTransform(pointsOri, pointsListAfter)
            res = cv2.warpPerspective(img, M, (512, 512))
            return res
        if np.random.random()<0.5:
            image = img
        else:
            angle = int(np.random.random()*360)
            new_image = getPosByAngle(img_,angle)
            image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image

class RandomCrop(object):
    def __init__(self,rate=0.2):
        self.rate = rate

    def __call__(self, img):
        img_=np.array(img).copy()
        h,w,c = img_.shape
        random_width = int(np.random.random()*self.rate*w)
        random_height = int(np.random.random()*self.rate*h)
        x_l = random_width
        x_r = w-random_width
        y_l = random_height
        y_r = h-random_height
        new_image = img_[y_l:y_r,x_l:x_r,:]
        image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
        return image