import cv2
import numpy as np
from math import cos,sin,pi
from PIL import Image
import os

#生成随机rotateandcrop的结果并保存到rotateShow文件夹下


class RotateAndCrop(object):
    def __init__(self):
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

        if not os.path.exists("rotateShow"):
            os.mkdir("rotateShow")
        img.save("./rotateShow/ori.png")
        for i in range(10):
            angle = int(np.random.random() * 360)
            new_image = getPosByAngle(img_,angle)
            image = Image.fromarray(new_image.astype('uint8')).convert('RGB')
            image.save("./rotateShow/{}.png".format(i))

if __name__ == '__main__':
    img = Image.open("/media/dmmm/CE31-3598/DataSets/DenseCV_Data/实际测试图像(计量)/part2/DJI_0317.JPG")
    rotate = RotateAndCrop()
    rotate(img)



