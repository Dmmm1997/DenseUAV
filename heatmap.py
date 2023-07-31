import os
# import sys
# sys.path.insert(0,"./")
import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tool.utils import load_network
import yaml
import argparse
import torch
from torchvision import datasets, models, transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='Training')
import math

parser.add_argument('--data_dir',default='/home/dmmm/Dataset/DenseUAV/data_2022/test',type=str, help='./test_data')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--checkpoint',default="net_119.pth", help='weights' )
parser.add_argument('--platform',default="satellite", help='weights' )
opt = parser.parse_args()

config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
for cfg, value in config.items():
    if cfg not in opt:
        setattr(opt, cfg, value)


def heatmap2d(img, arr):
    # fig = plt.figure()
    # ax0 = fig.add_subplot(121, title="Image")
    # ax1 = fig.add_subplot(122, title="Heatmap")
    # fig, ax = plt.subplots(）
    # ax[0].imshow(Image.open(img))
    plt.figure()
    heatmap = plt.imshow(arr, cmap='viridis')
    plt.axis('off')
    # fig.colorbar(heatmap, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('heatmap_dbase')

data_transforms = transforms.Compose([
        transforms.Resize((opt.h, opt.w), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

model = load_network(opt)
model = model.eval().cuda()

print(opt.data_dir)
for i in ["000090","000013","000015","000016","000018","000035","000039","000116","000130"]:
    print(i)
    imgpath = os.path.join(opt.data_dir,"gallery_{}/{}".format(opt.platform,i))
    imgpath = os.path.join(imgpath, "H100.JPG" if opt.platform == "drone" else "H100_old.tif")
    print(imgpath)
    img = Image.open(imgpath)
    img = data_transforms(img)
    img = torch.unsqueeze(img,0)

    with torch.no_grad():
        # print(model)
        features = model.backbone(img.cuda())
        # pos_embed = model.backbone.pos_embed
        if opt.backbone=="resnet50":
            output = features
        else:
            part_features = features[:,1:]
            part_features = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
            output = part_features.permute(0,3,1,2)

    heatmap = output.squeeze().sum(dim=0).cpu().numpy()
    # print(heatmap.shape)
    # print(heatmap)
    # heatmap = np.mean(heatmap, axis=0)
    #
    # heatmap = np.maximum(heatmap, 0)
    heatmap = normalization(heatmap)
    img = cv2.imread(imgpath)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    heatmap = cv2.applyColorMap(heatmap, 2)  # 将热力图应用于原始图像model.py
    ratio = 0.8 if opt.platform == "drone" else 0.3
    superimposed_img = heatmap * ratio + img  # 这里的0.4是热力图强度因子
    if not os.path.exists("heatout"):
        os.mkdir("./heatout")
    save_file = "./heatout/{}_{}.jpg".format(opt.platform,i)
    cv2.imwrite(save_file, superimposed_img)