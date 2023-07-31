import os

import cv2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from tool.utils_server import load_network
import yaml
import argparse
import torch
from torchvision import datasets, models, transforms
from PIL import Image
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='Training')
import math

parser.add_argument('--data_dir',default='/media/dmmm/CE31-3598/DataSets/DenseCV_Data/improvedOriData/test',type=str, help='./test_data')
parser.add_argument('--name', default='from_transreid_256_4B_small_lr005_kl', type=str, help='save model path')
parser.add_argument('--batchsize', default=1, type=int, help='batchsize')
parser.add_argument('--checkpoint',default="net_119.pth", help='weights' )
opt = parser.parse_args()

config_path = 'opts.yaml'
with open(config_path, 'r') as stream:
    config = yaml.load(stream)
opt.stride = config['stride']
opt.views = config['views']
opt.transformer = config['transformer']
opt.pool = config['pool']
opt.views = config['views']
opt.LPN = config['LPN']
opt.block = config['block']
opt.nclasses = config['nclasses']
opt.droprate = config['droprate']
opt.share = config['share']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']
if 'nclasses' in config: # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 751


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

# image_datasets = {x: datasets.ImageFolder( os.path.join(opt.data_dir,x) ,data_transforms) for x in ['satellite']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
#                                              shuffle=False, num_workers=1) for x in ['satellite']}

# imgpath = image_datasets['satellite'].imgs
# print(imgpath)
from glob import glob
print(opt.data_dir)
#list = os.listdir(os.path.join(opt.data_dir,"gallery_drone"))
for i in ["000009","000013","000015","000016","000018","000035","000039","000116","000130"]:
    print(i)
    imgpath = os.path.join(opt.data_dir,"gallery_drone/"+i)
    #imgname = 'gallery_drone/0726/image-28.jpeg'
    # imgname = 'query_satellite/0721/0721.jpg'
    #imgpath = os.path.join(opt.data_dir,imgname)
    imgpath = os.path.join(imgpath, "1.JPG")
    print(imgpath)
    img = Image.open(imgpath)
    img = data_transforms(img)
    img = torch.unsqueeze(img,0)
    #print(img.shape)
    model = load_network(opt)

    model = model.eval().cuda()

    # data = next(iter(dataloaders['satellite']))
    # img, label = data
    with torch.no_grad():
        # x = model.model_1.model.conv1(img.cuda())
        # x = model.model_1.model.bn1(x)
        # x = model.model_1.model.relu(x)
        # x = model.model_1.model.maxpool(x)
        # x = model.model_1.model.layer1(x)
        # x = model.model_1.model.layer2(x)
        # x = model.model_1.model.layer3(x)
        # output = model.model_1.model.layer4(x)
        features = model.model_1.transformer(img.cuda())
        part_features = features[:,1:]
        part_features = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
        output = part_features.permute(0,3,1,2)
    #print(output.shape)
    heatmap = output.squeeze().sum(dim=0).cpu().numpy()
    # print(heatmap.shape)
    # print(heatmap)
    # heatmap = np.mean(heatmap, axis=0)
    #
    # heatmap = np.maximum(heatmap, 0)
    # heatmap /= np.max(heatmap)
    heatmap = (heatmap - np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))

    #print(heatmap)
    # cv2.imshow("1",heatmap)
    # cv2.waitKey(0)
    #test_array = np.arange(100 * 100).reshape(100, 100)
    # Result is saved tas `heatmap.png`
    img = cv2.imread(imgpath)  # 用cv2加载原始图像
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # 将热力图的大小调整为与原始图像相同
    heatmap = np.uint8(255 * heatmap)  # 将热力图转换为RGB格式
    #print(heatmap)
    heatmap = cv2.applyColorMap(heatmap, 2)  # 将热力图应用于原始图像model.py
    superimposed_img = heatmap * 0.8 + img  # 这里的0.4是热力图强度因子
    if not os.path.exists("heatout"):
        os.mkdir("./heatout")
    cv2.imwrite("./heatout/"+i+".jpg", superimposed_img)
#heatmap2d(imgpath,heatmap)