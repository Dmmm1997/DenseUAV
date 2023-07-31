import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
#######################################################################
# Evaluate
parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=77, type=int, help='test_image_index')
parser.add_argument('--test_dir',default='/home/dmmm/Dataset/DenseUAV/data_2022/test',type=str, help='./test_data')
parser.add_argument('--config',
                    default="/home/dmmm/Dataset/DenseUAV/data_2022/Dense_GPS_ALL.txt", type=str,
                    help='./test_data')
opts = parser.parse_args()

configDict = {}
with open(opts.config, "r") as F:
    context = F.readlines()
    for line in context:
        splitLineList = line.split(" ")
        configDict[splitLineList[0].split("/")[-2]] = [float(splitLineList[1].split("E")[-1]),
                                                       float(splitLineList[2].split("N")[-1])]

gallery_name = 'gallery_satellite'
query_name = 'query_drone'
# gallery_name = 'gallery_drone'
# query_name = 'query_satellite'

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ) for x in [gallery_name, query_name]}

#####################################################################
#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)  # pause a bit so that plots are updated

######################################################################
result = scipy.io.loadmat('pytorch_result_1.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_label = result['query_label'][0]
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_label = result['gallery_label'][0]

multi = os.path.isfile('multi_query.mat')

if multi:
    m_result = scipy.io.loadmat('multi_query.mat')
    mquery_feature = torch.FloatTensor(m_result['mquery_f'])
    mquery_cam = m_result['mquery_cam'][0]
    mquery_label = m_result['mquery_label'][0]
    mquery_feature = mquery_feature.cuda()

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()

#######################################################################
# sort the images
def sort_img(qf, ql, gf, gl):
    query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  #from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)

    #good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index = np.argwhere(gl==-1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index

i = opts.query_index
index = sort_img(query_feature[i],query_label[i],gallery_feature,gallery_label)

########################################################################
# Visualize the rank result
query_path, _ = image_datasets[query_name].imgs[i]
query_label = query_label[i]
print(query_path)
label6num = query_path.split("/")[-2]
x_q,y_q = configDict[label6num]

print('Top 10 images are as follow:')
save_folder = 'image_show/%02d' % opts.query_index
if not os.path.exists("image_show"):
    os.mkdir("image_show")
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)
os.system('cp %s %s/query.jpg'%(query_path, save_folder))

try: # Visualize Ranking Result 
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    imshow(query_path)
    ax.set_title("x:{:.7f}\ny:{:.7f}".format(x_q,y_q), color='blue',fontsize=5)
    for i in range(10):
        ax = plt.subplot(1,11,i+2)
        ax.axis('off')
        img_path, _ = image_datasets[gallery_name].imgs[index[i]]
        label = gallery_label[index[i]]
        labelg6num = img_path.split("/")[-2]
        x_g,y_g = configDict[label6num]
        print(label)
        imshow(img_path)
        os.system('cp %s %s/s%02d.tif'%(img_path, save_folder, i))
        if label == query_label:
            ax.set_title("x:{:.7f}\ny:{:.7f}".format(x_g,y_g),color='green',fontsize=5)
        else:
            ax.set_title("x:{:.7f}\ny:{:.7f}".format(x_g,y_g), color='red',fontsize=5)
        print(img_path)
    #plt.pause(100)  # pause a bit so that plots are updated
except RuntimeError:
    for i in range(10):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig(save_folder+"/show.png",dpi = 600)

