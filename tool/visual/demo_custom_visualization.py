import matplotlib.pyplot as plt
import json


with open("test.json", "r", encoding='utf-8') as f:
    data = json.loads(f.read())    # load的传入参数为字符串类型

K = data["K"]
query = data["query"]
gallery = data["gallery"]

#Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    # plt.pause(0.1)  # pause a bit so that plots are updated

##报qt的错误
try:
    fig = plt.figure(figsize=(12, 4))
    ax = plt.subplot(1, K+1, 1)
    ax.axis('off')
    imshow(query, 'query')
    for i,path in enumerate(gallery):
        ax = plt.subplot(1, 11, i + 2)
        ax.axis('off')
        imshow(path)
except:
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')

fig.savefig("show.png",dpi=600)
