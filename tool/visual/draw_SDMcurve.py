import matplotlib.pyplot as plt
import os
import json
from glob import glob
from matplotlib import font_manager

output_dir = "/home/dmmm/VscodeProject/demo_DenseUAV/visualization/SDMCurve/Backbone"

os.makedirs(output_dir, exist_ok = True)

plt.rcParams['font.sans-serif'] = ['Times New Roman']

source_dir = {
    # backbone
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_ConvnextT":"ConvNeXt-T",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_DeitS":"DeiT-S",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_EfficientNet-B3":"EfficientNet-B3",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_EfficientNet-B5":"EfficientNet-B5",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_PvTv2b2":"PvTv2-B2",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_resnet50":"ResNet50",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_Swinv2T-256":"Swinv2-T",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-Global":"ViT-S",
    "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Backbone_Experiment_ViTB":"ViT-B",
    # head
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-MaxPool":"MaxPool",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-AvgMaxPool":"AvgMaxPool",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-AvgPool":"AvgPool",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-Global":"Global",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-GeM":"GemPool",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-FSRA2B":"FSRA(block=2)",
    # "/home/dmmm/VscodeProject/demo_DenseUAV/checkpoints/Head_Experiment-LPN2B":"LPN(Block=2)",
}

json_name = "SDM*.json"

fig = plt.figure(figsize=(4,4))
plt.grid()
plt.yticks(fontproperties='Times New Roman', size=12)
plt.xticks(fontproperties='Times New Roman', size=12)

x = list(range(1,101))

color = ['r', 'k', 'y', 'c', 'g', 'm', 'b', 'coral', 'tan']
ind = 0
for path, name in source_dir.items():
    print(name)
    target_file = glob(os.path.join(path, json_name))[0]
    with open(target_file, 'r') as F:
        data = json.load(F)
    y = list(data.values())
    plt.plot(x,y,c=color[ind],marker = 'o',label=name,linewidth=1.0,markersize=1)
    ind+=1

plt.legend(loc="upper right",prop={'family' : 'Times New Roman', 'size'   : 12})
plt.ylabel("SDM@K",fontdict={'family' : 'Times New Roman', 'size': 12})
plt.xlabel("K",fontdict={'family' : 'Times New Roman', 'size': 12})
plt.tight_layout()


fig.savefig(os.path.join(output_dir, "backbone.eps"), dpi=600, format='eps')
plt.show()

