import matplotlib.pyplot as plt
import numpy as np

ResNet50 = [0.165, 0.343, 0.473, 0.535, 1.0]
ConvNextT = [0.6023, 0.7404, 0.7975, 0.828, 1.0]
ViTS = [0.8018, 0.888, 0.9197,  0.9335, 1.0]

y1 = ConvNextT
for i in range(len(y1)-1,0,-1):
    y1[i] = y1[i]-y1[i-1]

y2 = ViTS
for i in range(len(y2)-1,0,-1):
    y2[i] = y2[i]-y2[i-1]

y0 = ResNet50
for i in range(len(y0)-1,0,-1):
    y0[i] = y0[i]-y0[i-1]

totol_images = 2331

# 创建数据
categories = ["0","1","2","3", "other"]

bar_width = 0.3

r0 = np.arange(len(categories))
r1 = [x + bar_width for x in r0]
r2 = [x + bar_width for x in r1]

# 创建左右两个子图
plt.subplots(figsize=(7, 5))

plt.xlabel("Error on Sampling Interval",fontdict={'family' : 'Times New Roman', 'size': 17})
plt.xticks(fontproperties='Times New Roman',fontsize=15)

plt.ylabel('Proportion(%)', fontdict={'family': 'Times New Roman', 'size': 17})  # 添加x轴的标签
plt.yticks(fontproperties='Times New Roman',fontsize=15)
plt.ylim(0,0.85)

# 绘制左边的图
plt.bar(r0, y0, width=bar_width, color="#eed777", edgecolor="k",linewidth=2, label='ResNet-50')
plt.bar(r1, y1, width=bar_width, color="#45a776", edgecolor="k",linewidth=2, label='ConvNext-T')
plt.bar(r2, y2, width=bar_width, color="#b3974e", edgecolor="k",linewidth=2, label="ViT-S")
for i, value in zip(r0,y0):
    plt.text(i, value, "{:.1f}".format(value*100), ha='center', va='bottom', color="black", fontsize=13, fontproperties='Times New Roman')
for i, value in zip(r1,y1):
    plt.text(i, value, "{:.1f}".format(value*100), ha='center', va='bottom', color="black", fontsize=13, fontproperties='Times New Roman')
for i, value in zip(r2,y2):
    plt.text(i, value, "{:.1f}".format(value*100), ha='center', va='bottom', color="black", fontsize=13, fontproperties='Times New Roman')

# 添加刻度标签
plt.xticks([r + bar_width for r in range(len(categories))], categories)

plt.legend(prop={'family': 'Times New Roman', 'size': 15})

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig("SDM@K_analyze.eps", dpi = 600)


