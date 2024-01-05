import matplotlib.pyplot as plt
import random
from glob import glob
import numpy as np

m_50 = [] 
m_20 = []
m_10 = []
m_5 = []
m_3 = []
m = [m_3,m_5,m_10,m_20,m_50]

x_label=[]
RDS_value = []
num_a=[3,5,10,20,50]
filenames = glob("output/result_files/height-level/*.txt")
filenames.sort(key=lambda x:int(x.split("/")[-1].split("_")[0]))
for num, path in enumerate(filenames):
    x_label.append(path.split("/")[-1].split("_")[0])
    RDS_value.append(float(path.split("/")[-1].split(".txt")[0].split("=")[-1]))
    result = []
    with open(path, "r") as F:
        lines = F.readlines()
        for i,a in enumerate(num_a):
            line = lines[a-1]
            out = line.split(' ')[-1]
            out = out.split("\n")[0]
            m[i].append(float(out))



fig = plt.figure(figsize=(6, 10))
ax1 = fig.subplots()
# ax1.tick_params(axis='x', labelrotation=10)
plt.xlabel("Height (m)",fontdict={'family' : 'Times New Roman', 'size': 21})
plt.xticks(fontproperties='Times New Roman',fontsize=18)
ax1.set_ylabel('Accurary', fontdict={'family': 'Times New Roman', 'size': 21})  # 添加x轴的标签
ax1.set_ylim(0,1.1)
ax1.set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0], fontproperties='Times New Roman', fontsize=16)  # 添加x轴上的标签


# 绘制横向条形图
bar1 = ax1.bar(x_label, m_50, width=0.5, color="#45a776", label="MA@"+str(num_a[4]),edgecolor="k",linewidth=2)
bar2 = ax1.bar(x_label, m_20, width=0.5, color="#3682be", label="MA@"+str(num_a[3]),edgecolor="k",linewidth=2)
bar3 = ax1.bar(x_label, m_10, width=0.5, color="#b3974e", label="MA@"+str(num_a[2]),edgecolor="k",linewidth=2)
bar4 = ax1.bar(x_label, m_5, width=0.5, color="#eed777", label="MA@"+str(num_a[1]),edgecolor="k",linewidth=2)
bar5 = ax1.bar(x_label, m_3, width=0.5, color="#f05326", label="MA@"+str(num_a[0]),edgecolor="k",linewidth=2)

# 在横向条形图上添加数据
for (a, b) in zip(x_label, m_3):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(x_label, m_5):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(x_label, m_10):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(x_label, m_20):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
for (a, b) in zip(x_label, m_50):
    plt.text(a, b-0.03, "{:.3f}".format(b), color='black', fontsize=15, ha="center",va="bottom", fontproperties='Times New Roman')
# legend_bar = plt.legend(handles=[bar1,bar2,bar3,bar4,bar5], loc="upper left", ncol=2, prop={'family': 'Times New Roman', 'size': 16})


ax2 = ax1.twinx()
rds, = ax2.plot(x_label,RDS_value,color="red",linestyle="--", marker='*',label="RDS", markersize=16, linewidth=3)
# legend_rds = plt.legend(handles=[rds],loc = "upper right",prop={'family': 'Times New Roman', 'size': 16})
for ind, (a, b) in enumerate(zip(x_label, RDS_value)):
    bias = + 0.002
    plt.text(a, b+bias, "{:.3f}".format(b), color='darkred', fontsize=17, ha="center",va="bottom", fontproperties='Times New Roman')

ax2.set_ylabel('RDS', fontdict={'family': 'Times New Roman', 'size': 21})  # 添加x轴的标签
ax2.set_ylim(0.68,0.82)
# ax1.set_xticks(fontproperties='Times New Roman',fontsize=22)  # 添加y轴上的刻度
ax2.set_yticklabels([0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84], fontproperties='Times New Roman', fontsize=16)  # 添加x轴上的标签

# ax = ax1.add_artist(legend_bar)

legend_bar = plt.legend(handles=[bar1,bar2,bar3,bar4,bar5,rds], loc="upper left", ncol=2, prop={'family': 'Times New Roman', 'size': 16})


# 保存图像到当前文件夹下，图像名称为 image.png
plt.tight_layout()
plt.savefig('tool/visual/MA_curve/RDS_MA_height.jpg', dpi=600)
plt.savefig('tool/visual/MA_curve/RDS_MA_height.eps', dpi=600)