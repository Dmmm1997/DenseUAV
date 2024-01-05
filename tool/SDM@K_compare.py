import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib as mpl
mpl.rcParams['font.family'] = 'Times New Roman'


def evaluateSingle(distance, K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*5e3)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def evaluate_enclidean(distance, K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0, K, 1))/K
    # m1 = distance / maxDistance
    m2 = 1-distance*1e3
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def Recall_Data(data):
    x_len, y_len = data.shape
    data = np.zeros_like(data)
    data[x_len//2, y_len//2] = 1
    return data

def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def SDM_Data(data):
    x_len, y_len = data.shape
    x = np.linspace(120.358111-0.0003, 120.358111+0.0003, x_len)
    y = np.linspace(30.317842-0.0003, 30.317842+0.0003, y_len)
    x_,y_ = np.meshgrid(x,y)
    x_ = x_.reshape(-1,1)
    y_ = y_.reshape(-1,1)
    input = np.concatenate((x_,y_),axis=-1)

    target = np.array((120.358111,30.317842)).reshape(-1,2)
    distance = euclideanDistance(input, target)
    # compute single query evaluate result
    P_list = np.array([evaluateSingle(dist_single, 1) for dist_single in distance])
    return P_list.reshape(x_len,y_len)


def Enclidean_Data(data):
    x_len, y_len = data.shape
    x = np.linspace(120.358111-0.0003, 120.358111+0.0003, x_len)
    y = np.linspace(30.317842-0.0003, 30.317842+0.0003, y_len)
    x_,y_ = np.meshgrid(x,y)
    x_ = x_.reshape(-1,1)
    y_ = y_.reshape(-1,1)
    input = np.concatenate((x_,y_),axis=-1)

    target = np.array((120.358111,30.317842)).reshape(-1,2)
    distance = euclideanDistance(input, target)
    # compute single query evaluate result
    P_list = np.array([evaluate_enclidean(dist_single, 1) for dist_single in distance])
    return P_list.reshape(x_len,y_len)



# 创建一个随机的2D数组作为网格数据
data = np.random.rand(7, 7)

Recall_data = Recall_Data(data)
SDM_data = SDM_Data(data)
Enclidean_data = Enclidean_Data(data)


# 创建一个figure和axes对象
fig, ax = plt.subplots(1,3,figsize=(14,4))

# 使用imshow函数显示网格数据
img1 = ax[0].imshow(Recall_data, cmap='coolwarm', interpolation='nearest')
img2 = ax[2].imshow(SDM_data, cmap='coolwarm', interpolation='nearest')
img3 = ax[1].imshow(Enclidean_data, cmap='coolwarm', interpolation='nearest')


# 在每个格子中间显示数值
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ax[0].text(j, i, f'{Recall_data[i, j]:.1f}', ha='center', va='center', color='white')
        ax[2].text(j, i, f'{SDM_data[i, j]:.1f}', ha='center', va='center', color='white')
        ax[1].text(j, i, f'{Enclidean_data[i, j]:.1f}', ha='center', va='center', color='white')

for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    a.set_xticklabels([])
    a.set_yticklabels([])

# 添加颜色条
cbar = plt.colorbar(img1)


# 设置子图标题
ax[0].set_title('(a) Recall', fontsize=16, pad=10)
ax[2].set_title('(b) SDM', fontsize=16, pad=10)
ax[1].set_title('(c) Euclidean Distance', fontsize=16, pad=10)

plt.subplots_adjust(wspace=0.0, hspace=0.000)

# 调整布局，确保子图之间的间距合适
plt.tight_layout()



# ax[0].grid(True, which='both', linestyle='-', linewidth=2, color='black')
# ax[1].grid(True, which='both', linestyle='-', linewidth=2, color='black')
# ax[2].grid(True, which='both', linestyle='-', linewidth=2, color='black')

plt.savefig("SDM_Recall_Enclidean_compare.eps", dpi=300)
