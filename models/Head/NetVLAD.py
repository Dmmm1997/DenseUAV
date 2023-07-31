import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ClassBlock, Pooling, vector2image


class NetVLAD(nn.Module):
    def __init__(self, opt) -> None:
        super(NetVLAD, self).__init__()
        self.opt = opt
        self.classifier = ClassBlock(
            int(opt.in_planes*opt.block), opt.nclasses, opt.droprate, num_bottleneck=opt.num_bottleneck)
        self.netvlad = NetVLAD_block(
            num_clusters=opt.block, dim=opt.in_planes, alpha=100.0, normalize_input=True)

    def forward(self, features):
        local_feature = features[:, 1:]
        local_feature = local_feature.transpose(1, 2)

        local_feature = vector2image(local_feature, dim=2)
        local_features = self.netvlad(local_feature)

        cls, feature = self.classifier(local_features)
        return [cls, feature]


class NetVLAD_block(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD_block, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(
            torch.rand(num_clusters, dim))  # 聚类中心，参见注释1
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):  # x: (N, C, H, W), H * W对应论文中的N表示局部特征的数目，C对应论文中的D表示特征维度
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim，使用L2归一化特征维度

        # soft-assignment
        # (N, C, H, W)->(N, num_clusters, H, W)->(N, num_clusters, H * W)
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # (N, num_clusters, H * W)  # 参见注释3
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)  # (N, C, H, W) -> (N, C, H * W)

        # calculate residuals to each clusters
        # 减号前面前记为a，后面记为b, residual = a - b
        # a: (N, C, H * W) -> (num_clusters, N, C, H * W) -> (N, num_clusters, C, H * W)
        # b: (num_clusters, C) -> (H * W, num_clusters, C) -> (num_clusters, C, H * W)
        # residual: (N, num_clusters, C, H * W) 参见注释2
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -
                                  1, -1).permute(1, 2, 0).unsqueeze(0)

        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        # (N, num_clusters, C, H * W) -> (N, num_clusters, C)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # flatten；vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = vlad.view(x.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
