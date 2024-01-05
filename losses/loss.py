import torch
from torch import nn
from .TripletLoss import SameDomainTripletLoss, WeightedSoftTripletLoss, HardMiningTripletLoss, TripletLoss
from .FocalLoss import FocalLoss
from pytorch_metric_learning import losses, miners  # pip install pytorch-metric-learning
import torch.nn.functional as F
from torch.autograd import Variable

class Loss(nn.Module):
    def __init__(self, opt) -> None:
        super(Loss,self).__init__()
        self.opt = opt
        # 分类损失
        if opt.cls_loss == "CELoss":
            self.cls_loss = nn.CrossEntropyLoss()
        elif opt.cls_loss == "FocalLoss":
            self.cls_loss = FocalLoss(alpha=0.25, gamma=2, num_classes = opt.nclasses)
        else:
            self.cls_loss = None

        # 对比损失
        if opt.feature_loss == "TripletLoss":
            self.feature_loss = TripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "HardMiningTripletLoss":
            self.feature_loss = HardMiningTripletLoss(margin=0.3, normalize_feature=True)
        elif opt.feature_loss == "SameDomainTripletLoss":
            self.feature_loss = SameDomainTripletLoss(margin=0.3)
        elif opt.feature_loss == "WeightedSoftTripletLoss":
            self.feature_loss = WeightedSoftTripletLoss()
        elif opt.feature_loss == "ContrastiveLoss":
            self.feature_loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        else:
            self.feature_loss = None

        # KL 损失
        if opt.kl_loss == "KLLoss":
            self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        else:
            self.kl_loss = None
        

    def forward(self, outputs, outputs2, labels, labels2):
        cls1,feature1 = outputs
        cls2,feature2 = outputs2
        loss = 0

        # 分类损失
        res_cls_loss = torch.tensor((0))
        if self.cls_loss is not None:
            res_cls_loss = self.calc_cls_loss(cls1, labels, self.cls_loss) + \
                self.calc_cls_loss(cls2, labels2, self.cls_loss)
            loss += res_cls_loss

        # 特征对比损失
        res_triplet_loss = torch.tensor((0))
        if self.feature_loss is not None:
            split_num = self.opt.batchsize//self.opt.sample_num
            res_triplet_loss = self.calc_triplet_loss(
                feature1, feature2, labels, self.feature_loss, split_num)
            loss += res_triplet_loss

        # 相互学习
        res_kl_loss = torch.tensor((0))
        if self.kl_loss is not None:
            res_kl_loss = self.calc_kl_loss(cls1, cls2, self.kl_loss)
            loss += res_kl_loss

        # if self.opt.epoch < self.opt.warm_epoch:
        #     warm_up = 0.1  # We start from the 0.1*lrRate
        #     warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
        #     warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
        #     loss *= warm_up

        return loss, res_cls_loss, res_triplet_loss, res_kl_loss
    

    def calc_cls_loss(self, outputs, labels, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in outputs:
                loss += loss_func(i, labels)
            loss = loss/len(outputs)
        else:
            loss = loss_func(outputs, labels)
        return loss


    def calc_kl_loss(self, outputs, outputs2, loss_func):
        loss = 0
        if isinstance(outputs, list):
            for i in range(len(outputs)):
                loss += loss_func(F.log_softmax(outputs[i], dim=1),
                                F.softmax(Variable(outputs2[i]), dim=1))
            loss = loss/len(outputs)
        else:
            loss = loss_func(F.log_softmax(outputs, dim=1),
                            F.softmax(Variable(outputs2), dim=1))
        return loss


    def calc_triplet_loss(self, outputs, outputs2, labels, loss_func, split_num=8):
        if isinstance(outputs, list):
            loss = 0
            for i in range(len(outputs)):
                out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
                labels_concat = torch.cat((labels, labels), dim=0)
                loss += loss_func(out_concat, labels_concat)
            loss = loss/len(outputs)
        else:
            out_concat = torch.cat((outputs, outputs2), dim=0)
            labels_concat = torch.cat((labels, labels), dim=0)
            loss = loss_func(out_concat, labels_concat)
        return loss


