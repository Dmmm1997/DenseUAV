import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torch import nn
from .TripletLoss import Tripletloss, WeightedSoftTripletLoss


def cal_cls_loss(outputs, labels, loss_func):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def cal_kl_loss(outputs, outputs2, loss_func):
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


def cal_triplet_loss(outputs, outputs2, labels, loss_func, split_num=8):
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


def cal_loss(opt, outputs, outputs2, labels, labels3):
    cls1,feature1 = outputs
    cls2,feature2 = outputs2
    loss = 0.0
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    if opt.WSTR:
        triplet_loss = WeightedSoftTripletLoss()
    else:
        triplet_loss = Tripletloss(margin=0.3)
    # 三元组损失
    f_triplet_loss = torch.tensor((0))
    if opt.triplet_loss:
        split_num = opt.batchsize//opt.sample_num
        f_triplet_loss = cal_triplet_loss(
            feature1, feature2, labels, triplet_loss, split_num)
        loss += f_triplet_loss

    # 分类损失
    cls_loss = cal_cls_loss(cls1, labels, criterion) + \
        cal_cls_loss(cls2, labels3, criterion)
    loss += cls_loss
    # 增加klLoss来做mutual learning
    kl_loss = torch.tensor((0))
    if opt.kl_loss:
        kl_loss = cal_kl_loss(cls1, cls2, loss_kl)
        loss += kl_loss

    # if opt.epoch < opt.warm_epoch:
    #     warm_up = 0.1  # We start from the 0.1*lrRate
    #     warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch
    #     warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
    #     loss *= warm_up

    return loss, cls_loss, f_triplet_loss, kl_loss
