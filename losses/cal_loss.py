import torch.nn.functional as F
from torch.autograd import Variable
import torch

def cal_loss(outputs,labels,loss_func):
    loss = 0
    if isinstance(outputs,list):
        for i in outputs:
            loss += loss_func(i,labels)
        loss = loss/len(outputs)
    else:
        loss = loss_func(outputs,labels)
    return loss

def cal_kl_loss(outputs,outputs2,loss_func):
    loss = 0
    if isinstance(outputs,list):
        for i in range(len(outputs)):
            loss += loss_func(F.log_softmax(outputs[i], dim=1),
                               F.softmax(Variable(outputs2[i]), dim=1))
        loss = loss/len(outputs)
    else:
        loss = loss_func(F.log_softmax(outputs, dim=1),
                          F.softmax(Variable(outputs2), dim=1))
    return loss

def cal_triplet_loss(outputs,outputs2,labels,loss_func,split_num=8):
    if isinstance(outputs,list):
        loss = 0
        for i in range(len(outputs)):
            out_concat = torch.cat((outputs[i], outputs2[i]), dim=0)
            labels_concat = torch.cat((labels,labels),dim=0)
            loss += loss_func(out_concat,labels_concat)
        loss = loss/len(outputs)
    else:
        out_concat = torch.cat((outputs, outputs2), dim=0)
        labels_concat = torch.cat((labels,labels),dim=0)
        loss = loss_func(out_concat,labels_concat)
    return loss