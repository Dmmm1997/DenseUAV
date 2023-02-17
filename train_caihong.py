# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.cuda.amp import autocast,GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tool.utils import save_network,copyfiles2checkpoints
import warnings
from losses.triplet_loss import Tripletloss,WeightedSoftTripletLoss
from losses.cal_loss import cal_kl_loss,cal_loss,cal_triplet_loss

warnings.filterwarnings("ignore")
version =  torch.__version__
#fp16
try:
    from apex.fp16_utils import *
    from apex import amp, optimizers
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name',default='test1', type=str, help='output model name')
    parser.add_argument('--pool',default='max', type=str, help='pool avg')
    parser.add_argument('--data_dir',default='/home/dmmm/traincjh/train2/',type=str, help='training dir path')
    parser.add_argument('--train_all', action='store_true', help='use all training data' )
    parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training' )
    parser.add_argument('--num_worker', default=8,type=int, help='' )
    parser.add_argument('--batchsize', default=8, type=int, help='batchsize')
    parser.add_argument('--stride', default=1, type=int, help='stride')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=256, type=int, help='height')
    parser.add_argument('--w', default=256, type=int, help='width')
    parser.add_argument('--rotate', default=1, type=int, help='rotate in transform')
    parser.add_argument('--randomcrop', default=0, type=float, help='crop in transform')
    parser.add_argument('--views', default=2, type=int, help='the number of views')
    parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--moving_avg', default=1.0, type=float, help='moving average')
    parser.add_argument('--droprate', default=0.5, type=float, help='drop rate')
    parser.add_argument('--DA', action='store_true', help='use Color Data Augmentation' )
    parser.add_argument('--share', action='store_true',default=True, help='share weight between different view' )
    parser.add_argument('--extra_Google', action='store_true',default=False, help='using extra noise Google' )
    parser.add_argument('--fp16', action='store_true',default=False, help='use float16 instead of float32, which will save about 50% memory' )
    parser.add_argument('--autocast', action='store_true',default=True, help='use mix precision' )
    parser.add_argument('--transformer', default=1, type=int,help='use transformer' )
    parser.add_argument('--LPN',  default=0, type=int, help='')
    parser.add_argument('--MSBA', action='store_true', default=False, help='')
    parser.add_argument('--block', default=1, type=int, help='')
    parser.add_argument('--kl_loss', action='store_true',default=False, help='kl_loss' )
    parser.add_argument('--triplet_loss', default=1, type=int, help='')
    parser.add_argument('--WSTR', default=0, type=int, help='weighted soft triplet loss')
    parser.add_argument('--sample_num', default=3, type=int, help='num of repeat sampling' )
    parser.add_argument('--num_epochs', default=120, type=int, help='' )
    parser.add_argument('--deit',default=0,type=int,help='')
    opt = parser.parse_args()
    print(opt.lr)
    return opt


def train_model(model,opt, optimizer, scheduler, dataloaders,dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs

    since = time.time()
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction='batchmean')
    if opt.WSTR:
        triplet_loss = WeightedSoftTripletLoss()
    else:
        triplet_loss = Tripletloss(margin=0.3)
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_cls_loss = 0.0
            running_triplet = 0.0
            running_kl_loss = 0.0
            running_loss = 0.0
            running_corrects = 0.0
            running_corrects2 = 0.0
            running_corrects3 = 0.0
            # Iterate over data.
            # for data, data2, data3, data4 in zip(dataloaders['satellite'], dataloaders['street'],
            #                                      dataloaders['drone'], dataloaders['google']):
            start = time.time()
            for data,data3 in dataloaders:
                t1 = time.time()
                t1_comsume = t1-start


                loss = 0.0
                # get the inputs
                inputs, labels = data
                inputs3, labels3 = data3
                # inputs4, labels4 = data4
                now_batch_size, c, h, w = inputs.shape
                if now_batch_size < opt.batchsize:  # skip the last batch
                    continue
                if use_gpu:
                    inputs = Variable(inputs.cuda().detach())
                    inputs3 = Variable(inputs3.cuda().detach())
                    labels = Variable(labels.cuda().detach())
                    labels3 = Variable(labels3.cuda().detach())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'val':
                    with torch.no_grad():
                        outputs, outputs2 = model(inputs, inputs3)
                else:
                    if opt.views == 2:
                        with autocast():
                            outputs, outputs2 = model(inputs, inputs3)
                f_triplet_loss=torch.tensor((0))
                if opt.triplet_loss:
                    features = outputs[1]
                    features2 = outputs2[1]
                    split_num = opt.batchsize//opt.sample_num
                    f_triplet_loss = cal_triplet_loss(features,features2,labels,triplet_loss,split_num)
                    loss += f_triplet_loss

                    outputs = outputs[0]
                    outputs2 = outputs2[0]

                if isinstance(outputs,list):
                    preds = []
                    preds2 = []
                    for out,out2 in zip(outputs,outputs2):
                        preds.append(torch.max(out.data,1)[1])
                        preds2.append(torch.max(out2.data,1)[1])
                else:
                    _, preds = torch.max(outputs.data, 1)
                    _, preds2 = torch.max(outputs2.data, 1)

                if opt.views == 2:
                    cls_loss = cal_loss(outputs, labels,criterion) + cal_loss(outputs2, labels3,criterion)
                    loss += cls_loss
                    #增加klLoss来做mutual learning
                    kl_loss = torch.tensor((0))
                    if opt.kl_loss:
                        kl_loss = cal_kl_loss(outputs,outputs2,loss_kl)
                        loss += kl_loss

                # backward + optimize only if in training phase
                if epoch < opt.warm_epoch and phase == 'train':
                    warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                    loss *= warm_up

                if phase == 'train':
                    if opt.autocast:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                # statistics
                if int(version[0]) > 0 or int(version[2]) > 3:  # for the new version like 0.4.0, 0.5.0 and 1.0.0
                    running_loss += loss.item() * now_batch_size
                    running_cls_loss += cls_loss.item()*now_batch_size
                    running_triplet += f_triplet_loss.item() * now_batch_size
                    running_kl_loss += kl_loss.item() * now_batch_size
                else:  # for the old version like 0.3.0 and 0.3.1
                    running_loss += loss.data[0] * now_batch_size
                    running_cls_loss += cls_loss.data[0] * now_batch_size
                    running_triplet += f_triplet_loss.data[0] * now_batch_size
                    running_kl_loss += kl_loss.data[0] * now_batch_size


                if isinstance(preds,list) and isinstance(preds2,list):
                    running_corrects += sum([float(torch.sum(pred == labels.data)) for pred in preds])/len(preds)
                    if opt.views==2:
                        running_corrects2 += sum([float(torch.sum(pred == labels3.data)) for pred in preds2]) / len(preds2)
                else:
                    running_corrects += float(torch.sum(preds == labels.data))
                    if opt.views == 2:
                        running_corrects2 += float(torch.sum(preds2 == labels3.data))

                start = time.time()
                time_other_cost = start - t1
                # print("datatime = {};;".format(t1_comsume / (t1_comsume + time_other_cost)))


            epoch_cls_loss = running_cls_loss/dataset_sizes['satellite']
            epoch_kl_loss = running_kl_loss /dataset_sizes['satellite']
            epoch_triplet_loss = running_triplet/dataset_sizes['satellite']
            epoch_loss = running_loss / dataset_sizes['satellite']
            epoch_acc = running_corrects / dataset_sizes['satellite']
            epoch_acc2 = running_corrects2 / dataset_sizes['satellite']


            lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
            lr_other = optimizer.state_dict()['param_groups'][1]['lr']
            if opt.views == 2:
                print(
                    '{} Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                                                                                .format(phase, epoch_loss,epoch_cls_loss,epoch_kl_loss,
                                                                                        epoch_triplet_loss, epoch_acc,
                                                                                        epoch_acc2,lr_backbone,lr_other))

            # deep copy the model
            if phase == 'train':
                scheduler.step()
            if epoch % 10 == 9 and epoch>=110:
                save_network(model, opt.name, epoch)



        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print()


if __name__ == '__main__':
    opt = get_parse()
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders,class_names,dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    model = make_model(opt)

    optimizer_ft, exp_lr_scheduler = make_optimizer(model,opt)

    model = model.cuda()
    #移动文件到指定文件夹
    copyfiles2checkpoints(opt)

    if opt.fp16:
        model, optimizer_ft = amp.initialize(model, optimizer_ft, opt_level="O1")


    train_model(model,opt, optimizer_ft, exp_lr_scheduler,dataloaders,dataset_sizes)
