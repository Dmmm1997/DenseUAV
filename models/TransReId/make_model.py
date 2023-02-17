import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
# from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import math
import torch.nn.functional as F


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x,f
            else:
                x = self.classifier(x)
                return x
        else:
            return x

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)

    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

class Gem_heat(nn.Module):
    def __init__(self, dim = 768, p=3, eps=1e-6):
        super(Gem_heat, self).__init__()
        self.p = nn.Parameter(torch.ones(dim) * p)  # initial p
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)


    def gem(self, x, p=3, eps=1e-6):
        # x = torch.transpose(x, 1, -1)
        p = F.softmax(p).unsqueeze(-1)
        x = torch.matmul(x,p)
        # x = torch.transpose(x, 1, -1)
        # x = F.avg_pool1d(x, x.size(-1))
        x = x.view(x.size(0), x.size(1))
        # x = x.pow(1. / p)
        return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    # classname = m.__class__.__name__
    # if classname.find('Linear') != -1:
    #     nn.init.normal_(m.weight, std=0.001)
    #     if m.bias:
    #         nn.init.constant_(m.bias, 0.0)
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, factory,block = 4 ,return_f=False,deit=False):
        super(build_transformer, self).__init__()
        self.return_f = return_f
        # model_path = "/home/dmmm/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth"
        # small
        model_path = "/home/dmmm/checkpoints/vit_small_p16_224-15ec54c9.pth"
        transformer_name = "vit_small_patch16_224_TransReID"
        self.in_planes = 768
        if deit:
            model_path = "/home/dmmm/.cache/torch/hub/checkpoints/deit_small_distilled_patch16_224-649709d9.pth"
            transformer_name = "deit_small_patch16_224_TransReID"
            self.in_planes = 384
        pretrain_choice = "imagenet"
        # small



        print('using Transformer_type: {} as a backbone'.format(transformer_name))

        self.transformer = factory[transformer_name](img_size=(256,256), sie_xishu=3.0,
                                                        camera=0, view=0, stride_size=[16, 16], drop_path_rate=0.1,
                                                        drop_rate= 0.0,
                                                        attn_drop_rate=0.0)

        if pretrain_choice == 'imagenet':
            self.transformer.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.pooling = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.num_classes = num_classes

        # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        # self.classifier.apply(weights_init_classifier)
        #
        # self.bottleneck = nn.BatchNorm1d(self.in_planes)
        # self.bottleneck.bias.requires_grad_(False)
        # self.bottleneck.apply(weights_init_kaiming)

        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.classifier2 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        # self.mask_branch = classif
        self.block = block
        self.mask_block = block
        for i in range(self.block):
            name = 'classifier_lpn' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))
        for i in range(self.block):
            name = 'classifier_heat' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))
        for i in range(5):
            name = 'classifier_merge' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))
        self.sort_Linear = nn.Linear(self.in_planes, 256)
        self.sort_Linear.apply(weights_init_classifier)
        self.mask_Linear = nn.Linear(self.in_planes, self.mask_block)

    def forward(self, x):
        # 仅仅让第一个分支去分类，其他分支做tripletloss
        features, all_features = self.transformer(x)

        # tranformer_feature = self.classifier1(features[:, 0])
        # if self.block == 0:
        #     return tranformer_feature

        # sort_feature = self.sort_Linear(features[:,0])

        # cls_results = self.pool_and_classifier(features[:,1:])
        part_features = features[:, 1:]
        # global_feature
        # part_features_ = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
        # part_features_ = part_features_.permute(0,3,1,2)
        # global_feature = self.gap(part_features_).view(part_features_.size(0),-1)
        # global_feature = self.classifier2(global_feature)
        # kmeans branch
        # LPN_result = self.Kmean_pool(part_features)
        # mask branch
        # mask = self.mask_Linear(part_features)
        # LPN_result = self.get_heartmap_pool(mask, part_features)
        # heatmap sort
        heat_result = self.get_heartmap_pool(part_features)
        # other_feature = self.classifier2(other_feature)
        # original LPN
        # part_features = part_features.view(part_features.size(0),int(math.sqrt(part_features.size(1))),int(math.sqrt(part_features.size(1))),part_features.size(2))
        # part_features = part_features.permute(0,3,1,2)
        # LPN_result = self.get_part_pool(part_features).squeeze()
        # merge
        # merge_result = self.merge_block(LPN_result)
        # y = self.part_classifier(5,merge_result, cls_name='classifier_merge')

        y = self.part_classifier(self.block, heat_result, cls_name='classifier_heat')

        if self.training:
            y = y + [tranformer_feature]

            if self.return_f:
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return cls, features
        else:
            # global_feature = global_feature.view(global_feature.size(0), -1, 1)
            tranformer_feature = tranformer_feature.view(tranformer_feature.size(0), -1, 1)
            # other_feature = other_feature.view(other_feature.size(0),-1,1)
            # y = torch.cat((y,tranformer_feature), dim=2)
            y = torch.cat([y, tranformer_feature], dim=2)

        return y

    def learn_from_heatmap(self, part_features):
        labels = torch.tensor(part_features.size(0), part_features.size(1))
        heatmap = torch.mean(part_features, dim=-1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        arg_chunk = arg.chunk(self.block, dim=1)
        for i, arg_ in enumerate(arg_chunk):
            labels[arg_] = i

        # mask分支去预测热力图
        mask = self.mask_branch(part_features)
        # 分类计算损失
        loss = F.cross_entropy(heatmap, labels)

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        # new add
        # heatmap = torch.mean(part_features, dim=-1)
        # size1 = part_features.size(1)
        # size0 = part_features.size(0)
        # size2 = part_features.size(2)
        #
        # mu = torch.mean(heatmap, dim = -1,keepdim=True)
        # std = torch.std(heatmap, dim = -1,keepdim=True)
        # heatmap = (heatmap-mu)/std
        #
        #
        # heatmap = torch.sigmoid(heatmap)
        # heatmap1 = (part_features*(heatmap>0.5).view(size0,size1,1).repeat(1,1,size2)).view(size0, size1, -1).view(size0, -1, size1).view(size0,-1,16,16)
        # heatmap2 = (part_features*(heatmap<=0.5).view(size0,size1,1).repeat(1,1,size2)).view(size0, size1, -1).view(size0, -1, size1).view(size0,-1,16,16)
        # block1 = self.pooling(heatmap1).view(size0,-1)
        # block2 = self.pooling(heatmap2).view(size0,-1)
        # split_list = [block1,block2]

        # orignal
        heatmap = torch.mean(part_features, dim=-1)
        size =part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :] for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)
        # x_sort,_ = torch.sort(x_sort,dim=1,descending=True)
        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)
        # split_list = [torch.mean(split, dim=1) for split in split_x]
        split_list = [torch.mean(split, dim=1) for split in split_x]

        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

        # mask = mask.view(mask.size(0),mask.size(1),-1)
        # soft_max_out = torch.softmax(mask,dim=2)
        # _,cls_arg = torch.max(soft_max_out,dim=2)
        #
        # # 切块part_features中热力图
        # v,_ = torch.max(part_features,dim=2)
        # v_sort,arg = torch.sort(v,dim=1,descending=True)
        # sort_feature = torch.stack([part_features[i,arg[i],:] for i in range(v.size(0))],dim=0)
        # chunk_v = torch.chunk(sort_feature,self.mask_block,dim=1)
        #
        # blocks = []
        # for j in range(self.mask_block):
        #     bat = []
        #     for i in range(mask.size(0)):
        #         feat = part_features[i,cls_arg[i]==j,:]
        #         chunk_feat = chunk_v[j][i]
        #         if feat==None:
        #             feat = chunk_feat
        #         else:
        #             feat = torch.cat((feat,chunk_feat),dim=0)
        #         max_feat,_ = torch.max(feat,dim=0)
        #         bat.append(max_feat)
        #     block = torch.stack(bat,dim=0)
        #     blocks.append(block)
        # return torch.stack(blocks,dim=2)

    def pool_and_classifier(self, features):
        N = features.size(1)
        margin = int((N - 1) / 4)
        cls_results = []
        for i in range(4):
            feat = features[:, i * margin:(i + 1) * margin]
            feat = torch.mean(feat, dim=1).view(feat.size(0), feat.size(-1))
            name = 'classifier' + str(i + 1)
            func = getattr(self, name)
            cls_r = func(feat)
            cls_results.append(cls_r)
        return cls_results

    def Kmean_pool(self, x):
        kmeans = KMeans(n_clusters=self.block)
        # x_ = torch.max(x,dim=-1)[0]
        x_ = torch.mean(x, dim=-1)
        b_feature = []
        for i in range(x_.size(0)):
            kmeans_ = kmeans.fit(x_[i].view(-1, 1).cpu().detach())
            blocks = []
            for j in range(self.block):
                y = x[i, kmeans_.labels_ == j, :]
                y = torch.mean(y, dim=0)
                blocks.append(y)
            blocks = torch.stack(blocks, dim=1)
            b_feature.append(blocks)
        return torch.stack(b_feature, dim=0)

    def merge_block(self, x):
        y = []
        for i in range(x.size(-1) - 1):
            y_ = torch.sum(x[:, :, i:i + 2], dim=-1)
            y.append(y_)
        for i in range(x.size(-1) - 2):
            y_ = torch.sum(x[:, :, i:i + 3], dim=-1)
            y.append(y_)
        return torch.stack(y, dim=2)

    def get_part_pool(self, x, pool='avg', no_overlap=True):
        result = []
        if pool == 'avg':
            pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
        elif pool == 'max':
            pooling = torch.nn.AdaptiveMaxPool2d((1, 1))
        H, W = x.size(2), x.size(3)
        c_h, c_w = int(H / 2), int(W / 2)
        per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        if per_h < 1 and per_w < 1:
            new_H, new_W = H + (self.block - c_h) * 2, W + (self.block - c_w) * 2
            x = nn.functional.interpolate(x, size=[new_H, new_W], mode='bilinear', align_corners=True)
            H, W = x.size(2), x.size(3)
            c_h, c_w = int(H / 2), int(W / 2)
            per_h, per_w = H / (2 * self.block), W / (2 * self.block)
        per_h, per_w = math.floor(per_h), math.floor(per_w)
        for i in range(self.block):
            i = i + 1
            if i < self.block:
                xmin = c_h - i * per_h
                xmax = c_h + i * per_h
                ymin = c_w - i * per_w
                ymax = c_w + i * per_w
                x_curr = x[:, :, xmin:xmax, ymin:ymax]
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    x_pad = F.pad(x_pre, (per_h, per_h, per_w, per_w), "constant", 0)
                    x_curr = x_curr - x_pad
                avgpool = pooling(x_curr)
                result.append(avgpool)
            else:
                if no_overlap and i > 1:
                    x_pre = x[:, :, (c_h - (i - 1) * per_h):(c_h + (i - 1) * per_h),
                            (c_w - (i - 1) * per_w):(c_w + (i - 1) * per_w)]
                    pad_h = c_h - (i - 1) * per_h
                    pad_w = c_w - (i - 1) * per_w
                    # x_pad = F.pad(x_pre,(pad_h,pad_h,pad_w,pad_w),"constant",0)
                    if x_pre.size(2) + 2 * pad_h == H:
                        x_pad = F.pad(x_pre, (pad_h, pad_h, pad_w, pad_w), "constant", 0)
                    else:
                        ep = H - (x_pre.size(2) + 2 * pad_h)
                        x_pad = F.pad(x_pre, (pad_h + ep, pad_h, pad_w + ep, pad_w), "constant", 0)
                    x = x - x_pad
                avgpool = pooling(x)
                result.append(avgpool)
        return torch.cat(result, dim=2)

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            # return torch.cat(y,dim=1)
            return torch.stack(y, dim=2)
        return y

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, view_num, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = "jx_vit_base_p16_224-80ecf9dd.pth"
        pretrain_choice = "imagenet"
        transformer_name = "vit_base_patch16_224_TransReID"
        self.cos_layer = False
        self.neck = 'bnneck'
        self.neck_feat = 'after'
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(transformer_name))
        # model_path = cfg.MODEL.PRETRAIN_PATH
        # pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        # self.cos_layer = cfg.MODEL.COS_LAYER
        # self.neck = cfg.MODEL.NECK
        # self.neck_feat = cfg.TEST.NECK_FEAT
        # self.in_planes = 768
        #
        # print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        camera_num = 0

        view_num = view_num

        self.transformer = factory[transformer_name](img_size=(256,256), sie_xishu=3.0,local_feature=True,
                                                        camera=camera_num, view=view_num, stride_size=[16, 16], drop_path_rate=0.1,
                                                        drop_rate= 0.0,
                                                        attn_drop_rate=0.0)
        if pretrain_choice == 'imagenet':
            self.transformer.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.transformer.blocks[-1]
        layer_norm = self.transformer.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes

        self.classifier = ClassBlock(768,num_classes,0.5)
        self.classifier_1 = ClassBlock(768,num_classes,0.5)
        self.classifier_2 = ClassBlock(768,num_classes,0.5)
        self.classifier_3 = ClassBlock(768,num_classes,0.5)
        self.classifier_4 = ClassBlock(768,num_classes,0.5)


        self.shuffle_groups = 2
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = 5
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = 4
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x):  # label is unused if self.cos_layer == 'no'

        features = self.transformer(x)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]

        if self.rearrange:
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.classifier(global_feat)

        local_feat_1_bn = self.classifier_1(local_feat_1)
        local_feat_2_bn = self.classifier_2(local_feat_2)
        local_feat_3_bn = self.classifier_3(local_feat_3)
        local_feat_4_bn = self.classifier_4(local_feat_4)

        if self.training:
            return [feat, local_feat_1_bn, local_feat_2_bn, local_feat_3_bn,local_feat_4_bn]  # global feature for triplet loss
        else:
            feats = torch.stack(
                [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=2)
            return feats

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_transformer_model(num_class,block = 4,return_f=False,deit=False):
    model = build_transformer(num_class, __factory_T_type,block=block,return_f=return_f,deit=deit)
    print('===========building transformer===========')
    return model
