import torch
import torch.nn as nn
from .utils import ClassBlock


class FSRA(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_classes = opt.nclasses
        droprate = opt.droprate
        in_planes = opt.in_planes
        self.class_name = "classifier_heat"
        self.block = opt.block
        # global classifier
        self.classifier1 = ClassBlock(in_planes, num_classes, droprate)
        # local classifier
        for i in range(self.block):
            name = self.class_name + str(i+1)
            setattr(self, name, ClassBlock(in_planes, num_classes, droprate))

    def forward(self, features):
        global_cls, global_feature = self.classifier1(features[:, 0])
        # tranformer_feature = torch.mean(features,dim=1)
        # tranformer_feature = self.classifier1(tranformer_feature)
        if self.block == 1:
            return global_cls, global_feature

        part_features = features[:, 1:]

        heat_result = self.get_heartmap_pool(part_features)
        cls_list, features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)

        total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls, total_features]

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            cls_list.append(res[0])
            features_list.append(res[1])
        return cls_list, features_list


class FSRA_CNN(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()

        self.opt = opt
        num_classes = opt.nclasses
        droprate = opt.droprate
        in_planes = opt.in_planes
        self.class_name = "classifier_heat"
        self.block = opt.block
        # global classifier
        self.classifier1 = ClassBlock(in_planes, num_classes, droprate)
        # local classifier
        for i in range(self.block):
            name = self.class_name + str(i+1)
            setattr(self, name, ClassBlock(in_planes, num_classes, droprate))

    def forward(self, features):
        # global_cls, global_feature = self.classifier1(features[:, 0])
        features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1,2)
        global_feature = torch.mean(features,dim=1)
        global_cls, global_feature = self.classifier1(global_feature)
        if self.block == 1:
            return global_cls, global_feature

        part_features = features
        # print(part_features.shape)


        heat_result = self.get_heartmap_pool(part_features)
        cls_list, features_list = self.part_classifier(
            self.block, heat_result, cls_name=self.class_name)

        total_cls = [global_cls] + cls_list
        total_features = [global_feature] + features_list
        if not self.training:
            total_features = torch.stack(total_features,dim=-1)
        return [total_cls, total_features]

    def get_heartmap_pool(self, part_features, add_global=False, otherbranch=False):
        heatmap = torch.mean(part_features, dim=-1)
        size = part_features.size(1)
        arg = torch.argsort(heatmap, dim=1, descending=True)
        x_sort = [part_features[i, arg[i], :]
                  for i in range(part_features.size(0))]
        x_sort = torch.stack(x_sort, dim=0)

        split_each = size / self.block
        split_list = [int(split_each) for i in range(self.block - 1)]
        split_list.append(size - sum(split_list))
        split_x = x_sort.split(split_list, dim=1)

        split_list = [torch.mean(split, dim=1) for split in split_x]
        part_featuers_ = torch.stack(split_list, dim=2)
        if add_global:
            global_feat = torch.mean(part_features, dim=1).view(
                part_features.size(0), -1, 1).expand(-1, -1, self.block)
            part_featuers_ = part_featuers_ + global_feat
        if otherbranch:
            otherbranch_ = torch.mean(
                torch.stack(split_list[1:], dim=2), dim=-1)
            return part_featuers_, otherbranch_
        return part_featuers_

    def part_classifier(self, block, x, cls_name='classifier_lpn'):
        part = {}
        cls_list, features_list = [], []
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            # part[i] = torch.squeeze(x[:,:,i])
            name = cls_name + str(i+1)
            c = getattr(self, name)
            res = c(part[i])
            cls_list.append(res[0])
            features_list.append(res[1])
        return cls_list, features_list