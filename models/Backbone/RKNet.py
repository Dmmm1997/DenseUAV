from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models

class USAM(nn.Module):
    def __init__(self, kernel_size=3, padding=1, polish=False):
        super(USAM, self).__init__()

        kernel = torch.ones((kernel_size, kernel_size))
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        

        kernel2 = torch.ones((1, 1)) * (kernel_size * kernel_size)
        kernel2 = kernel2.unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

        self.polish = polish
        self.pad = padding
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(1)

    def __call__(self, x):
        fmap = x.sum(1, keepdim=True)      
        x1 = F.conv2d(fmap, self.weight, padding=self.pad)
        x2 = F.conv2d(fmap, self.weight2, padding=0) 
        
        att = x2 - x1
        att = self.bn(att)
        att = self.relu(att)

        if self.polish:
            att[:, :, :, 0] = 0
            att[:, :, :, -1] = 0
            att[:, :, 0, :] = 0
            att[:, :, -1, :] = 0

        output = x + att * x

        return output



class RKNet(nn.Module):
    def __init__(self, stride=2, init_model=None, pool='avg'):
        super(RKNet, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)

        self.pool = pool
        if pool =='avg+max':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
            #self.classifier = ClassBlock(4096, class_num, droprate)
        elif pool=='avg':
            model_ft.avgpool2 = nn.AdaptiveAvgPool2d((1,1))
            #self.classifier = ClassBlock(2048, class_num, droprate)
        elif pool=='max':
            model_ft.maxpool2 = nn.AdaptiveMaxPool2d((1,1))
        elif pool=='gem':
            model_ft.gem2 = GeM(dim=2048)

        self.model = model_ft

        if init_model!=None:
            self.model = init_model.model
            self.pool = init_model.pool
            #self.classifier.add_block = init_model.classifier.add_block

        self.usam_1 = USAM()
        self.usam_2 = USAM()

    def forward_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.usam_1(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.usam_2(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        return x