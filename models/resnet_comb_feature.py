import torch
import torchvision
from torchvision import models
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        self.pretrained = models.resnet18(pretrained=True)
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x

class comb_resnet(nn.Module):
    def __init__(self):
        super(comb_resnet,self).__init__()
        self.l1 = new_model(output_layer = 'layer1').eval().cuda()
        self.l2 = new_model(output_layer = 'layer2').eval().cuda()
        self.l3 = new_model(output_layer = 'layer3').eval().cuda()
        self.l4 = new_model(output_layer = 'layer4').eval().cuda()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    
    def forward(self,img1):
        f1 = self.pool(self.l1(img1)).squeeze()
        f2 = self.pool(self.l2(img1)).squeeze()
        f3 = self.pool(self.l3(img1)).squeeze()
        f4 = self.pool(self.l4(img1)).squeeze()
        con = torch.cat((f1,f2,f3,f4),1)
        return con