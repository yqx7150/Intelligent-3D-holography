import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch
import math
from torchsummary import summary


def Conv1(in_planes,out_planes,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=3,stride=stride,padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )

def Conv2(in_planes,out_planes,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=3,stride=stride,padding=1),
        nn.BatchNorm2d(out_planes),
    )


class Bottleneck(nn.Module):
    def __init__(self,in_places,out_places,filter_size=3,stride=1):
        super(Bottleneck,self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_places,out_places,kernel_size=filter_size,stride=stride,padding=1),
            nn.BatchNorm2d(out_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_places,out_places,kernel_size=filter_size,stride=stride,padding=1),
            nn.BatchNorm2d(out_places),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = x
        out = self.bottleneck(x)

        out = out + residual
        out = self.relu(out)
        return out



class CNN(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,num_filters_per_layer,filter_size):
        super(CNN, self).__init__()
        
        self.conv1 = Conv1(in_planes=input_dim,out_planes=num_filters_per_layer)
        self.conv2 = Conv1(in_planes=num_filters_per_layer,out_planes=input_dim)
        self.conv3 = Conv2(in_planes=input_dim,out_planes=output_dim)

        self.layer = self.make_layer(num_filters_per_layer,num_filters_per_layer,filter_size,num_layers)
        self.tanH = torch.nn.Hardtanh(-math.pi, math.pi)
        self.sig = torch.nn.Sigmoid()


    def make_layer(self,in_places,out_places,filter_size,num_layers,stride=1):
        layers = []
        for i in range(1,num_layers):
            layers.append(Bottleneck(in_places,in_places,filter_size,stride))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.layer(l1)
        l3 = self.conv2(l2)
        l4 = l3 + x
        l5= self.conv3(l4)
        out = self.tanH(l5)
        # out = self.sig(l5)
        
        return out