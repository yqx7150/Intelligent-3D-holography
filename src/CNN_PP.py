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



class CNN_PP(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,num_filters_per_layer,filter_size, img_size=1024):
        super(CNN_PP, self).__init__()
        
        self.conv1 = Conv1(in_planes=input_dim,out_planes=num_filters_per_layer)
        self.conv2 = Conv1(in_planes=num_filters_per_layer,out_planes=input_dim)
        self.conv3 = Conv2(in_planes=input_dim,out_planes=output_dim)

        self.layer = self.make_layer(num_filters_per_layer,num_filters_per_layer,filter_size,num_layers)
        self.tanH = torch.nn.Hardtanh(-math.pi, math.pi)
        self.sig = torch.nn.Sigmoid()
        self.img_size = img_size



    def make_layer(self,in_places,out_places,filter_size,num_layers,stride=1):
        layers = []
        for i in range(1,num_layers):
            layers.append(Bottleneck(in_places,in_places,filter_size,stride))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        amp1 = x[:,0,0:self.img_size:2,0:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        amp2 = x[:,0,1:self.img_size:2,0:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        amp3 = x[:,0,0:self.img_size:2,1:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        amp4 = x[:,0,1:self.img_size:2,1:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)

        phase1 = x[:,1,0:self.img_size:2,0:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        phase2 = x[:,1,1:self.img_size:2,0:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        phase3 = x[:,1,0:self.img_size:2,1:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)
        phase4 = x[:,1,1:self.img_size:2,1:self.img_size:2].reshape(1,1,self.img_size//2,self.img_size//2)

        slm_field = torch.cat([amp1, amp2, amp3, amp4, phase1, phase2, phase3, phase4], dim=-3)
        
        l1 = self.conv1(slm_field)
        l2 = self.layer(l1)
        l3 = self.conv2(l2)
        l4 = l3 + slm_field
        l5= self.conv3(l4)
        out = self.tanH(l5)

        holo = torch.zeros(1, 1, self.img_size, self.img_size, device='cuda')
        holo[0,0,0:self.img_size:2,0:self.img_size:2] = out[0,0,:,:]
        holo[0,0,1:self.img_size:2,0:self.img_size:2] = out[0,1,:,:]
        holo[0,0,0:self.img_size:2,1:self.img_size:2] = out[0,2,:,:]
        holo[0,0,1:self.img_size:2,1:self.img_size:2] = out[0,3,:,:]

        return holo