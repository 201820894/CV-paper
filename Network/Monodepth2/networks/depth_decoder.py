import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch
from layers import *

class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()
        
        self.num_output_channels= num_output_channels
        self.use_skips= use_skips
        # Interpolation
        self.unsample_mode= 'nearest'
        self.scales=scales
        
        # [64, 64, 128, 256, 512]
        self.num_ch_enc=num_ch_enc
        self.num_ch_dec=np.array([16, 32, 64, 128, 256])
        
        self.convs=OrderedDict()
        for i in range(4,-1, -1):
            # upconv0
            # 역순으로 가는 정상적 convolution
            num_ch_in=self.num_ch_enc[-1] if i==4 else self.num_ch_dec[i+1]
            num_ch_out=self.num_ch_dec[i]
            self.convs[("upconv", i ,0)]=ConvBlock(num_ch_in, num_ch_out)
            
            # upconv1
            # skip connection 적용시, 이전꺼 더하고 conv
            # 미적용시 그냥 conv
            num_ch_in=self.num_ch_dec[i]
            if self.use_skips and i>0:
                num_ch_in+=self.num_ch_enc[i-1]
            num_ch_out=self.num_ch_dec[i]
            self.convs[("upconv", i, 1)]=ConvBlock(num_ch_in, num_ch_out)
            
        # 0, 1, 2, 3이 scale
        for s in self.scales: 
            self.convs[("dispconv", s)]=Conv3x3(self.num_ch_dec[s], self.num_output_channels)
            
        self.decoder=nn.ModuleList(list(self.convs.values()))
        self.sigmoid=nn.Sigmoid()
    
    # 여기서 input_features 는 resnet encoder의 output: 각 layer 통과한 output
    # torch.tensor
    def forward(self, input_features):
        self.outputs={}
        #512 or 2048
        x=input_features[-1]
        
        #256으로
        for i in range(4, -1, -1):
            # 여기서 왜 줄이고 늘리는거?: upconv로 줄이고, upsample(scale 2)로 늘리는거 왜
            x=self.convs[("upconv", i, 0)](x)
            x=[upsample(x)]
            
            if self.use_skips and i>0:
                x+=[input_features[i-1]]
            
            # 0는 batch size
            x=torch.cat(x,1)
            x=self.convs[("upconv", i, 1)](x)

            # self.scales: -1, 0, 1, 2, 3
            # 마지막 부분 
            if i in self.scales:
                self.outputs[("disp", i)]=self.sigmoid(self.convs[("dispconv", i)](x))
                
        return self.outputs