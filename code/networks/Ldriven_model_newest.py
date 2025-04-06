import numpy as np
import os
import sys
import collections
from itertools import repeat
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from networks.vnet2 import VNet
from scipy.special import softmax

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x


class Ldriven_model(nn.Module):
    def __init__(self,n_channels=3,n_classes=2, n_filters=16, normalization='none', has_dropout=False):
        super().__init__()
        self.backbone = VNet(n_channels=1, n_classes=n_classes, normalization='batchnorm', has_dropout=True)
        self.precls_conv = nn.Sequential(
            nn.GroupNorm(16, 16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 8, kernel_size=1)
        )
        self.GAP = nn.Sequential(
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(256, 256, kernel_size=1, stride=1, padding=0)
        )
        self.adapter = Adapter(512, 4)
        self.controller1 = nn.Conv3d(256 + 256, 256, kernel_size=1, stride=1, padding=0)
        self.controller2 = nn.Conv3d(256, 128, kernel_size=1, stride=1, padding=0)
        self.controller3 = nn.Conv3d(128, 64, kernel_size=1, stride=1, padding=0)
        self.controller4 = nn.Conv3d(64, 8, kernel_size=1, stride=1, padding=0)

        self.register_buffer('organ_embedding', torch.randn(n_classes, 512))
        self.text_to_vision = nn.Linear(512, 256)
        self.class_num = n_classes
        self.temp_conv = nn.Conv3d(16,8,kernel_size=1,padding=0)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.last_out_conv = nn.Conv3d(8,2,1,padding=0)

    def heads_forward(self, features, weights, biases):
        assert features.dim() == 5
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv3d(
                x, w, bias=b,
                stride=1, padding=0)
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x_in):
        dec4, out = self.backbone(x_in)
        task_encoding = F.relu(self.text_to_vision(self.adapter(self.organ_embedding)))
        task_encoding = task_encoding.unsqueeze(2).unsqueeze(2).unsqueeze(2)  # 256
        # task_encoding torch.Size([31, 256, 1, 1, 1])
        x_feat = self.GAP(dec4) # 256
        b = x_feat.shape[0]
        logits_array = []
        for i in range(b):
            temp = x_feat[i].unsqueeze(0)
            x_cond = torch.cat([x_feat[i].unsqueeze(0),task_encoding],1)
            #x_cond = torch.cat([x_feat[i].unsqueeze(0), x_feat[i].unsqueeze(0)], 1)
            params = self.controller1(x_cond)
            params = self.controller2(params)
            params = self.controller3(params)
            params = self.controller4(params)
            # params.squeeze_(-1).squeeze_(-1).squeeze_(-1)
            temp2 = out[i].unsqueeze(0)
            #head_inputs = out[i].unsqueeze(0)
            head_inputs = self.precls_conv(out[i].unsqueeze(0))
            params = params.expand_as(head_inputs)
            head_inputs = head_inputs + params

            logits = self.last_out_conv(head_inputs)
            logits_array.append(logits)
        out = torch.cat(logits_array,dim=0)
        return out

if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from ptflops import get_model_complexity_info
    # model = VNet(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=False)
    model = Ldriven_model(n_channels=1, n_classes=2, normalization='batchnorm', has_dropout=True)
    word_embedding = torch.load(".././pretrained_weights/txt_encoding.pth")
    model.organ_embedding.data = word_embedding.float()
    model = model.cuda()

    with torch.cuda.device(5):
      macs, params = get_model_complexity_info(model, (1, 112, 112, 80), as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
