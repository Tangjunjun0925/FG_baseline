from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models
from FG_baseline import utils
import numpy as np
import pdb
import torch
import torch.nn.functional as F
import FG_Encoding.encoding as encoding
import FG_Encoding.encoding.models.resnet as resnet
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        res50 = models.resnet50(pretrained=True)
        res50.fc = nn.Linear(res50.fc.in_features, 200)
        #res50.fc = nn.Linear(3072, 200)
        self.conv4 = nn.Sequential(*list(res50.children()))[:-3]
        self.conv5 = nn.Sequential(*list(res50.children()))[-3:-2]

        n_codes = 32
        self.head = nn.Sequential(
            nn.Conv2d(2048, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            encoding.nn.Encoding(D=128, K=n_codes),
            encoding.nn.View(-1, 128 * n_codes),
            encoding.nn.Normalize(),
            nn.Linear(128 * n_codes, 200),
        )
        #self.upsample = F.upsample(self, (14, 14), mode='bilinear')

        # print(self.fc_2)
        # self.fc = nn.Linear(res50.fc.in_features, 200)

    def forward(self, x):
        # pdb.set_trace()       #设置断点
        output4 = self.conv4(x)

        output5 = self.conv5(output4)

        fea = self.head(output5)

        return fea

#
# if __name__ == '__main__':
#     indata = torch.randn((1, 3, 224, 224))
#     net = Model()
#
#     out = net(indata)
