from __future__ import print_function
from __future__ import division
import torch.nn as nn
from torchvision import models
from FG_baseline import utils
import numpy as np
import pdb
import torch
import torch.nn.functional as F


def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        # model_ft.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # utils.set_parameter_requires_grad(model_ft.parameters(), requires_grad=False)
        num_ftrs = model_ft.fc.in_features
        print(num_ftrs)
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        utils.set_parameter_requires_grad(model_ft.parameters(), requires_grad=False)
        input_size = 448

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        res50 = models.resnet50(pretrained=True)
        #res50.fc = nn.Linear(res50.fc.in_features, 200)
        res50.fc = nn.Linear(3072, 200)
        self.conv4 = nn.Sequential(*list(res50.children()))[:-3]
        self.conv5 = nn.Sequential(*list(res50.children()))[-3:-1]
        self.fc_2 = nn.Sequential(*list(res50.children()))[-1:]

        self.maxpool = nn.MaxPool2d(kernel_size=14)
        self.fc_1 = nn.Linear(1024, 200)
        # self.conv6 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        # self.relu = nn.ReLU()
        #self.upsample = nn.UpsamplingBilinear2d(size=(14, 14))
        self.conv_1x1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1)
        #self.upsample = F.upsample(self, (14, 14), mode='bilinear')

        # print(self.fc_2)
        # self.fc = nn.Linear(res50.fc.in_features, 200)

    def forward(self, x):
        # pdb.set_trace()       #设置断点
        output4 = self.conv4(x)
        output4_1 = output4.detach()
        # output4_1 = self.conv6(output4_1)
        # output4_1 = self.relu(output4_1)
        #pool_1 = self.maxpool(output4_1)
        # print(pool_1.shape)
        # pool_1 = pool_1.view(1, -1)
        #fea_1 = self.fc_1(pool_1)

        output5 = self.conv5(output4)
        output5_1 = output5.detach()
        output5_1 = F.interpolate(output5_1, (14, 14), mode='bilinear')
        #output5_1 = self.upsample(output5_1)
        output5_1 = self.conv_1x1(output5_1)


        #output7 = torch.cat([])

        output6 = torch.mul(output4_1, output5_1)
        output6 = self.maxpool(output6)
        output6 = output6.view(1, -1)
        fea_1 = self.fc_1(output6)
        fea_2 = self.fc_2(output5)
        # fea = torch.cat([pool_1, output5], dim=1)
        # fea = fea.view(1, -1)
        # fea = self.fc_2(fea)
        # print(fea.shape)

        #return fea_1, fea_2
        return fea_1, fea_2


if __name__ == '__main__':
    indata = torch.randn((1, 3, 224, 224))
    net = Model()

    out = net(indata)
