import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
import math

class CNN(nn.Module):
    def __init__(self,num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        self.conv1=nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.relu1=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2=nn.BatchNorm2d(32)
        self.relu2=nn.ReLU(inplace=True)
        self.pool2=nn.AvgPool2d(2,2)

        self.conv3=nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3=nn.BatchNorm2d(64)
        self.relu3=nn.ReLU(inplace=True)
        self.pool3=nn.AvgPool2d(2,2)

        self.conv4=nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4=nn.BatchNorm2d(64)
        self.relu4=nn.ReLU(inplace=True)
        self.pool4=nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight,mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu2(x)
        x=self.pool2(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x=self.relu3(x)
        x=self.pool3(x)
        x=self.conv4(x)
        x=self.bn4(x)
        x=self.relu4(x)
        x=self.pool4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)