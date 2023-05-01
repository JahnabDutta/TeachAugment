import math

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary


## convolutional network for CFAR-10 dataset

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2), # 32x32x3 -> 32x32x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32x16 -> 16x16x16
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2), # 16x16x16 -> 16x16x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer4 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16x32 -> 8x8x32
        self.fc = nn.Linear(8*8*32, num_classes) # 8x8x32 -> 10

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
        



