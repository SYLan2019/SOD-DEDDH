import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channel=0, out_channel=0, kernel_size=3, padding=0, stride=1, relu=True, bn=True):
        super(Conv, self).__init__()
        self.use_relu = relu
        self.use_bn = bn
        assert in_channel > 0 and out_channel > 0 and kernel_size > 0
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.conv1(x)
        if self.use_bn:
            out = self.bn(out)
        if self.use_relu:
            out = self.relu(out)
        return out