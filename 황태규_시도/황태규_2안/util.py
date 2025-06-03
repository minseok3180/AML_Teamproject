# util.py
import torch
import torch.nn as nn
import math

class BiasAct(nn.Module):
    def __init__(self, channels, act=nn.LeakyReLU(0.2)):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.act = act
    def forward(self, x):
        return self.act(x + self.bias)

def MSRInitializer(layer, activation_gain=1):
    fan_in = layer.weight.data.size(1) * layer.weight.data[0][0].numel()
    std = activation_gain / math.sqrt(fan_in) if fan_in > 0 else 1.0
    if std == 0:
        layer.weight.data.zero_()
    else:
        layer.weight.data.normal_(0, std)
    return layer


class GroupedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, cardinality, bias=False):
        super().__init__()
        self.conv = MSRInitializer(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding=kernel_size//2, groups=cardinality, bias=bias)
        )
    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, cardinality=1):
        super().__init__()
        # 1x1 Conv (확장)
        self.conv1 = MSRInitializer(nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False))
        self.bias1 = BiasAct(mid_channels)
        # Grouped 3x3 Conv (카디널리티 지원)
        self.conv2 = GroupedConv2d(mid_channels, mid_channels, kernel_size=3, cardinality=cardinality, bias=False)
        self.bias2 = BiasAct(mid_channels)
        # 1x1 Conv (복원)
        self.conv3 = MSRInitializer(nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False), activation_gain=0)
    def forward(self, x):
        y = self.conv1(x)
        y = self.bias1(y)
        y = self.conv2(y)
        y = self.bias2(y)
        y = self.conv3(y)
        return x + y
