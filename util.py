# util.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def MSRInitializer(Layer: nn.Module, ActivationGain: float = 1.0) -> nn.Module:
    with torch.no_grad():
        weight = Layer.weight.data
        fan_in = weight.size(1) * weight[0][0].numel()  
        if ActivationGain == 0 or fan_in == 0: 
            weight.zero_()
        else:
            # Given input channel, set initial parameter
            std = ActivationGain / math.sqrt(fan_in)
            weight.normal_(0, std)
    return Layer


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_c: int,
                 expension: int = 2, 
                 cardinality: int = 2):
        
        super().__init__()
        self.in_c = in_c
        self.cardinality = cardinality
        self.expanded_c = in_c * expension

        self.conv1 = MSRInitializer(nn.Conv2d(self.in_c, self.expanded_c, kernel_size=1, padding=0))
        self.conv2 = MSRInitializer(nn.Conv2d(self.expanded_c, self.expanded_c, kernel_size=3, padding=1, groups=cardinality))
        self.conv3 = MSRInitializer(nn.Conv2d(self.expanded_c, self.in_c, kernel_size=1, padding=0))

        self.bias1 = nn.Parameter(torch.zeros(self.expanded_c))
        self.bias2 = nn.Parameter(torch.zeros(self.expanded_c))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.leaky_relu1(y + self.bias1.view(1, -1, 1, 1))
        y = self.conv2(y)
        y = self.leaky_relu2(y + self.bias2.view(1, -1, 1, 1))
        y = self.conv3(y)
        return x + y
