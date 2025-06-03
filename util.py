# util.py

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_c: int,
                 expension: int = 2, 
                 cardinality: int = 2):
        
        super().__init__()
        self.in_c = in_c
        self.cardinality = cardinality
        self.expanded_c = in_c * expension

        self.conv1 = nn.Conv2d(self.in_c, self.expanded_c, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(self.expanded_c, self.expanded_c, kernel_size=3, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.expanded_c, self.in_c, kernel_size=1, padding=0)

        self.bias1 = nn.Parameter(torch.zeros(self.expanded_c))
        self.bias2 = nn.Parameter(torch.zeros(self.expanded_c))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True) # slope 확인
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.leaky_relu1(y + self.bias1.view(1, -1, 1, 1)) #고민
        y = self.conv2(y)
        y = self.leaky_relu2(y + self.bias2.view(1, -1, 1, 1))
        y = self.conv3(y)
        return x + y
