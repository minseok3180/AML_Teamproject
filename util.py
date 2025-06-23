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

            std = ActivationGain / math.sqrt(fan_in)
            weight.normal_(0, std)
    return Layer


class ResidualBlock(nn.Module):
    """
    기본 ResidualBlock 구조: Conv1x1 → Conv3x3(Grouped) → Conv1x1
    LeakyReLU와 Bias, Dropout 적용 포함
    """
    def __init__(self, 
                 in_c: int,
                 expension: int = 2, 
                 cardinality: int = 2,
                 dropout_prob: float = 0.0):
        super().__init__()
        self.in_c = in_c
        self.cardinality = cardinality
        self.expanded_c = in_c * expension
        self.dropout_prob = dropout_prob

        self.conv1 = MSRInitializer(nn.Conv2d(self.in_c, self.expanded_c, kernel_size=1, padding=0))
        self.conv2 = MSRInitializer(nn.Conv2d(self.expanded_c, self.expanded_c, kernel_size=3, padding=1, groups=cardinality))
        self.conv3 = MSRInitializer(nn.Conv2d(self.expanded_c, self.in_c, kernel_size=1, padding=0))

        self.bias1 = nn.Parameter(torch.zeros(self.expanded_c))
        self.bias2 = nn.Parameter(torch.zeros(self.expanded_c))
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0.0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.leaky_relu1(y + self.bias1.view(1, -1, 1, 1))
        y = self.conv2(y)
        y = self.leaky_relu2(y + self.bias2.view(1, -1, 1, 1))
        if self.dropout:
            y = self.dropout(y)
        y = self.conv3(y)
        return x + y

class MinibatchDiscrimination(nn.Module):
    """
    Minibatch Discrimination 모듈:
    입력 feature 간의 유사도 차이를 측정하여 mode collapse를 방지
    """
    def __init__(self, in_features: int, out_features: int, kernel_dims: int = 50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        self.T = nn.Parameter(torch.Tensor(in_features, out_features * kernel_dims))
        nn.init.normal_(self.T, mean=0.0, std=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        B = x.size(0)

        # Project to feature-kernel space
        M = x @ self.T  # (B, out_features * kernel_dims)
        M = M.view(B, self.out_features, self.kernel_dims)

        # Compute L1 distance between all pairs in minibatch
        out = torch.zeros(B, self.out_features, device=x.device)
        for i in range(B):
            diffs = torch.abs(M[i].unsqueeze(0) - M)  # (B, out_features, kernel_dims)
            exp_sum = torch.sum(torch.exp(-diffs.sum(2)), dim=0) - 1  # exclude self
            out[i] = exp_sum
        return torch.cat([x, out], dim=1)

