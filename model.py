# model.py

import torch
import torch.nn as nn
from util import ResidualBlock

class Generator(nn.Module):
    """
    Generator 네트워크
    Forward Input:
        x (torch.Tensor): 입력 이미지 (B, C_in, H, W)
    Output:
        out (torch.Tensor): 생성 이미지 (B, C_out, H, W)
    """
    def __init__(self):
        super().__init__()
        # TODO: Generator 아키텍처 구현
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Generator forward 구현
        pass

class Discriminator(nn.Module):
    """
    Discriminator 네트워크
    Forward Input:
        x (torch.Tensor): 이미지 (B, C, H, W)
    Output:
        out (torch.Tensor): 판별 결과 (B, 1)
    """
    def __init__(self):
        super().__init__()
        # TODO: Discriminator 아키텍처 구현
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Discriminator forward 구현
        pass

