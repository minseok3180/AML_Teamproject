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
    def __init__(self, in_channels=3, out_channels=3, num_blocks=5, base_channels=64):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=7, padding=3)
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels, base_channels) for _ in range(num_blocks)]
        )

        # 여기선 upsampling이 필요없다고 가정 (SR이나 z→이미지 생성이면 upsample 추가)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.input_conv(x)
        out = self.res_blocks(out)
        out = self.output_conv(out)
        out = self.tanh(out)  # [-1, 1]로 스케일 (GAN 표준)
        return out

class Discriminator(nn.Module):
    """
    Discriminator 네트워크
    Forward Input:
        x (torch.Tensor): 이미지 (B, C, H, W)
    Output:
        out (torch.Tensor): 판별 결과 (B, 1)
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        # Conv → Down → ResidualBlock (간단화)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels, base_channels * 2),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(base_channels * 4, base_channels * 4),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=2, padding=1),  # 16x16
        )
        # 출력 (B, 1, 16, 16): 각 패치별로 진짜/가짜 score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
