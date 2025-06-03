# util.py

import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Residual Block 모듈
    Args:
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
    Forward Input:
        x (torch.Tensor): 입력 텐서 (B, C, H, W)
    Output:
        out (torch.Tensor): 출력 텐서 (B, C, H, W)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 입력/출력 채널이 다르면 projection shortcut 사용 (1x1 conv)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
