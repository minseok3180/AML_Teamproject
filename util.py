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
        # TODO: Residual block 내부 모듈 구현
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: forward 구현
        pass
