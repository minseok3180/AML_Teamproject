# model_mhsa.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import ResidualBlock
from mhsa import MultiHeadSelfAttention


class GBlock(nn.Module):
    """간단화된 ResidualBlock 기반 Upsampling 블록"""
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2):
        super().__init__()
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)
        self.conv1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=True)
        self.res_block1 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class DBlock(nn.Module):
    """간단화된 ResidualBlock 기반 Downsampling 블록"""
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2):
        super().__init__()
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)
        self.res_block1 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        self.conv1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        x = self.conv1x1(x)
        return x


class Generator(nn.Module):
    """
    MHSA를 중간 해상도(≈32×32)에서 적용하도록 수정된 Generator
    """
    def __init__(
        self,
        NoiseDim: int = 100,
        BaseChannels: list = [256, 128, 64, 32, 16, 8, 4],
        cardinality: int = 2,
        expension: int = 2,
        mhsa_heads: int = 2
    ):
        super().__init__()
        self.noise_dim = NoiseDim
        self.BaseChannels = BaseChannels

        # 4×4로 펼치는 FC + 초기 Conv
        C0 = BaseChannels[0]
        self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4)
        self.conv4x4 = nn.Conv2d(int(1.5*C0), int(1.5*C0), kernel_size=3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Upsampling GBlocks
        self.gblocks = nn.ModuleList()
        for Cin, Cout in zip(BaseChannels[:-1], BaseChannels[1:]):
            self.gblocks.append(GBlock(Cin, Cout, cardinality, expension))

        # 중간 해상도에 MHSA 삽입: gblocks[2] 이후 (4→8→16→32 해상도)
        self.mid_idx = 2  
        mid_C = BaseChannels[self.mid_idx + 1]           # gblocks[2] 출력 채널 기준
        mhsa_in = int(1.5 * mid_C)
        assert mhsa_in % mhsa_heads == 0, "MHSA head 수가 채널 수를 나눌 수 있어야 합니다"
        self.mhsa = MultiHeadSelfAttention(in_channels=mhsa_in, num_heads=mhsa_heads)

        # 최종 RGB 투영
        C_last = BaseChannels[-1]
        out_ch = int(1.5 * C_last)
        self.to_rgb = nn.Conv2d(out_ch, 3, kernel_size=1, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        C0 = self.BaseChannels[0]

        x = self.fc(z).view(B, int(1.5*C0), 4, 4)
        x = self.lrelu(self.conv4x4(x))

        for idx, gblock in enumerate(self.gblocks):
            x = gblock(x)
            # 중간 블록 통과 후 MHSA 적용
            if idx == self.mid_idx:
                x = self.mhsa(x)

        x = self.to_rgb(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    """
    MHSA를 중간 해상도(≈32×32)에서 적용하도록 수정된 Discriminator
    """
    def __init__(
        self,
        BaseChannels: list = [4, 8, 16, 32, 64, 128, 256],
        cardinality: int = 2,
        expension: int = 2,
        mhsa_heads: int = 2
    ):
        super().__init__()
        self.BaseChannels = BaseChannels
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # RGB → feature map
        C0 = BaseChannels[0]
        self.from_rgb = nn.Conv2d(3, int(1.5*C0), kernel_size=1, bias=True)

        # Downsampling DBlocks
        self.dblocks = nn.ModuleList()
        for Cin, Cout in zip(BaseChannels[:-1], BaseChannels[1:]):
            self.dblocks.append(DBlock(Cin, Cout, cardinality, expension))

        # 중간 해상도에 MHSA 삽입: dblocks[2] 이후 (256→128→64→32 해상도)
        self.mid_idx = 2
        mid_C = BaseChannels[self.mid_idx + 1]
        mhsa_in = int(1.5 * mid_C)
        assert mhsa_in % mhsa_heads == 0, "MHSA head 수가 채널 수를 나눌 수 있어야 합니다"
        self.mhsa = MultiHeadSelfAttention(in_channels=mhsa_in, num_heads=mhsa_heads)

        # 분류 헤드
        final_ch = int(1.5 * BaseChannels[-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_ch * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lrelu(self.from_rgb(x))

        for idx, dblock in enumerate(self.dblocks):
            h = dblock(h)
            # 중간 블록 통과 후 MHSA 적용
            if idx == self.mid_idx:
                h = self.mhsa(h)

        h = self.flatten(h)
        logit = self.fc(h)
        return logit.view(-1)
