# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import ResidualBlock  

class GBlock(nn.Module):
    """
    Input channels: 1.5 * C_in
    (1) Conv1x1: (1.5*C_in) → (1.5*C_out)
    (2) Bilinear Upsample *2
    (3) 2* ResidualBlock
    """
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2):
        super().__init__()
        # Block channels
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)

        # 1×1 conv to change channels
        self.conv1x1 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # Two simple ResidualBlocks
        self.res_block1 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Change channels
        x = self.conv1x1(x)
        # Upsample spatially
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        # Apply ResidualBlocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class DBlock(nn.Module):
    """
    Input channels: 1.5 * C_in
    (1) 2* ResidualBlock
    (2) Bilinear Downsample *0.5
    (3) Conv1x1: (1.5*C_in) → (1.5*C_out)
    """
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2):
        super().__init__()
        # Block channels
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)

        # Two simple ResidualBlocks
        self.res_block1 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        # 1×1 conv to change channels after downsampling
        self.conv1x1 = nn.Conv2d(
            in_channels=self.in_ch,
            out_channels=self.out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply ResidualBlocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        # Downsample spatially
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        # Change channels
        x = self.conv1x1(x)
        return x


class Generator(nn.Module):
    """
    R3GAN-style Generator with simple ResidualBlocks
    """
    def __init__(
        self,
        NoiseDim: int = 100,
        BaseChannels: list = [],
        cardinality: int = 2,
        expension: int = 2
    ):
        super().__init__()
        self.noise_dim = NoiseDim
        self.BaseChannels = BaseChannels

        # FC to initial feature map
        C0 = BaseChannels[0]
        self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4)
        # Initial conv + activation
        self.conv4x4 = nn.Conv2d(
            in_channels=int(1.5 * C0),
            out_channels=int(1.5 * C0),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # Progressive GBlocks
        self.gblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin = BaseChannels[i]
            Cout = BaseChannels[i + 1]
            self.gblocks.append(
                GBlock(Cin, Cout, cardinality=cardinality, expension=expension)
            )

        # To RGB
        C_last = BaseChannels[-1]
        self.to_rgb = nn.Conv2d(
            in_channels=int(1.5 * C_last),
            out_channels=3,
            kernel_size=1,
            bias=True
        )
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        C0 = self.BaseChannels[0]
        x = self.fc(z).view(B, int(1.5 * C0), 4, 4)
        x = self.lrelu(self.conv4x4(x))
        for gblock in self.gblocks:
            x = gblock(x)
        x = self.to_rgb(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    """
    R3GAN-style Discriminator with simple ResidualBlocks
    """
    def __init__(
        self,
        BaseChannels: list = [],
        cardinality: int = 2,
        expension: int = 2
    ):
        super().__init__()
        self.BaseChannels = BaseChannels
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        # From RGB
        C0 = BaseChannels[0]
        self.from_rgb = nn.Conv2d(
            in_channels=3,
            out_channels=int(1.5 * C0),
            kernel_size=1,
            bias=True
        )

        # Progressive DBlocks
        self.dblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin = BaseChannels[i]
            Cout = BaseChannels[i + 1]
            self.dblocks.append(
                DBlock(Cin, Cout, cardinality=cardinality, expension=expension)
            )

        # Final classifier
        final_ch = int(1.5 * BaseChannels[-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_ch * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lrelu(self.from_rgb(x))
        for dblock in self.dblocks:
            h = dblock(h)
        h = self.flatten(h)
        logit = self.fc(h)
        return logit.view(h.shape[0])


