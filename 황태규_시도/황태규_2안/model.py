# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import ResidualBlock, MSRInitializer

class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=1):
        super().__init__()
        self.conv = MSRInitializer(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.resblock1 = ResidualBlock(out_channels, int(3*out_channels), out_channels, cardinality=cardinality)
        self.resblock2 = ResidualBlock(out_channels, int(3*out_channels), out_channels, cardinality=cardinality)
    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim=128, base_channels=64, img_size=128, cardinality=1, out_channels=3):
        super().__init__()
        # latent z -> feature map (4x4)
        self.fc = nn.Linear(z_dim, base_channels*6*4*4)
        self.initial_shape = (base_channels*6, 4, 4)  # ex: 384ch, 4x4
        # 2 GBlock (8x8, 16x16, ... 등)
        self.gblocks = nn.ModuleList([
            GBlock(base_channels*6, base_channels*3, cardinality),
            GBlock(base_channels*3, base_channels, cardinality)
        ])
        self.to_rgb = MSRInitializer(nn.Conv2d(base_channels, out_channels, 1))
        self.tanh = nn.Tanh()
    def forward(self, z):
        out = self.fc(z).view(z.size(0), *self.initial_shape)
        for gblock in self.gblocks:
            out = gblock(out)
        out = self.to_rgb(out)
        return self.tanh(out)

class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cardinality=1):
        super().__init__()
        self.resblock1 = ResidualBlock(in_channels, int(3*in_channels), in_channels, cardinality=cardinality)
        self.resblock2 = ResidualBlock(in_channels, int(3*in_channels), in_channels, cardinality=cardinality)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        self.conv = MSRInitializer(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.downsample(x)
        x = self.conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, base_channels=64, img_size=128, cardinality=1, in_channels=3):
        super().__init__()
        self.from_rgb = MSRInitializer(nn.Conv2d(in_channels, base_channels, 1))
        # 2 DBlock (128→64→32→16 ...)
        self.dblocks = nn.ModuleList([
            DBlock(base_channels, base_channels*3, cardinality),
            DBlock(base_channels*3, base_channels*6, cardinality)
        ])
        self.final_conv = nn.Conv2d(base_channels*6, 1, 4)  # global score
    def forward(self, x):
        out = self.from_rgb(x)
        for dblock in self.dblocks:
            out = dblock(out)
        out = self.final_conv(out) # (B, 1, H', W')
        out = F.adaptive_avg_pool2d(out, 1)   # (B, 1, 1, 1)
        return out.view(out.size(0), 1)       # (B, 1)
