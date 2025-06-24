import torch
import torch.nn as nn
import torch.nn.functional as F
from network import ResidualBlock, MinibatchDiscrimination


class GBlock(nn.Module):
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2, dropout_rate: float = 0.2):
        super().__init__()
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)

        self.conv1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.res_block1 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.out_ch, expension=expension, cardinality=cardinality)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1x1(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        x = self.res_block1(x)
        x = self.dropout(x)
        x = self.res_block2(x)
        return x


class DBlock(nn.Module):
    def __init__(self, C_in: int, C_out: int, cardinality: int = 2, expension: int = 2):
        super().__init__()
        self.in_ch = int(1.5 * C_in)
        self.out_ch = int(1.5 * C_out)

        self.res_block1 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        self.res_block2 = ResidualBlock(self.in_ch, expension=expension, cardinality=cardinality)
        self.conv1x1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True)
        x = self.conv1x1(x)
        return x


class Generator(nn.Module):
    def __init__(self, NoiseDim: int = 100,
                 BaseChannels: list = [256, 128, 64, 32, 16, 8, 4],
                 cardinality: int = 2, expension: int = 2,
                 dropout_rate: float = 0.2):

        super().__init__()
        self.noise_dim = 100
        self.BaseChannels = BaseChannels

        C0 = BaseChannels[0]
        self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4)
        self.conv4x4 = nn.Conv2d(int(1.5 * C0), int(1.5 * C0), kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        self.gblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin, Cout = BaseChannels[i], BaseChannels[i + 1]
            self.gblocks.append(GBlock(Cin, Cout, cardinality=cardinality, expension=expension, dropout_rate=dropout_rate))

        C_last = BaseChannels[-1]
        self.to_rgb = nn.Conv2d(int(1.5 * C_last), 3, kernel_size=1, bias=True)
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
    def __init__(self, BaseChannels: list = [4, 8, 16, 32, 64, 128, 256],
                 cardinality: int = 2, expension: int = 2):

        super().__init__()
        self.BaseChannels = BaseChannels
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        C0 = BaseChannels[0]
        self.from_rgb = nn.Conv2d(3, int(1.5 * C0), kernel_size=1, bias=True)

        self.dblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin, Cout = BaseChannels[i], BaseChannels[i + 1]
            self.dblocks.append(DBlock(Cin, Cout, cardinality=cardinality, expension=expension))

        final_ch = int(1.5 * BaseChannels[-1])
        self.flatten = nn.Flatten()

        # Minibatch Discrimination before the final layer
        self.mb_disc = MinibatchDiscrimination(in_features=final_ch * 4 * 4, out_features=100, kernel_dims=5)
        self.fc = nn.Linear(final_ch * 4 * 4 + 100, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            x = x + torch.randn_like(x) * 0.1  # Instance noise 삽입
        h = self.lrelu(self.from_rgb(x))
        for dblock in self.dblocks:
            h = dblock(h)
        h = self.flatten(h)
        h = self.mb_disc(h)
        logit = self.fc(h)
        return logit.view(h.shape[0])

