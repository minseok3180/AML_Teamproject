# model.py

import torch
import torch.nn as nn
from util import ResidualBlock

class Generator(nn.Module):
    """
    GAN Generator for 256×256 이미지 생성.
    """
    def __init__(self, latent_dim: int = 100, base_channels: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels

        # 1) latent_dim(=100) → (base_channels=512) 채널로 upsampling: 1×1 → 4×4
        self.initial = nn.ConvTranspose2d(
            in_channels=latent_dim,
            out_channels=base_channels,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 각 단계별 채널 수 (4×4→8×8→16×16→…→256×256)
        # 512→256→128→64→32→16→8 순서로 채널 수를 반으로 줄임
        ch_list = [
            base_channels,            # 4×4 단계: 512
            base_channels // 2,       # 8×8 단계: 256
            base_channels // 4,       # 16×16 단계: 128
            base_channels // 8,       # 32×32 단계: 64
            base_channels // 16,      # 64×64 단계: 32
            base_channels // 32,      # 128×128 단계: 16
            base_channels // 64       # 256×256 단계: 8
        ]

        # 7단계씩 ResidualBlock + Upsample(ConvTranspose2d) 쌍을 만듦
        self.resblocks = nn.ModuleList()
        self.upconvs   = nn.ModuleList()
        in_c = base_channels  # 첫 번째는 512 채널

        for out_c in ch_list[1:]:
            # 1) 해당 해상도에서 in_c 채널로 ResidualBlock
            self.resblocks.append(ResidualBlock(in_c, expension=2, cardinality=2))
            # 2) Upsampling: in_c → out_c, kernel=4, stride=2, padding=1
            self.upconvs.append(
                nn.ConvTranspose2d(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            in_c = out_c

        # “(B, 8, 256, 256)” → “(B, 3, 256, 256)” 변환
        self.res_final = ResidualBlock(ch_list[-1], expension=2, cardinality=2)
        self.conv_to_rgb = nn.Conv2d(
            in_channels = ch_list[-1],
            out_channels = 3,
            kernel_size = 3,
            padding = 1,
            bias = False
        )
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim, 1, 1)
        """
        # 1) latent_dim → (base_channels, 4, 4)
        x = self.initial(z)           # (B, 512, 4, 4)
        x = self.activation(x)

        # 2) 7단계: ResidualBlock → Upsample (ConvTranspose2d) → LeakyReLU
        for rb, up in zip(self.resblocks, self.upconvs):
            x = rb(x)                                # skip connection 유지, (B, in_c, H, W)
            x = up(x)                                # (B, out_c, 2H, 2W)
            x = self.activation(x)

        # 3) 마지막 해상도 256×256에서 채널 8로 유지한 뒤 RGB 변환
        x = self.res_final(x)                         # (B, 8, 256, 256)
        x = self.conv_to_rgb(x)                       # (B, 3, 256, 256)
        return self.tanh(x)                           # 값 범위를 [-1, +1]로 맞춤


class Discriminator(nn.Module):
    """
    GAN Discriminator for 256×256 이미지 판별.
    입력: image ∈ ℝ^{3×256×256}, shape = (B, 3, 256, 256)
    출력: prob ∈ ℝ^{1}, shape = (B, 1) ‒ ‘진짜일 확률’
    """
    def __init__(self, base_channels: int = 8):
        super().__init__()
        self.base_channels = base_channels
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 1) 이미지(3 채널) → (base_channels=8) 채널로 변환, 해상도 유지 (256×256)
        self.conv_in = nn.Conv2d(
            in_channels = 3,
            out_channels = base_channels,
            kernel_size = 3,
            padding = 1,
            bias = False
        )

        # 2) 7단계씩 ResidualBlock → Downsample(Conv2d stride=2) 순서
        # 채널: 8→16→32→64→128→256→512, 해상도: 256→128→64→32→16→8→4
        ch_list = [
            base_channels,            # 256×256 → 8
            base_channels * 2,        # 128×128 → 16
            base_channels * 4,        # 64×64   → 32
            base_channels * 8,        # 32×32   → 64
            base_channels * 16,       # 16×16   → 128
            base_channels * 32,       # 8×8     → 256
            base_channels * 64        # 4×4     → 512
        ]

        self.resblocks = nn.ModuleList()
        self.downconvs = nn.ModuleList()
        in_c = ch_list[0]  # 처음 채널은 8

        for out_c in ch_list[1:]:
            # ① ResidualBlock(in_c)
            self.resblocks.append(ResidualBlock(in_c, expension=2, cardinality=2))
            # ② Downsample: in_c → out_c, kernel=4, stride=2, padding=1
            self.downconvs.append(
                nn.Conv2d(
                    in_channels = in_c,
                    out_channels = out_c,
                    kernel_size = 4,
                    stride = 2,
                    padding = 1,
                    bias = False
                )
            )
            in_c = out_c

        # 3) 맨 마지막 (B, 512, 4, 4) 상태 →(fc→sigmoid)→ (B, 1)
        self.res_final = ResidualBlock(ch_list[-1], expension=2, cardinality=2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(ch_list[-1] * 4 * 4, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 256, 256)
        """
        # 1) 입력 채널 3 → 8, 활성화
        h = self.conv_in(x)                     # (B, 8, 256, 256)
        h = self.activation(h)

        # 2) 7단계 ResidualBlock → Downsample → LeakyReLU
        for rb, down in zip(self.resblocks, self.downconvs):
            h = rb(h)                           # (B, in_c,    H,    W)
            h = down(h)                         # (B, out_c, H/2, W/2)
            h = self.activation(h)

        # 3) 마지막 (B, 512, 4, 4) 상태 처리
        h = self.res_final(h)                   # (B, 512, 4, 4)
        h = self.flatten(h)                     # (B, 512*4*4)
        logits = self.fc(h)                     # (B, 1)  (logit)
        return self.sigmoid(logits)             # (B, 1)  확률 반환
