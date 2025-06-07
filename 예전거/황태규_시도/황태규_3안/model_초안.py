# # model.py

# import torch
# import torch.nn as nn
# from util import ResidualBlock  # util.py에 정의된 Normalization-Free ResidualBlock 사용

# class Generator(nn.Module):
#     """
#     (B, NoiseDim) 또는 (B, NoiseDim, 1, 1) 형태의 latent 벡터를 받아
#     (B, 3, 256, 256) 이미지를 생성하는 네트워크.
#     내부에 ResidualBlock을 여러 단계 거쳐 배치.
#     """
#     def __init__(
#         self,
#         NoiseDim: int = 100,
#         BaseChannels: int = 256,
#         ExpansionFactor: int = 2,
#         NumResBlocksPerStage: int = 2
#     ):
#         super().__init__()
#         self.noise_dim = NoiseDim
#         self.base_ch = BaseChannels
#         self.num_blocks = NumResBlocksPerStage

#         # 1) latent(NoiseDim) → (BaseChannels × 4 × 4)
#         self.fc = nn.Linear(NoiseDim, BaseChannels * 4 * 4)
#         # 2) (BaseChannels, 4×4) → (BaseChannels, 4×4) conv
#         self.conv_in = nn.Conv2d(
#             in_channels=BaseChannels,
#             out_channels=BaseChannels,
#             kernel_size=3,
#             padding=1,
#             bias=True
#         )
#         self.lrelu = nn.LeakyReLU(0.1, inplace=True)

#         # 각 해상도별 채널 리스트 (4×4 단계는 BaseChannels 그대로)
#         # 4×4  → BaseChannels
#         # 8×8  → BaseChannels//2
#         # 16×16→ BaseChannels//4
#         # 32×32→ BaseChannels//8
#         # 64×64→ BaseChannels//16
#         # 128×128→BaseChannels//32
#         # 256×256→BaseChannels//64
#         self.ch_list = [
#             BaseChannels,
#             BaseChannels // 2,
#             BaseChannels // 4,
#             BaseChannels // 8,
#             BaseChannels // 16,
#             BaseChannels // 32,
#             BaseChannels // 64
#         ]

#         # ResidualBlock과 Upsample(ConvTranspose2d) 레이어를 순차적으로 정의
#         self.resblocks = nn.ModuleList()
#         self.upconvs = nn.ModuleList()

#         # 첫 단계(4×4 단계)에서는 ResidualBlock을 적용하지 않고 바로 conv_in → leakyReLU
#         # 이후 6단계(4→8, 8→16, …, 128→256)에 걸쳐서 각각
#         #   [ResidualBlock × NumResBlocksPerStage] → ConvTranspose2d(upsample) → LeakyReLU
#         for stage in range(len(self.ch_list) - 1):
#             in_c = self.ch_list[stage]
#             out_c = self.ch_list[stage + 1]
#             # 해당 stage별로 NumResBlocksPerStage개만큼 ResidualBlock
#             for _ in range(NumResBlocksPerStage):
#                 self.resblocks.append(
#                     ResidualBlock(
#                         InputChannels=in_c,
#                         Cardinality=2,
#                         ExpansionFactor=ExpansionFactor,
#                         KernelSize=3,
#                         VarianceScalingParameter=NumResBlocksPerStage * len(self.ch_list)
#                     )
#                 )
#             # Upsample: ConvTranspose2d(in_c → out_c, kernel=4, stride=2, padding=1)
#             self.upconvs.append(
#                 nn.ConvTranspose2d(
#                     in_channels=in_c,
#                     out_channels=out_c,
#                     kernel_size=4,
#                     stride=2,
#                     padding=1,
#                     bias=True
#                 )
#             )

#         # 마지막 해상도(256×256, 채널 = ch_list[-1]) 단계 후에 다시 ResidualBlock × NumResBlocksPerStage
#         self.res_final = nn.Sequential(
#             *[
#                 ResidualBlock(
#                     InputChannels=self.ch_list[-1],
#                     Cardinality=2,
#                     ExpansionFactor=ExpansionFactor,
#                     KernelSize=3,
#                     VarianceScalingParameter=NumResBlocksPerStage * len(self.ch_list)
#                 )
#                 for _ in range(NumResBlocksPerStage)
#             ]
#         )
#         # 마지막으로 (채널 = ch_list[-1]) → 3 (RGB)
#         self.conv_to_rgb = nn.Conv2d(
#             in_channels=self.ch_list[-1],
#             out_channels=3,
#             kernel_size=3,
#             padding=1,
#             bias=True
#         )
#         self.tanh = nn.Tanh()

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         z: (B, NoiseDim) 또는 (B, NoiseDim, 1, 1)
#         """
#         B = z.size(0)
#         # 1) latent → (B, BaseChannels, 4, 4)
#         if z.ndim == 2:
#             x = self.fc(z).view(B, self.base_ch, 4, 4)
#         else:
#             # (B, NoiseDim, 1, 1) 형태인 경우
#             x = z.view(B, self.noise_dim)
#             x = self.fc(x).view(B, self.base_ch, 4, 4)

#         # 2) conv_in → LeakyReLU
#         x = self.lrelu(self.conv_in(x))  # (B, BaseChannels, 4, 4)

#         # 3) 각 업샘플 스테이지 순회
#         #    - 먼저 해당 스테이지 단계에 맞춰 ResidualBlock × self.num_blocks만큼 적용
#         #    - 그다음 ConvTranspose2d(upsample) → LeakyReLU
#         idx = 0  # self.resblocks 인덱스 추적용
#         for stage_idx, up in enumerate(self.upconvs):
#             in_c = self.ch_list[stage_idx]  # 예: 256 → 128 → ...
#             # 해당 채널(in_c)로 된 ResidualBlock self.num_blocks개
#             for _ in range(self.num_blocks):
#                 x = self.resblocks[idx](x)
#                 idx += 1
#             # upsample: (in_c, H, W) → (out_c, 2H, 2W)
#             x = self.lrelu(up(x))

#         # 4) 마지막 해상도(256×256)에서 ResidualBlock × self.num_blocks
#         x = self.res_final(x)         # (B, self.ch_list[-1], 256, 256)
#         x = self.conv_to_rgb(x)       # (B, 3, 256, 256)
#         return self.tanh(x)           # 값 범위를 [-1, 1]로


# class Discriminator(nn.Module):
#     """
#     (B, 3, 256, 256) 이미지를 입력받아 raw logit (B,) 을 반환.
#     내부에 ResidualBlock을 여러 단계 쌓고, Conv2d(stride=2)로 다운샘플링.
#     """
#     def __init__(
#         self,
#         BaseChannels: int = 4,
#         ExpansionFactor: int = 2,
#         NumResBlocksPerStage: int = 2
#     ):
#         super().__init__()
#         self.base_ch = BaseChannels
#         self.num_blocks = NumResBlocksPerStage
#         self.lrelu = nn.LeakyReLU(0.1, inplace=True)

#         # 1) 입력 이미지(3) → BaseChannels (예: 4)
#         self.conv_in = nn.Conv2d(
#             in_channels=3,
#             out_channels=BaseChannels,
#             kernel_size=3,
#             padding=1,
#             bias=True
#         )

#         # 각 해상도 단계별 채널 수 (256×256→4, 128×128→8, 64×64→16, 32×32→32, 16×16→64, 8×8→128, 4×4→256)
#         self.ch_list = [
#             BaseChannels,        # 256×256 → 4
#             BaseChannels * 2,    # 128×128 → 8
#             BaseChannels * 4,    # 64×64   → 16
#             BaseChannels * 8,    # 32×32   → 32
#             BaseChannels * 16,   # 16×16   → 64
#             BaseChannels * 32,   # 8×8     → 128
#             BaseChannels * 64    # 4×4     → 256
#         ]

#         # ResidualBlock과 Downsample(Conv2d stride=2) 레이어 정의
#         self.resblocks = nn.ModuleList()
#         self.downconvs = nn.ModuleList()

#         for stage in range(len(self.ch_list) - 1):
#             in_c = self.ch_list[stage]
#             out_c = self.ch_list[stage + 1]
#             # 해당 단계별로 NumResBlocksPerStage개만큼 ResidualBlock
#             for _ in range(NumResBlocksPerStage):
#                 self.resblocks.append(
#                     ResidualBlock(
#                         InputChannels=in_c,
#                         Cardinality=2,
#                         ExpansionFactor=ExpansionFactor,
#                         KernelSize=3,
#                         VarianceScalingParameter=NumResBlocksPerStage * len(self.ch_list)
#                     )
#                 )
#             # Downsample: Conv2d(in_c → out_c, kernel=4, stride=2, padding=1)
#             self.downconvs.append(
#                 nn.Conv2d(
#                     in_channels=in_c,
#                     out_channels=out_c,
#                     kernel_size=4,
#                     stride=2,
#                     padding=1,
#                     bias=True
#                 )
#             )

#         # 마지막 해상도(4×4, 채널 = self.ch_list[-1])에서 ResidualBlock × self.num_blocks
#         self.res_final = nn.Sequential(
#             *[
#                 ResidualBlock(
#                     InputChannels=self.ch_list[-1],
#                     Cardinality=2,
#                     ExpansionFactor=ExpansionFactor,
#                     KernelSize=3,
#                     VarianceScalingParameter=NumResBlocksPerStage * len(self.ch_list)
#                 )
#                 for _ in range(self.num_blocks)
#             ]
#         )
#         # 4×4 → raw logit
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(self.ch_list[-1] * 4 * 4, 1, bias=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 3, 256, 256)
#         returns: (B,) raw logit
#         """
#         h = self.lrelu(self.conv_in(x))  # (B, 4, 256, 256)

#         idx = 0
#         # 각 다운샘플 단계별로
#         for stage_idx, down in enumerate(self.downconvs):
#             in_c = self.ch_list[stage_idx]
#             # 해당 단계별로 ResidualBlock × self.num_blocks
#             for _ in range(self.num_blocks):
#                 h = self.resblocks[idx](h)
#                 idx += 1
#             # downsample: (in_c, H, W) → (out_c, H/2, W/2)
#             h = self.lrelu(down(h))

#         # 마지막 4×4 단계
#         h = self.res_final(h)             # (B, 256, 4, 4)
#         h = self.flatten(h)               # (B, 256*4*4)
#         logit = self.fc(h)                # (B, 1)
#         return logit.view(h.shape[0])     # (B,)


#혹시 몰라서 정리해두는 32*32 전용
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import Convolution, BiasedActivation, ResidualBlock

class GBlock(nn.Module):
    """
    Generator residual block with upsampling:
      1) Upsample by factor 2 (bilinear interpolation)
      2) 1x1 projection to adjust channels
      3) Two normalization-free residual blocks
    """
    def __init__(self, C_in, C_out, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # channel projection after upsampling
        self.proj = Convolution(C_in, C_out, KernelSize=1, ActivationGain=1.0)
        # two residual blocks at the new resolution
        self.res1 = ResidualBlock(C_out, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)
        self.res2 = ResidualBlock(C_out, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)

    def forward(self, x):
        x = self.upsample(x)
        x = self.proj(x)
        x = self.res1(x)
        x = self.res2(x)
        return x

class Generator(nn.Module):
    """
    R3GAN Generator for 32x32 output images
    - latent z (B, NoiseDim) -> Linear -> reshape to (B, 1.5*C0, 4, 4)
    - initial conv + activation
    - series of GBlocks to upsample: 4x4->8x8->16x16->32x32
    - 1x1 conv to map channels to RGB, Tanh output
    """
    def __init__(
        self,
        NoiseDim: int = 100,
        BaseChannels: list = [256, 128, 64, 32],  # stages: 4x4,8x8,16x16,32x32
        Cardinality: int = 2,
        ExpansionFactor: int = 2,
        KernelSize: int = 3
    ):
        super().__init__()
        self.noise_dim = NoiseDim
        C0 = BaseChannels[0]
        # (A) latent z -> (B, 1.5*C0, 4, 4)
        self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4)
        # (B) initial conv + activation at 4x4
        self.conv4x4 = nn.Conv2d(int(1.5 * C0), int(1.5 * C0), kernel_size=3, padding=1)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        # (C) GBlocks for upsampling to 32x32
        var_scaling = len(BaseChannels)
        self.gblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin = BaseChannels[i]
            Cout = BaseChannels[i + 1]
            self.gblocks.append(
                GBlock(
                    C_in=Cin * int(1.5**0),  # channels after previous stage
                    C_out=Cout,
                    Cardinality=Cardinality,
                    ExpansionFactor=ExpansionFactor,
                    KernelSize=KernelSize,
                    VarianceScalingParameter=var_scaling
                )
            )
        # (D) final 1x1 conv to 3 channels (RGB)
        C_last = BaseChannels[-1]
        self.to_rgb = nn.Conv2d(int(1.5 * C_last), 3, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        # latent -> fc -> reshape
        if z.ndim == 2:
            x = self.fc(z).view(B, int(1.5 * self.BaseChannels[0]), 4, 4)
        else:
            z_flat = z.view(B, self.noise_dim)
            x = self.fc(z_flat).view(B, int(1.5 * self.BaseChannels[0]), 4, 4)
        # initial conv + activation
        x = self.lrelu(self.conv4x4(x))
        # apply GBlocks
        for gblock in self.gblocks:
            x = gblock(x)
        # to RGB and tanh
        x = self.to_rgb(x)
        return self.tanh(x)

class DBlock(nn.Module):
    """
    Discriminator residual block with downsampling:
      1) Two normalization-free residual blocks
      2) Downsample by factor 2 (bilinear interpolation)
      3) 1x1 projection if channels change
    """
    def __init__(self, C_in, C_out, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter):
        super().__init__()
        self.res1 = ResidualBlock(C_in, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)
        self.res2 = ResidualBlock(C_in, Cardinality, ExpansionFactor, KernelSize, VarianceScalingParameter)
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        if C_in != C_out:
            self.proj = Convolution(C_in, C_out, KernelSize=1)
        else:
            self.proj = None

    def forward(self, x):
        x = self.res1(x)
        x = self.res2(x)
        x = self.downsample(x)
        if self.proj is not None:
            x = self.proj(x)
        return x

class Discriminator(nn.Module):
    """
    R3GAN Discriminator for 32x32 input images
    - input RGB (B,3,32,32) -> 1x1 Conv -> leakyReLU
    - series of DBlocks: 32->16->8->4 downsampling
    - final 4x4 features -> flatten -> linear -> raw logit
    """
    def __init__(
        self,
        BaseChannels: list = [4, 8, 16, 32],  # stages: 32x32,16x16,8x8,4x4
        Cardinality: int = 2,
        ExpansionFactor: int = 2,
        KernelSize: int = 3
    ):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.from_rgb = nn.Conv2d(3, int(1.5 * BaseChannels[0]), kernel_size=1)
        var_scaling = len(BaseChannels)
        # build DBlocks for downsampling
        self.dblocks = nn.ModuleList()
        for i in range(len(BaseChannels) - 1):
            Cin = BaseChannels[i]
            Cout = BaseChannels[i + 1]
            self.dblocks.append(
                DBlock(
                    C_in=Cin,
                    C_out=Cout,
                    Cardinality=Cardinality,
                    ExpansionFactor=ExpansionFactor,
                    KernelSize=KernelSize,
                    VarianceScalingParameter=var_scaling
                )
            )
        # final linear from flattened 4x4 features
        final_ch = int(1.5 * BaseChannels[-1])
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(final_ch * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # from RGB
        h = self.lrelu(self.from_rgb(x))
        # apply DBlocks
        for dblock in self.dblocks:
            h = dblock(h)
        # flatten and linear
        h = self.flatten(h)
        logit = self.fc(h)
        return logit.view(h.shape[0])
