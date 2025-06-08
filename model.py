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
        BaseChannels: list = [256, 128, 64, 32, 16, 8, 4],
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
        BaseChannels: list = [4, 8, 16, 32, 64, 128, 256],
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

# class GBlock(nn.Module):
#     """
#       Input channels: 1.5 * C_in
#       (1) Conv1x1: (1.5*C_in) → (1.5*C_out)
#       (2) Bilinear Upsample *2 (double spatial size)
#       (3) ResidualBlock 1: channels (1.5*C_out)
#       (4) ResidualBlock 2: channels (1.5*C_out)
#     """
#     def __init__(
#         self,
#         C_in: int,
#         C_out: int,
#         Cardinality: int = 2,      # reduces parameter's number, computation
#         ExpansionFactor: int = 2,  #1.5 -> 3
#         KernelSize: int = 3   #Standard convolution size
#     ):
#         super().__init__()
#         # Block input channels = 1.5 * C_in
#         self.in_ch = int(1.5 * C_in)
#         # Block output channels = 1.5 * C_out
#         self.out_ch = int(1.5 * C_out)

#         # (1) Conv1x1: (1.5*C_in) → (1.5*C_out)
#         self.conv1x1 = nn.Conv2d(
#             in_channels=self.in_ch,
#             out_channels=self.out_ch,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True
#         )

#         # (3,4) Define two separate ResidualBlocks
#         num_blocks = 2
#         var_scaling = num_blocks  
#         self.res_block1 = ResidualBlock(
#             InputChannels=self.out_ch,
#             Cardinality=Cardinality,
#             ExpansionFactor=ExpansionFactor,
#             KernelSize=KernelSize,
#             VarianceScalingParameter=var_scaling
#         )
#         self.res_block2 = ResidualBlock(
#             InputChannels=self.out_ch,
#             Cardinality=Cardinality,
#             ExpansionFactor=ExpansionFactor,
#             KernelSize=KernelSize,
#             VarianceScalingParameter=var_scaling
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 1.5*C_in, H, W)
#         returns: (B, 1.5*C_out, 2H, 2W)
#         """
#         # (1) Conv1x1 → change channels
#         x = self.conv1x1(x)  # (B, 1.5*C_out, H, W)

#         # (2) Bilinear 2× upsampling (double spatial size); use interpolate for efficiency
#         x = F.interpolate(
#             x,
#             scale_factor=2,
#             mode="bilinear",
#             align_corners=False,
#             recompute_scale_factor=True
#         )  # (B, 1.5*C_out, 2H, 2W)

#         # (3) 첫 번째 ResidualBlock 
#         x = self.res_block1(x)  # (B, 1.5*C_out, 2H, 2W)

#         # (4) 두 번째 ResidualBlock 
#         x = self.res_block2(x)  # (B, 1.5*C_out, 2H, 2W)

#         return x


# class DBlock(nn.Module):
#     """
#       입력 채널: 1.5 * C_in, 해상도: H * W
#       (1) ResidualBlock 1: 채널 (1.5*C_in), 해상도 유지
#       (2) ResidualBlock 2: 채널 (1.5*C_in), 해상도 유지
#       (3) Bilinear Downsample *0.5 (해상도 절반, 채널 그대로)
#       (4) Conv1x1: (1.5*C_in) → (1.5*C_out)
#     """
#     def __init__(
#         self,
#         C_in: int,
#         C_out: int,
#         Cardinality: int = 2,
#         ExpansionFactor: int = 2,
#         KernelSize: int = 3
#     ):
#         super().__init__()
#         # R3GAN은 블록 입력 채널 = 1.5 * C_in
#         self.in_ch = int(1.5 * C_in)
#         # 블록 출력 채널 = 1.5 * C_out
#         self.out_ch = int(1.5 * C_out)

#         # (1,2) 두 개의 ResidualBlock을 따로 정의
#         num_blocks = 2
#         var_scaling = num_blocks
#         self.res_block1 = ResidualBlock(
#             InputChannels=self.in_ch,
#             Cardinality=Cardinality,
#             ExpansionFactor=ExpansionFactor,
#             KernelSize=KernelSize,
#             VarianceScalingParameter=var_scaling
#         )
#         self.res_block2 = ResidualBlock(
#             InputChannels=self.in_ch,
#             Cardinality=Cardinality,
#             ExpansionFactor=ExpansionFactor,
#             KernelSize=KernelSize,
#             VarianceScalingParameter=var_scaling
#         )

#         # (4) Conv1x1: (1.5*C_in) → (1.5*C_out)
#         self.conv1x1 = nn.Conv2d(
#             in_channels=self.in_ch,
#             out_channels=self.out_ch,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 1.5*C_in, H, W)
#         returns: (B, 1.5*C_out, H/2, W/2)
#         """
#         # (1) 첫 번째 ResidualBlock (채널/해상도 유지)
#         x = self.res_block1(x)  # (B, 1.5*C_in, H, W)

#         # (2) 두 번째 ResidualBlock (채널/해상도 유지)
#         x = self.res_block2(x)  # (B, 1.5*C_in, H, W)

#         # (3) Bilinear 0.5× 다운샘플 (해상도 절반)
#         x = F.interpolate(
#             x,
#             scale_factor=0.5,
#             mode="bilinear",
#             align_corners=False,
#             recompute_scale_factor=True
#         )  # (B, 1.5*C_in, H/2, W/2)

#         # (4) Conv1x1 → 채널 변경
#         x = self.conv1x1(x)  # (B, 1.5*C_out, H/2, W/2)

#         return x


# class Generator(nn.Module):
#     """
#     R3GAN Generator 구현
#     *latent z
#     main함수에서 noise = torch.randn(batch_size, nz, device=device)로 생성
#     fake_imgs = generator(noise)로 latent vector를 generator의 forward에 넣기
#     (1)latent z → (B, 1.5*C0, 4, 4)로 변환
#     self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4) 로 1.5*C_in 적용
#     저차원인 Latent vector를 고차원인 Feature map으로 펼치는 과정
#     효율적이면서 안정적인 점진적 upsampling을 위해 4*4부터 시작
#     (2) x = self.lrelu(self.conv4x4(x)): 4*4로 convolution 수행 + Leakyrelu 적용 
#     negative_slope=0.2로 설정
#     (3) GBlocks 생성 (총 6단계: 4→8→16→32→64→128→256)
#     이후 GBlocks를 통해 점차 해상도 8*8 → 16*16 → … → 256*256(원래 이미지 크기)로 upsampling
#     (4) x = self.to_rgb(x): RGB값 얻기
#     (5) self.tanh = nn.Tanh() : 값을 (-1, 1)로 scaling하여 Discriminator와 정규화 범위 동기화
    
#     출력 범위: Tanh → [-1, +1]
#     """
#     def __init__(
#         self,
#         NoiseDim: int = 100,
#         BaseChannels: list = [256, 128, 64, 32, 16, 8, 4],
#         Cardinality: int = 2,
#         ExpansionFactor: int = 2,
#         KernelSize: int = 3
#     ):
#         super().__init__()
#         self.noise_dim = NoiseDim
#         self.BaseChannels = BaseChannels

#         # (1) latent z → (B, 1.5*C0, 4, 4)로 변환
#         C0 = BaseChannels[0]  
#         self.fc = nn.Linear(NoiseDim, int(1.5 * C0) * 4 * 4) 

#         # (2) x = self.lrelu(self.conv4x4(x)): 4*4로 convolution 수행 + Leakyrelu 적용
#         self.conv4x4 = nn.Conv2d(
#             in_channels=int(1.5 * C0),
#             out_channels=int(1.5 * C0),
#             kernel_size=3,
#             padding=1,
#             bias=True
#         )
#         self.lrelu = nn.LeakyReLU(0.2, inplace=True)

#         # (3) GBlocks 생성 (총 6단계: 4→8→16→32→64→128→256)
#         self.gblocks = nn.ModuleList()
#         for i in range(len(BaseChannels) - 1):
#             Cin = BaseChannels[i]
#             Cout = BaseChannels[i + 1]
#             self.gblocks.append(
#                 GBlock(
#                     C_in=Cin,
#                     C_out=Cout,
#                     Cardinality=Cardinality,
#                     ExpansionFactor=ExpansionFactor,
#                     KernelSize=KernelSize
#                 )
#             )

#         # (4) x = self.to_rgb(x): RGB값 얻기
#         C6 = BaseChannels[-1]  # 예: 4
#         self.to_rgb = nn.Conv2d(
#             in_channels=int(1.5 * C6),
#             out_channels=3,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True
#         )

#         # (5) self.tanh = nn.Tanh() : 값을 (-1, 1)로 scaling하여 Discriminator와 정규화 범위 동기화
#         self.tanh = nn.Tanh()

#     def forward(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         z: (B, NoiseDim) 또는 (B, NoiseDim, 1, 1)
#         returns: (B, 3, 256, 256)
#         """
#         B = z.size(0)
#         C0 = self.BaseChannels[0]

#         # (1) latent z → (B, 1.5*C0, 4, 4)로 변환
#         x = self.fc(z).view(B, int(1.5 * C0), 4, 4)

#         # (2) x = self.lrelu(self.conv4x4(x)): 4*4로 convolution 수행 + Leakyrelu 적용
#         x = self.lrelu(self.conv4x4(x))  # (B, 1.5*C0, 4, 4)

#         # (3) GBlocks를 순서대로 적용 (4→8→16→32→64→128→256)
#         for gblock in self.gblocks:
#             x = gblock(x)

#         # (4) x = self.to_rgb(x): RGB값 얻기
#         x = self.to_rgb(x)        # (B, 3, 256, 256)
        
#         # (5) self.tanh = nn.Tanh() : 값을 (-1, 1)로 scaling하여 Discriminator와 정규화 범위 동기화
#         return self.tanh(x)       # (-1, +1) 범위


# class Discriminator(nn.Module):
#     """
#     R3GAN Discriminator 구현
#     (1) RGB(3) → (B, 1.5*C0, 4, 4)로 변환
#     (2) DBlocks 정의 (총 len(BaseChannels)-1개 = 6개)
#     DBlocks를 통해 해상도 256→128→64→32→16→8→4 (총 6번 다운샘플링)
#     (3) self.fc = nn.Linear(final_ch * 4 * 4, 1, bias=True): 이미지의 real, fake를 판단하는 raw logit 저장
#     """
#     def __init__(
#         self,
#         BaseChannels: list = [4, 8, 16, 32, 64, 128, 256],
#         Cardinality: int = 2,
#         ExpansionFactor: int = 2,
#         KernelSize: int = 3
#     ):
#         super().__init__()
#         self.BaseChannels = BaseChannels
#         self.lrelu = nn.LeakyReLU(0.2, inplace=True)

#         # (1) RGB(3) → (B, 1.5*C0, 4, 4)로 변환
#         C0 = BaseChannels[0]  # 4
#         self.from_rgb = nn.Conv2d(
#             in_channels=3,
#             out_channels=int(1.5 * C0),
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=True
#         )

#         # (2) DBlocks 정의 (총 len(BaseChannels)-1개 = 6개)
#         self.dblocks = nn.ModuleList()
#         for i in range(len(BaseChannels) - 1):
#             Cin = BaseChannels[i]
#             Cout = BaseChannels[i + 1]
#             self.dblocks.append(
#                 DBlock(
#                     C_in=Cin,
#                     C_out=Cout,
#                     Cardinality=Cardinality,
#                     ExpansionFactor=ExpansionFactor,
#                     KernelSize=KernelSize
#                 )
#             )

#         # (3) self.fc = nn.Linear(final_ch * 4 * 4, 1, bias=True): 이미지의 real, fake를 판단하는 raw logit 저장
#         final_ch = int(1.5 * BaseChannels[-1]) #256*1.5 = 384
#         self.flatten = nn.Flatten() #(B, 384, 4, 4) -> (B, 6144)
#         self.fc = nn.Linear(final_ch * 4 * 4, 1, bias=True) 

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 3, 256, 256)
#         returns: (B,) raw logit
#         """
#         # (1) RGB(3) → (B, 1.5*C0, 4, 4)로 변환
#         h = self.lrelu(self.from_rgb(x))  # (B, 6, 256, 256)

#         # (2) DBlocks 정의 (총 len(BaseChannels)-1개 = 6개)
#         for dblock in self.dblocks:
#             h = dblock(h)

#         # (3) self.fc = nn.Linear(final_ch * 4 * 4, 1, bias=True): 이미지의 real, fake를 판단하는 raw logit 저장
#         h = self.flatten(h)       
#         logit = self.fc(h)        
#         return logit.view(h.shape[0])  
