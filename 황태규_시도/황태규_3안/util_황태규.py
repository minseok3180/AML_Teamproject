# util.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

###논문 R3GAN/Networks.py 참조
def MSRInitializer(Layer: nn.Module, ActivationGain: float = 1.0) -> nn.Module:
    """
    ActivationGain을 활용한 R3GAN 방식의 가중치 초기화 
    """
    with torch.no_grad():
        weight = Layer.weight.data
        fan_in = weight.size(1) * weight[0][0].numel()  # 입력 채널 수 × (kernel_w × kernel_h)
        if ActivationGain == 0 or fan_in == 0:
            weight.zero_()
        else:
            std = ActivationGain / math.sqrt(fan_in)
            weight.normal_(0, std)
    return Layer

###논문 R3GAN/Networks.py 참조
class Convolution(nn.Module):
    """
    R3GAN 스타일 Convolution
    nn.Conv2d(bias=False)를 MSRInitializer로 초기화
    """
    def __init__(
        self,
        InputChannels: int,
        OutputChannels: int,
        KernelSize: int,
        Groups: int = 1,
        ActivationGain: float = 1.0
    ):
        super().__init__()
        padding = (KernelSize - 1) // 2
        ###Residualblock 내부에서는 bias를 skip, Biasedactivation에서 적용
        conv = nn.Conv2d(
            in_channels=InputChannels,
            out_channels=OutputChannels,
            kernel_size=KernelSize,
            stride=1,
            padding=padding,
            groups=Groups,
            bias=False
        )
        self.Layer = MSRInitializer(conv, ActivationGain=ActivationGain)

    ### bias가 없으므로 self.Layer.weight.to(x.dtype)를 그대로 넣어서 dtype 변환 직접 제어
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.Layer.weight.to(x.dtype),
            padding=self.Layer.padding,
            groups=self.Layer.groups
        )

###논문 R3GAN / FusedOperators.py 참조
class BiasedActivation(nn.Module):
    """
    When repeating the Conv → ReLU → Conv → ReLU -> Conv structure, put the learnable bias back and forth of the ReLU
    Then, feature distribution이 좀 더 골고루 퍼지고, 죽은 ReLU(dead neuron) 발생률 감소
    편향(bias) + LeakyReLU(negative_slope=0.2) + Gain을 한번에 처리
    """
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))  # LeakyReLU(alpha=0.2) 기준
    
    def __init__(self, Channels: int):
        super().__init__()
        # 채널별 학습 가능한 bias 파라미터
        self.bias = nn.Parameter(torch.zeros(Channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # bias를 채널 차원에 맞춰 broadcast
        x = x + self.bias.view(1, -1, 1, 1)
        # LeakyReLU → Gain 곱셈
        return F.leaky_relu(x, negative_slope=0.2) * BiasedActivation.Gain


class ResidualBlock(nn.Module):
    """
    R3GAN형 Normalization-Free ResidualBlock:
      1) 1*1 Conv → BiasedActivation
      2) 3*3 Grouped Conv → BiasedActivation
      3) 1*1 Conv (ActivationGain=0) → skip connection
    """
    ### 변수만 정의해두고 세부적인 값은 대부분 model.py에서 정의
    ### Ex) VarianceScalingParameter = num_blocks
    def __init__(
        self,
        InputChannels: int,
        Cardinality: int,     
        ExpansionFactor: int, 
        KernelSize: int, 
        VarianceScalingParameter: float
    ):
        super().__init__()
        num_linear = 3
        expanded_c = InputChannels * ExpansionFactor
        # BiasedActivation.Gain과 VarianceScalingParameter를 조합하여 ActivationGain 계산
        activation_gain = (
            BiasedActivation.Gain
            * VarianceScalingParameter ** (-1 / (2 * num_linear - 2))
        )

        # (1) in_c → expanded_c (1×1 conv)
        self.conv1 = Convolution(
            InputChannels, expanded_c, KernelSize=1, ActivationGain=activation_gain
        )
        # (2) expanded_c → expanded_c (3×3, 그룹 conv)
        self.conv2 = Convolution(
            expanded_c, expanded_c, KernelSize=KernelSize,
            Groups=Cardinality, ActivationGain=activation_gain
        )
        # (3) expanded_c → in_c (1×1 conv), ActivationGain=0으로 가중치 0으로 초기화
        self.conv3 = Convolution(
            expanded_c, InputChannels, KernelSize=1, ActivationGain=0.0
        )

        # 두 번의 BiasedActivation
        self.act1 = BiasedActivation(expanded_c)
        self.act2 = BiasedActivation(expanded_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.act2(y)
        y = self.conv3(y)
        return x + y
