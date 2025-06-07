# util.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# MSR IN
# FID.py

# #Existing Code
# class ResidualBlock(nn.Module):
#     def __init__(self, 
#                  in_c: int,
#                  expension: int = 2, 
#                  cardinality: int = 2):
        
#         super().__init__()
#         self.in_c = in_c
#         self.cardinality = cardinality
#         self.expanded_c = in_c * expension

#         self.conv1 = nn.Conv2d(self.in_c, self.expanded_c, kernel_size=1, padding=0)
#         self.conv2 = nn.Conv2d(self.expanded_c, self.expanded_c, kernel_size=3, padding=1, groups=cardinality)
#         self.conv3 = nn.Conv2d(self.expanded_c, self.in_c, kernel_size=1, padding=0)

#         self.bias1 = nn.Parameter(torch.zeros(self.expanded_c))
#         self.bias2 = nn.Parameter(torch.zeros(self.expanded_c))
#         self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True) # slope ?占쏙옙?占쏙옙
#         self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         y = self.conv1(x)
#         y = self.leaky_relu1(y + self.bias1.view(1, -1, 1, 1)) #怨좑옙??
#         y = self.conv2(y)
#         y = self.leaky_relu2(y + self.bias2.view(1, -1, 1, 1))
#         y = self.conv3(y)
#         return x + y

### Reference: R3GAN/Networks.py
def MSRInitializer(Layer: nn.Module, ActivationGain: float = 1.0) -> nn.Module:
    """
    R3GAN-style weight initialization using ActivationGain
    """
    with torch.no_grad():
        weight = Layer.weight.data
        fan_in = weight.size(1) * weight[0][0].numel()  # �엯�젰 梨꾨꼸 �닔 횞 (kernel_w 횞 kernel_h)
        if ActivationGain == 0 or fan_in == 0:
            weight.zero_()
        else:
            std = ActivationGain / math.sqrt(fan_in)
            weight.normal_(0, std)
    return Layer

### Reference: R3GAN/Networks.py
class Convolution(nn.Module):
    """
    R3GAN-style Convolution  
    nn.Conv2d(bias=False) initialized by MSRInitializer
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
        ### Skip bias in ResidualBlock; bias is applied in BiasedActivation
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

    ### Since there is no bias, dtype is controlled manually using 'self.Layer.weight.to(x.dtype)'
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.Layer.weight.to(x.dtype),
            padding=self.Layer.padding,
            groups=self.Layer.groups
        )

### See paper R3GAN/FusedOperators.py
class BiasedActivation(nn.Module):
    """
    When repeating the Conv → ReLU → Conv → ReLU -> Conv structure, put the learnable bias before and after the ReLU.
    This helps the feature distribution spread out more evenly and reduces the rate of dead ReLU neurons.
    / Handles bias (learnable) + LeakyReLU(negative_slope=0.2) + Gain at once.
    """
    Gain = math.sqrt(2 / (1 + 0.2 ** 2))  # For LeakyReLU(alpha=0.2)
    
    def __init__(self, Channels: int):
        super().__init__()
        # Channel-wise learnable bias parameter
        self.bias = nn.Parameter(torch.zeros(Channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # Broadcast bias to channel dimension
        x = x + self.bias.view(1, -1, 1, 1)
        # Apply LeakyReLU → then multiply Gain
        return F.leaky_relu(x, negative_slope=0.2) * BiasedActivation.Gain
    
class ResidualBlock(nn.Module):
    """
    R3GAN-style Normalization-Free ResidualBlock:
      1) 1*1 Conv → BiasedActivation
      2) 3*3 Grouped Conv → BiasedActivation
      3) 1*1 Conv (ActivationGain=0) → skip connection
    """
    ### Only variable definitions here; detailed values are mostly defined in model.py
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
        # Compute ActivationGain by combining BiasedActivation.Gain and VarianceScalingParameter
        activation_gain = (
            BiasedActivation.Gain
            * VarianceScalingParameter ** (-1 / (2 * num_linear - 2))
        )

        # (1) in_c → expanded_c (1×1 conv)
        self.conv1 = Convolution(
            InputChannels, expanded_c, KernelSize=1, ActivationGain=activation_gain
        )
        # (2) expanded_c → expanded_c (3×3, grouped conv)
        self.conv2 = Convolution(
            expanded_c, expanded_c, KernelSize=KernelSize,
            Groups=Cardinality, ActivationGain=activation_gain
        )
        # (3) expanded_c → in_c (1×1 conv), ActivationGain=0 to initialize weights to zero
        self.conv3 = Convolution(
            expanded_c, InputChannels, KernelSize=1, ActivationGain=0.0
        )

        # Two BiasedActivation layers
        self.act1 = BiasedActivation(expanded_c)
        self.act2 = BiasedActivation(expanded_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.act2(y)
        y = self.conv3(y)
        return x + y
