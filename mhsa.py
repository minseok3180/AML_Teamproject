import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.to_qkv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1, bias=False)
        self.unify_heads = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.scale = self.head_dim ** -0.5  # Scaled dot-product

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W

        # QKV 추출
        qkv = self.to_qkv(x)                   # [B, 3C, H, W]
        qkv = qkv.view(B, 3, self.num_heads, self.head_dim, N)
        q, k, v = qkv[:,0], qkv[:,1], qkv[:,2] # 각각 [B, heads, head_dim, N]

        # Attention score 계산
        # (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        q = q.permute(0,1,3,2)  # [B, heads, N, head_dim]
        k = k                    # [B, heads, head_dim, N]
        attn = torch.softmax(torch.matmul(q, k) * self.scale, dim=-1)  # [B, heads, N, N]

        # Value에 어텐션 적용
        # (B, heads, head_dim, N) @ (B, heads, N, N) -> (B, heads, head_dim, N)
        v = v                    # [B, heads, head_dim, N]
        out = torch.matmul(v, attn.permute(0,1,3,2))  # [B, heads, head_dim, N]
        out = out.reshape(B, C, H, W)  # 머리들을 채널 차원으로 합치기

        # Projection & residual
        out = self.unify_heads(out)    # [B, C, H, W]
        return out + x
