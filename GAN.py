import os
from typing import Callable, Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Optional but recommended dependencies for the dataloader helper
try:
    from torchvision import transforms, datasets
except ImportError as e:  # pragma: no cover
    raise ImportError("torchvision must be installed to use the make_dataloader helper.") from e

# Generator
class Generator(nn.Module):

    def __init__(self, latent_dim: int = 100, img_channels: int = 3, fmap: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, fmap * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(fmap * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(fmap * 8, fmap * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(fmap * 4, fmap * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(fmap * 2, fmap, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap),
            nn.ReLU(True),

            nn.ConvTranspose2d(fmap, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),  # output in [‑1, 1]
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (N, latent_dim) → (N, C, H, W)
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

# Discriminator
class Discriminator(nn.Module):
    """DCGAN‑style discriminator."""

    def __init__(self, img_channels: int = 3, fmap: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, fmap, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fmap, fmap * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fmap * 2, fmap * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fmap * 4, fmap * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(fmap * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(fmap * 8, 1, 4, 1, 0, bias=False),  # 4×4 → 1×1 logits
        )
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.view(x.size(0))

# GAN
class GAN:
    """High‑level wrapper bundling Generator, Discriminator, loss, and optimisers."""

    def __init__(self,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 loss_fn: Callable,
                 g_optimizer: torch.optim.Optimizer,
                 d_optimizer: torch.optim.Optimizer,
                 device: str | torch.device = "cuda"):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.loss_fn = loss_fn
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.device = torch.device(device)

    def _labels(self, preds: torch.Tensor, real: bool) -> torch.Tensor:
        return torch.ones_like(preds) if real else torch.zeros_like(preds)

    def train(self,
              dataloader: DataLoader,
              epochs: int = 100,
              noise_dim: int = 100,
              log_every: int = 100,
              schedulers: Optional[Dict[str, Any]] = None):
        self.G.train()
        self.D.train()
        for epoch in range(1, epochs + 1):
            for step, (real, _) in enumerate(dataloader, start=1):
                real = real.to(self.device)
                bs = real.size(0)

                # 1) Discriminator update
                self.d_optimizer.zero_grad(set_to_none=True)
                z = torch.randn(bs, noise_dim, device=self.device)
                fake = self.G(z).detach()
                d_loss = (
                    self.loss_fn(self.D(real), self._labels(self.D(real), True)) +
                    self.loss_fn(self.D(fake), self._labels(self.D(fake), False))
                )
                d_loss.backward()
                self.d_optimizer.step()

                # 2) Generator update
                self.g_optimizer.zero_grad(set_to_none=True)
                z = torch.randn(bs, noise_dim, device=self.device)
                g_loss = self.loss_fn(self.D(self.G(z)), self._labels(self.D(self.G(z)), True))
                g_loss.backward()
                self.g_optimizer.step()

                # LR scheduler hooks
                if schedulers:
                    schedulers.get("D", lambda: None)()
                    schedulers.get("G", lambda: None)()

                if step % log_every == 0:
                    print(f"[Epoch {epoch}/{epochs}] "
                          f"Step {step}/{len(dataloader)} "
                          f"D_loss: {d_loss.item():.4f} | G_loss")