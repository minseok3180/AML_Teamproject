import torch
from torch import autograd
import torch.nn as nn

def r1_penalty(discriminator, real_images):
    # 1) leaf tensor 분리 후 gradient 계산 허용
    real_images = real_images.clone().detach().requires_grad_(True)  # (B, C, H, W)

    # 2) discriminator 출력 얻고 스칼라로 변환
    real_logits = discriminator(real_images).view(-1)  # (B,)
    loss_real = real_logits.sum()

    # 3) gradient 계산
    grads = autograd.grad(
        outputs=loss_real,
        inputs=real_images,
        create_graph=True,
        retain_graph=True
    )[0]  # (B, C, H, W)

    # 4) penalty 계산
    grads = grads.view(grads.size(0), -1)                   # (B, C*H*W)
    grads_norm2 = torch.sum(grads ** 2, dim=1)            # (B,)
    penalty = torch.mean(grads_norm2)
    return penalty


def r2_penalty(discriminator, fake_images):
    # 1) leaf tensor 분리 후 gradient 계산 허용
    fake_images = fake_images.clone().detach().requires_grad_(True)  # (B, C, H, W)

    # 2) discriminator 출력 얻고 스칼라로 변환
    fake_logits = discriminator(fake_images).view(-1)  # (B,)
    loss_fake = fake_logits.sum()

    # 3) gradient 계산
    grads = autograd.grad(
        outputs=loss_fake,
        inputs=fake_images,
        create_graph=True,
        retain_graph=True
    )[0]  # (B, C, H, W)

    # 4) penalty 계산
    grads = grads.view(grads.size(0), -1)                   # (B, C*H*W)
    grads_norm2 = torch.sum(grads ** 2, dim=1)            # (B,)
    penalty = torch.mean(grads_norm2)
    return penalty


def discriminator_rploss(discriminator, real_images, fake_images, gamma):
    # 기본 RPLoss
    real_logits = discriminator(real_images).view(-1)               # (B,)
    fake_logits = discriminator(fake_images.detach()).view(-1)     # (B,)
    diff = real_logits - fake_logits
    d_rploss = nn.functional.softplus(-diff).mean()

    # gradient penalties
    penalty_r1 = r1_penalty(discriminator, real_images)
    penalty_r2 = r2_penalty(discriminator, fake_images)

    return d_rploss + (gamma / 2) * (penalty_r1 + penalty_r2)


def generator_rploss(discriminator, real_images, fake_images):
    real_logits = discriminator(real_images.detach()).view(-1)  # (B,)
    fake_logits = discriminator(fake_images).view(-1)           # (B,)
    diff = fake_logits - real_logits
    return nn.functional.softplus(-diff).mean()


def discriminator_hinge_rploss(discriminator, real_images, fake_images, gamma, margin: float = 1.0):
    real_logits = discriminator(real_images).view(-1)
    fake_logits = discriminator(fake_images.detach()).view(-1)
    diff = real_logits - fake_logits
    d_hinge = nn.functional.relu(margin - diff).mean()

    penalty_r1 = r1_penalty(discriminator, real_images)
    penalty_r2 = r2_penalty(discriminator, fake_images)

    return d_hinge + (gamma / 2) * (penalty_r1 + penalty_r2)


def generator_hinge_rploss(discriminator, real_images, fake_images, margin: float = 1.0):
    real_logits = discriminator(real_images.detach()).view(-1)
    fake_logits = discriminator(fake_images).view(-1)
    diff = fake_logits - real_logits
    return nn.functional.relu(margin - diff).mean()
