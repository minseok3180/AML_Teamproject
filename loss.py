import torch
from torch import autograd
import torch.nn as nn


def r1_penalty(discriminator, real_images, r1_lambda):
    # r1 penalty를 사용하지 않으면 0 반환 
    if r1_lambda <= 0.0:
        return torch.tensor(0.0, device=real_images.device)

    real_images.requires_grad_(True) # gradient 계산하도록. real images shape: (B, C, H, W)
    real_logits = discriminator(real_images).view(-1) # (B, 1) -> (B,)

    grads_real = autograd.grad(
        outputs=real_logits.sum(), # outputs 인자에는 스칼라가 와야 함. 각 샘플마다 Discriminator logit의 gradient를 계산. 
        inputs=real_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0] # inputs가 real_images 하나 -> 튜플의 첫 번째 요소를 grads에 저장. grads shape: (B, C, H, W)

    grads_real = grads_real.view(grads_real.size(0), -1)  # (B, C, H, W) -> (B, C*H*W)
    grads_real_norm2 = torch.sum(grads_real ** 2, dim=1)  # 텐서의 두 번째 차원을 따라 sum, (B, C*H*W) -> (B,)
    r1_penalty = 0.5 * r1_lambda * torch.mean(grads_real_norm2)

    real_images.requires_grad_(False)

    return r1_penalty


def r2_penalty(discriminator, fake_images, r2_lambda):
    # r2 penalty를 사용하지 않으면 0 반환 
    if r2_lambda <= 0.0:
        return torch.tensor(0.0, device=fake_images.device)

    fake_images.requires_grad_(True) # gradient 계산하도록. fake images shape: (B, C, H, W)
    fake_logits = discriminator(fake_images).view(-1) # (B, 1) -> (B,)

    grads_fake = autograd.grad(
        outputs=fake_logits.sum(), # outputs 인자에는 스칼라가 와야 함. 각 샘플마다 Discriminator logit의 gradient를 계산. 
        inputs=fake_images,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0] # inputs가 fake_images 하나 -> 튜플의 첫 번째 요소를 grads에 저장. grads shape: (B, C, H, W)

    grads_fake = grads_fake.view(grads_fake.size(0), -1)  # (B, C, H, W) -> (B, C*H*W)
    grads_fake_norm2 = torch.sum(grads_fake ** 2, dim=1)  # 텐서의 두 번째 차원을 따라 sum, (B, C*H*W) -> (B,)
    r2_penalty = 0.5 * r2_lambda * torch.mean(grads_fake_norm2)

    fake_images.requires_grad_(False)

    return r2_penalty


def discriminator_rploss(discriminator, real_images, fake_images, r1_lambda, r2_lambda):
    real_logits = discriminator(real_images).view(-1) # (B, 1) -> (B,)
    fake_logits = discriminator(fake_images.detach()).view(-1)  # (B, 1) -> (B,), detach()로 Generator로의 gradient 전파 차단

    diff_real_fake = real_logits - fake_logits # Discriminator 입장에서는 이 값을 크게 만들고 싶음. 
    d_loss = nn.functional.softplus(-diff_real_fake).mean()

    penalty_r1 = r1_penalty(discriminator, real_images, r1_lambda)
    penalty_r2 = r2_penalty(discriminator, fake_images, r2_lambda)

    final_d_loss = d_loss + penalty_r1 + penalty_r2

    return final_d_loss


def generator_rploss(discriminator, real_images, fake_images):
    real_logits = discriminator(real_images.detach()).view(-1)  # (B, 1) -> (B,), detach로 gradient 차단
    fake_logits = discriminator(fake_images).view(-1)  # (B, 1) -> (B,)

    diff_fake_real = fake_logits - real_logits # Generator 입장에서는 이 값을 크게 만들고 싶음.
    final_g_loss = nn.functional.softplus(-diff_fake_real).mean()
    
    return final_g_loss
