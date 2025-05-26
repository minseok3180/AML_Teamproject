from __future__ import annotations

import argparse
import os
import tqdm
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

from GAN import Generator, Discriminator  


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train basic GAN (Windows‑safe)")
    p.add_argument("--data_dir", type=pathlib.Path, default="ffhq256",
                   help="Path to image folder (ImageFolder layout)")
    p.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("images"),
                   help="Where to save generated sample grids")
    p.add_argument("--img_size", type=int, default=28)
    p.add_argument("--channels", type=int, default=1)
    p.add_argument("--latent_dim", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--b1", type=float, default=0.5)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--sample_interval", type=int, default=400)
    return p


def make_dataloader(root: pathlib.Path, img_size: int, batch_size: int, num_workers: int, channels: int):
    # Choose correct normalisation per channel count
    if channels == 3:
        mean = [0.5, 0.5, 0.5]; std = [0.5, 0.5, 0.5]
    else:
        mean = [0.5]; std = [0.5]

    ds = torchvision.datasets.ImageFolder(
        root=str(root),
        transform=transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

# main logic

def main():
    args = build_argparser().parse_args()

    # ---------------------------------------------------------------- data ----
    dataloader = make_dataloader(args.data_dir, args.img_size, args.batch_size,
                                 args.num_workers, args.channels)

    # ---------------------------------------------------------------- setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_shape = (args.channels, args.img_size, args.img_size)

    os.makedirs(args.out_dir, exist_ok=True)

    generator = Generator(); generator.make_model(args, img_shape*3)
    discriminator = Discriminator(); discriminator.make_model(img_shape*3)

    generator.to(device); discriminator.to(device)

    criterion = nn.BCELoss().to(device)
    opt_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    FloatTensor = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor


    bar = tqdm.tqdm(dataloader, # 진행 상황을 표시할 iterable 객체.
           desc = 'Description', # 진행 상황 표시줄의 제목.
           total = len(dataloader), # 전체 반복량 (확률에서의 분모).
           leave = True, # 잔상 남김 여부, 기본적으로 True
           ncols = 100, # 진행 바 길이. px 단위
           ascii = ' =', # 바 모양, 첫 번째 문자는 공백이어야 함
           initial = 0 # 진행 시작값
           )

    # ------------------------------------------------------------- training ---
    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(bar):
            real_imgs = imgs.to(device)
            batch = real_imgs.size(0)

            # ground‑truth labels
            valid = torch.ones(batch, 1, device=device)
            fake = torch.zeros(batch, 1, device=device)

            # ------------------------- train G -------------------------
            opt_G.zero_grad()
            z = torch.randn(batch, args.latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = criterion(discriminator(gen_imgs), valid)
            g_loss.backward(); opt_G.step()

            # ------------------------- train D -------------------------
            opt_D.zero_grad()
            real_loss = criterion(discriminator(real_imgs), valid)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) * 0.5
            d_loss.backward(); opt_D.step()

            # ---------------------- logging & save --------------------
            global_step = epoch * len(dataloader) + i
            if global_step % args.sample_interval == 0:
                grid = (gen_imgs[:25] + 1) / 2  # [-1,1] → [0,1]
                save_image(grid, args.out_dir / f"{global_step}.png", nrow=5)

            if i % 50 == 0:
                print(f"[E {epoch}/{args.n_epochs}] [B {i}/{len(dataloader)}] "
                      f"D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)  
    main()
