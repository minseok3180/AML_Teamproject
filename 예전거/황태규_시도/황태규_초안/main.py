# main.py

import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from util import ResidualBlock
from model import Generator, Discriminator
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class FlatImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0  # label은 dummy 값

def load_data(batch_size: int, image_dir) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    dataset = FlatImageDataset(image_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return loader


def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader,
    epochs: int,
    device: torch.device,
    lr: float = 2e-4,
    out_dir: str = '/home/elicer/AML_Teamproject/황태규_시도/samples'
):
    os.makedirs(out_dir, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    generator.train()
    discriminator.train()

    fixed_noise = torch.randn(16, 3, 256, 256, device=device)

    D_losses, G_losses = [], []
    print("train 진입, dataloader 길이:", len(dataloader))
    print("데이터 한 배치 shape 확인:", next(iter(dataloader))[0].shape)
    for x in dataloader:
        print("데이터 있음!")
        break
    print("==== Start Training ====")
    for epoch in range(epochs):
        print("step 진입")
        for i, (real_imgs, _) in enumerate(dataloader):
            print(f"step {i} 배치 shape: {real_imgs.shape}")
            if i >= 10:  # 20 step만 돌고 바로 다음 epoch                 #제대로 결과 내려면 이 if문을 제거
                break
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # 1. Train Discriminator
            optimizer_D.zero_grad()
            real_labels = torch.ones(batch_size, 1, 16, 16, device=device)
            fake_labels = torch.zeros(batch_size, 1, 16, 16, device=device)

            out_real = discriminator(real_imgs)
            loss_D_real = criterion(out_real, real_labels)

            noise = torch.randn(batch_size, 3, 256, 256, device=device)
            fake_imgs = generator(noise)
            out_fake = discriminator(fake_imgs.detach())
            loss_D_fake = criterion(out_fake, fake_labels)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # 2. Train Generator
            optimizer_G.zero_grad()
            out_fake = discriminator(fake_imgs)
            loss_G = criterion(out_fake, real_labels)
            loss_G.backward()
            optimizer_G.step()

            D_losses.append(loss_D.item())
            G_losses.append(loss_G.item())

            # ---- 실시간 Loss 숫자 출력 ----
            if (i+1) % 1 == 0 or (i+1) == len(dataloader):
                recent_D = np.mean(D_losses[-20:]) if len(D_losses) >= 20 else np.mean(D_losses)
                recent_G = np.mean(G_losses[-20:]) if len(G_losses) >= 20 else np.mean(G_losses)
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Step [{i+1}/{len(dataloader)}] "
                    f"D Loss (최근20평균): {recent_D:.4f} "
                    f"G Loss (최근20평균): {recent_G:.4f}"
                )

        # 에폭마다 샘플 이미지 저장
        with torch.no_grad():
            sample_imgs = generator(fixed_noise)
            sample_imgs = (sample_imgs + 1) / 2
            vutils.save_image(sample_imgs, f"{out_dir}/epoch_{epoch+1:03d}.png", nrow=4)

    print("==== Training finished ====")


if __name__ == "__main__":
    
    # 하이퍼파라미터 설정
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 데이터 로딩
    image_dir = "/home/elicer/AML_Teamproject/ffhq256"
    dataloader = load_data(batch_size, image_dir)

    # 모델 생성
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 학습 시작
    train(G, D, dataloader, epochs, device)
   
    
