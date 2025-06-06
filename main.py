# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from util import ResidualBlock
from model import Generator, Discriminator
from loss import discriminator_rploss, generator_rploss


class FFHQDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
    

def load_data(batch_size: int, img_dir: str) -> DataLoader:
    """
    FFHQ256 데이터셋 로더 반환
    Args:
        batch_size (int): 배치 크기 
        img_dir (str): 이미지 데이터 경로
    Returns:
        DataLoader: 학습용 데이터로더
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] 범위로 정규화
    ])
    
    dataset = FFHQDataset(img_dir, transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    
    return dataloader


def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    r1_lambda: float,
    r2_lambda: float,
    device: torch.device):
    """
    학습 루프 실행
    Args:
        generator (nn.Module): Generator 모델
        discriminator (nn.Module): Discriminator 모델
        dataloader (DataLoader): 데이터로더
        epochs (int): 에폭 수
        lr (float): 학습률 
        device (torch.device): 학습 디바이스
    """
    # 옵티마이저 설정
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))  # 깃헙 코드에서는 (0,0)으로 설정함. 
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    nz = generator.noise_dim
    fixed_noise = torch.randn(16, nz, device=device) # 50 에폭마다 같은 노이즈를 Generator에 넣어서 생성된 이미지를 비교
    
    G_losses = [] # epoch마다 평균 Generator loss 기록 
    D_losses = [] # epoch마다 평균 Discriminator loss 기록 
    os.makedirs('./results', exist_ok=True)
    
    print("학습을 시작합니다...")
    for epoch in range(epochs):
        epoch_D_loss = 0.0 # 한 epoch의 Discriminator loss
        epoch_G_loss = 0.0 # 한 epoch의 Generator loss 
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, real_images in enumerate(pbar): # i는 몇 번째 배치인지를 나타낸다. 
            batch_size = real_images.size(0) # 현재 배치에 들어 있는 이미지 개수 
            real_images = real_images.to(device)
            
            #######################
            # Discriminator 업데이트
            #######################

            # gradient 초기화 
            discriminator.zero_grad()
            
            # 노이즈로 fake images 생성
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)
            
            # RpGAN Discriminator loss + R1 + R2
            d_loss = discriminator_rploss(
                discriminator,
                real_images,
                fake_images,
                r1_lambda=r1_lambda,
                r2_lambda=r2_lambda
            )
            d_loss.backward() # 역전파 
            optimizer_D.step() # 파라미터 업데이트 
            

            ###################
            # Generator 업데이트
            ###################

            # gradient 초기화 
            generator.zero_grad()
            
            # 노이즈로 fake images 생성
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)
            
            # RpGAN Generator loss 
            g_loss = generator_rploss(
                discriminator,
                real_images,
                fake_images
            )
            g_loss.backward() # 역전파 
            optimizer_G.step() # 파라미터 업데이트 
            
            # 손실 기록
            epoch_D_loss += d_loss.item()
            epoch_G_loss += g_loss.item()
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
        
        avg_D_loss = epoch_D_loss / len(dataloader) # 한 에폭에서 평균 Discriminator loss 
        avg_G_loss = epoch_G_loss / len(dataloader) # 한 에폭에서 평균 Generator loss 
        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)
        
        # 주기적으로 생성 이미지 저장
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise).detach().cpu()
                plt.figure(figsize=(12, 12))
                plt.axis("off")
                plt.title(f"Generated FFHQ Images - Epoch {epoch+1}")
                for idx in range(16):
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(fake_samples[idx].permute(1, 2, 0) * 0.5 + 0.5)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'./results/generated_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # 주기적으로 체크포인트 저장
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, f'./results/checkpoint_epoch_{epoch+1}.pth')
        
        print(f'Epoch [{epoch+1}/{epochs}] - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')
    
    # 학습 완료 후 loss 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./results/training_loss.png')
    plt.close()
    
    print("학습이 완료되었습니다!")


if __name__ == "__main__":
    batch_size = 16
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("FFHQ256 데이터를 로딩합니다...")
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"
    dataloader = load_data(batch_size, img_dir)
    print(f"데이터셋 크기: {len(dataloader.dataset)}")

    print("모델을 생성합니다...")
    G = Generator().to(device)
    D = Discriminator().to(device)

    G.train()
    D.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Generator 파라미터 수: {count_parameters(G):,}")
    print(f"Discriminator 파라미터 수: {count_parameters(D):,}")

    lr = 0.0002
    r1_lambda = 10.0
    r2_lambda = 10.0
    train(G, D, dataloader, epochs, lr, r1_lambda, r2_lambda, device)