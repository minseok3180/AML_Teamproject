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
import loss
import fid 


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
    # FFHQ256 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize(256),  # 256x256 크기 유지
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
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.BCELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 노이즈 차원 (Generator 입력 크기에 맞춰 조정)
    nz = 100  # latent vector size
    
    # 고정된 노이즈 벡터 (학습 진행 상황 모니터링용)
    fixed_noise = torch.randn(16, nz, 1, 1, device=device)  # 256x256는 큰 이미지이므로 16개로 줄임
    
    # 레이블 생성
    real_label = 1.0
    fake_label = 0.0
    
    # 학습 기록
    G_losses = []
    D_losses = []
    
    # 결과 저장 디렉토리 생성
    os.makedirs('./results', exist_ok=True)
    
    print("학습을 시작합니다...")
    
    for epoch in range(epochs):
        epoch_D_loss = 0.0
        epoch_G_loss = 0.0
        
        # 진행률 표시바
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for i, real_images in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            ############################
            # (1) Discriminator 업데이트
            ############################
            discriminator.zero_grad()
            
            # 실제 이미지에 대한 학습
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = discriminator(real_images).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            
            # 가짜 이미지 생성
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = generator(noise)
            
            # 가짜 이미지에 대한 학습
            label.fill_(fake_label)
            output = discriminator(fake_images.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Discriminator 파라미터 업데이트
            errD = errD_real + errD_fake
            optimizer_D.step()
            
            ############################
            # (2) Generator 업데이트
            ############################
            generator.zero_grad()
            
            label.fill_(real_label)
            output = discriminator(fake_images).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Generator 파라미터 업데이트
            optimizer_G.step()
            
            # 손실 기록
            epoch_D_loss += errD.item()
            epoch_G_loss += errG.item()
            
            # 진행률 표시바 업데이트
            pbar.set_postfix({
                'D_loss': f'{errD.item():.4f}',
                'G_loss': f'{errG.item():.4f}',
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z1:.4f}/{D_G_z2:.4f}'
            })
        
        # 에폭별 평균 손실 계산
        avg_D_loss = epoch_D_loss / len(dataloader)
        avg_G_loss = epoch_G_loss / len(dataloader)
        
        G_losses.append(avg_G_loss)
        D_losses.append(avg_D_loss)
        
        # 주기적으로 생성된 이미지 저장
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
                
                # 이미지 저장
                plt.figure(figsize=(12, 12))
                plt.axis("off")
                plt.title(f"Generated FFHQ Images - Epoch {epoch+1}")
                
                # 4x4 격자로 16개 이미지 표시
                for idx in range(16):
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(fake_images[idx].permute(1, 2, 0) * 0.5 + 0.5)
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(f'./results/generated_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # 모델 체크포인트 저장
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
    
    # 학습 완료 후 손실 그래프 저장
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
    
    # 데이터 로딩
    print("FFHQ256 데이터를 로딩합니다...")
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"
    dataloader = load_data(batch_size, img_dir)
    print(f"데이터셋 크기: {len(dataloader.dataset)}")

    # 모델 생성
    print("모델을 생성합니다...")
    G = Generator().to(device)
    D = Discriminator().to(device)
    
    # 모델 파라미터 수 출력
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Generator 파라미터 수: {count_parameters(G):,}")
    print(f"Discriminator 파라미터 수: {count_parameters(D):,}")

    # 학습 시작
    lr = 0.0002
    train(G, D, dataloader, epochs, lr, device)