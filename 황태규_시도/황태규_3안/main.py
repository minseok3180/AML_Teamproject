# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from model import Generator, Discriminator
from pytorch_fid import fid_score   ###FID 때문에 새로 설치한 것


class FFHQDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.image_files = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith(".png")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.image_files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def load_data(batch_size: int, img_dir: str, max_images: int = None) -> DataLoader:
    """
    FFHQ256 데이터셋 로더 반환 (앞 max_images개만 사용)
    Args:
        batch_size (int): 배치 크기
        img_dir (str): 이미지 데이터 폴더 경로
        max_images (int, optional): 사용할 최대 이미지 수 (None일 경우 전체)
    """
    
    # FFHQ256 이미지 전처리 파이프라인
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])
    full_dataset = FFHQDataset(img_dir, transform)

    if max_images is not None and max_images < len(full_dataset):
        indices = list(range(max_images))
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,      
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )


def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    fid_batch_size: int = 32,
    fid_num_images: int = 1000,
    fid_every: int = 5
):
    """
    학습 루프 실행 + 주기적으로 FID 계산
    Args:
        generator (nn.Module): Generator 모델
        discriminator (nn.Module): Discriminator 모델
        dataloader (DataLoader): 학습용 데이터로더 (Subset 포함, max_images 제한됨)
        epochs (int): 에폭 수
        lr (float): 학습률
        device (torch.device): 학습 디바이스
        ###fid_batch_size (int): FID 계산 시 한 번에 생성/실제 이미지 몇 개씩 처리할지
        ###fid_num_images (int): FID 계산에 사용할 real/fake 이미지 개수 (예: 1000)
        ###fid_every (int): 몇 에폭마다 FID를 계산할지 (예: 5)
    """

    ### 손실 함수 및 옵티마이저 설정
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(0.5, 0.999))

    ### 노이즈 차원 (Generator 입력 크기에 맞춰 조정)
    nz = generator.noise_dim
    real_dataset = dataloader.dataset
    num_real_images = len(real_dataset)   #max_images와 동일(10000)

    ###FID 계산에 쓸 real 이미지 인덱스(0번부터 몇 번까지) 설정
    fid_real_indices = list(range(min(fid_num_images, num_real_images))) 
    ###위 인덱스만 뽑아 새로 만든 real 이미지 전용 데이터셋
    fid_real_subset = Subset(real_dataset, fid_real_indices) 
    ### 고정된 노이즈 벡터 (학습 진행 상황 모니터링용)
    fixed_fid_noise = torch.randn(len(fid_real_indices), nz, device=device)  

    ### 레이블 생성
    real_label_val = 0.9 ###발산 억제 위해 1->0.9로 수정
    fake_label_val = 0.0

    # 학습 기록
    G_losses, D_losses = [], []
    
    # 결과 저장 디렉토리 생성
    os.makedirs("./results", exist_ok=True)

    print("학습을 시작합니다...")
    for epoch in range(epochs):
        epoch_D_loss, epoch_G_loss = 0.0, 0.0
        # 진행률 표시바
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        # 학습 루프
        for real_imgs in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # (1) Discriminator 업데이트
            discriminator.zero_grad()
            ### 실제 이미지에 대한 학습
            labels_real = torch.full((batch_size,), real_label_val, device=device)
            out_real = discriminator(real_imgs).view(-1)
            errD_real = criterion(out_real, labels_real)
            errD_real.backward()
            ### sigmoid 함수를 통해 raw logit 값을 확률로 변환
            D_x = torch.sigmoid(out_real).mean().item()

            # 가짜 이미지 생성
            noise = torch.randn(batch_size, nz, device=device)
            fake_imgs = generator(noise)
            
            ### 가짜 이미지에 대한 학습
            ###기존 코드(label.fill)는 in-place 연산이라 부작용 생길 수 있어서 이렇게 수정
            labels_fake = torch.full((batch_size,), fake_label_val, device=device)
            out_fake = discriminator(fake_imgs.detach()).view(-1)
            errD_fake = criterion(out_fake, labels_fake)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(out_fake).mean().item()

            # Discriminator 파라미터 업데이트
            optimizer_D.step()
            
            ### (2) Generator 업데이트
            generator.zero_grad()
            
            ###기존 코드(label.fill)는 in-place 연산이라 부작용 생길 수 있어서 이렇게 수정
            out_fake_for_G = discriminator(fake_imgs).view(-1)
            errG = criterion(out_fake_for_G, labels_real)
            errG.backward()
            optimizer_G.step()
            D_G_z2 = torch.sigmoid(out_fake_for_G).mean().item()
            
            # 손실 기록
            epoch_D_loss += (errD_real + errD_fake).item()
            epoch_G_loss += errG.item()

            # 진행률 표시바 업데이트
            pbar.set_postfix({
                "D_loss": f"{(errD_real+errD_fake).item():.4f}",
                "G_loss": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.4f}",
                "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}"
            })

        ### 한 에폭 학습 다 끝난 뒤 GPU 메모리 정리
        torch.cuda.empty_cache()

        # 에폭별 평균 손실 계산
        avg_D = epoch_D_loss / len(dataloader)
        avg_G = epoch_G_loss / len(dataloader)
        
        G_losses.append(avg_G)
        D_losses.append(avg_D)

        # 매 에폭마다 샘플 생성 시 "16개만 미리보기" 저장 (샘플링은 적게)
        with torch.no_grad():
            preview_noise = torch.randn(16, nz, device=device)
            preview_images = generator(preview_noise).cpu()  # (16,3,256,256)
        os.makedirs("./results/samples", exist_ok=True)
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        for idx in range(16):
            img = preview_images[idx]
            img_01 = (img + 1) / 2.0
            plt.subplot(4, 4, idx + 1)
            plt.imshow(img_01.permute(1, 2, 0))
            plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"./results/samples/gen_epoch_{epoch+1}.png")
        plt.close()

        # 모델 체크포인트 저장
        torch.save({
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
        }, f"./results/ckpt_epoch_{epoch+1}.pth")

        ### FID 계산 주기인지 확인 
        if (epoch + 1) % fid_every == 0:
            #FID 생성 직전 GPU 메모리 정리
            torch.cuda.empty_cache()

            #안정적으로 결과 얻기 위해 eval()로 변경
            generator.eval()
            discriminator.eval()

            #이미지 폴더 생성
            fid_root = "./results/fid"
            real_dir = os.path.join(fid_root, "real")
            fake_dir = os.path.join(fid_root, "fake")

            if os.path.exists(fid_root):
                shutil.rmtree(fid_root)
            os.makedirs(real_dir)
            os.makedirs(fake_dir)

            # (1) Real 이미지 저장 (fid_num_images장)
            # shuffle=False 이기 때문에 epoch마다 일관성 있게 FID 계산 가능
            real_loader_for_fid = DataLoader(
                fid_real_subset,
                batch_size=fid_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
            # 이미지(-1,1)를 (0,1)로 변환 후 저장
            real_idx = 0
            for real_batch in real_loader_for_fid:
                for img in real_batch:
                    img_01 = (img + 1) / 2.0
                    pil_img = TF.to_pil_image(img_01)
                    pil_img.save(os.path.join(real_dir, f"real_{real_idx:05d}.png"))
                    real_idx += 1

            # (2) Fake 이미지 생성 후 저장
            fake_idx = 0
            with torch.no_grad():
                for start in range(0, len(fid_real_indices), fid_batch_size):
                    end = min(start + fid_batch_size, len(fid_real_indices))
                    noise_batch = fixed_fid_noise[start:end]
                    #fake image mini batch를 만든 후 PIL로 저장하기 위해 CPU로 이동
                    fake_batch = generator(noise_batch).cpu()

                    # 이미지(-1,1)를 (0,1)로 변환 후 저장
                    for img in fake_batch:
                        img_01 = (img + 1) / 2.0
                        pil_img = TF.to_pil_image(img_01)
                        pil_img.save(os.path.join(fake_dir, f"fake_{fake_idx:05d}.png"))
                        fake_idx += 1

                    del fake_batch, noise_batch
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            #FID 계산
            paths = [real_dir, fake_dir]
            fid_value = fid_score.calculate_fid_given_paths(
                paths, batch_size=fid_batch_size, device=device, dims=2048
            )
            print(f"Epoch {epoch+1:03d} | FID: {fid_value:.4f}")

            # real/fake 이미지 디렉토리는 FID 계산 후 삭제
            shutil.rmtree(real_dir)
            shutil.rmtree(fake_dir)

            #eval() 모드였던 것을 다시 train() 모드로 복원
            generator.train()
            discriminator.train()

        print(f"Epoch [{epoch+1}/{epochs}] D_loss: {avg_D:.4f}, G_loss: {avg_G:.4f}")

    # 학습 완료 후 손실 그래프 저장
    torch.cuda.empty_cache()  # 학습 종료 시에도 메모리 정리
    plt.figure(figsize=(6, 4))
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./results/training_loss.png")
    plt.close()

    print("학습이 완료되었습니다!")


if __name__ == "__main__":
    # 배치 크기와 사용할 이미지 수(max_images) 조정
    batch_size = 16       # 원하는 배치 크기 (예: 4, 8, 16 등)
    max_images = 10000   # 앞의 10000개만 사용
    epochs = 100         # 전체 에폭 수
    lr = 0.00005     # 학습률

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"

    # 데이터 로딩
    print("데이터 로딩 중...")
    dataloader = load_data(batch_size, img_dir, max_images=max_images)
    print(f"데이터셋 (앞 {max_images}개) 크기: {len(dataloader.dataset)}")

    # 모델 생성
    print("모델 생성 중...")
    G = Generator().to(device)
    D = Discriminator().to(device)

    # 모델 파라미터 수 출력
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Generator 파라미터: {count_params(G):,}")
    print(f"Discriminator 파라미터: {count_params(D):,}")

    # 학습 시작
    train(
        G, D, dataloader, epochs, lr, device,
        fid_batch_size=16,
        fid_num_images=1000,
        fid_every=1
    )
