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
from pytorch_fid import fid_score   ###FID ?λ¬Έμ ?λ‘? ?€μΉν κ²?


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
    FFHQ256 ?°?΄?°? λ‘λ λ°ν (? max_imagesκ°λ§ ?¬?©)
    Args:
        batch_size (int): λ°°μΉ ?¬κΈ?
        img_dir (str): ?΄λ―Έμ?? ?°?΄?° ?΄? κ²½λ‘
        max_images (int, optional): ?¬?©?  μ΅λ?? ?΄λ―Έμ?? ? (None?Ό κ²½μ° ? μ²?)
    """
    
    # FFHQ256 ?΄λ―Έμ?? ? μ²λ¦¬ ??΄??Ό?Έ
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
    ??΅ λ£¨ν ?€? + μ£ΌκΈ°? ?Όλ‘? FID κ³μ°
    Args:
        generator (nn.Module): Generator λͺ¨λΈ
        discriminator (nn.Module): Discriminator λͺ¨λΈ
        dataloader (DataLoader): ??΅?© ?°?΄?°λ‘λ (Subset ?¬?¨, max_images ? ??¨)
        epochs (int): ??­ ?
        lr (float): ??΅λ₯?
        device (torch.device): ??΅ ?λ°μ΄?€
        ###fid_batch_size (int): FID κ³μ° ? ? λ²μ ??±/?€?  ?΄λ―Έμ?? λͺ? κ°μ© μ²λ¦¬? μ§?
        ###fid_num_images (int): FID κ³μ°? ?¬?©?  real/fake ?΄λ―Έμ?? κ°μ (?: 1000)
        ###fid_every (int): λͺ? ??­λ§λ€ FIDλ₯? κ³μ°? μ§? (?: 5)
    """

    ### ??€ ?¨? λ°? ?΅?°λ§μ΄??? ?€? 
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(0.5, 0.999))

    ### ?Έ?΄μ¦? μ°¨μ (Generator ?? ₯ ?¬κΈ°μ λ§μΆ° μ‘°μ )
    nz = generator.noise_dim
    real_dataset = dataloader.dataset
    num_real_images = len(real_dataset)   #max_images??? ??Ό(10000)


    # fid
    fid_real_indices = list(range(min(fid_num_images, num_real_images))) 
    fid_real_subset = Subset(real_dataset, fid_real_indices) 
    fixed_fid_noise = torch.randn(len(fid_real_indices), nz, device=device)  

    ### ? ?΄λΈ? ??±
    real_label_val = 0.9 ###λ°μ° ?΅?  ??΄ 1->0.9λ‘? ?? 
    fake_label_val = 0.0

    # ??΅ κΈ°λ‘
    G_losses, D_losses = [], []
    
    # κ²°κ³Ό ????₯ ?? ? λ¦? ??±
    os.makedirs("./results", exist_ok=True)

    print("??΅? ???©??€...")
    for epoch in range(epochs):
        epoch_D_loss, epoch_G_loss = 0.0, 0.0
        # μ§νλ₯? ??λ°?
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        # ??΅ λ£¨ν
        for real_imgs in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # (1) Discriminator ??°?΄?Έ
            discriminator.zero_grad()
            ### ?€?  ?΄λ―Έμ??? ???? ??΅
            labels_real = torch.full((batch_size,), real_label_val, device=device)
            out_real = discriminator(real_imgs).view(-1)
            errD_real = criterion(out_real, labels_real)
            errD_real.backward()
            ### sigmoid ?¨?λ₯? ?΅?΄ raw logit κ°μ ?λ₯ λ‘ λ³??
            D_x = torch.sigmoid(out_real).mean().item()

            # κ°?μ§? ?΄λ―Έμ?? ??±
            noise = torch.randn(batch_size, nz, device=device)
            fake_imgs = generator(noise)
            
            ### κ°?μ§? ?΄λ―Έμ??? ???? ??΅
            ###κΈ°μ‘΄ μ½λ(label.fill)? in-place ?°?°?΄?Ό λΆ???© ?κΈ? ? ??΄? ?΄? κ²? ?? 
            labels_fake = torch.full((batch_size,), fake_label_val, device=device)
            out_fake = discriminator(fake_imgs.detach()).view(-1)
            errD_fake = criterion(out_fake, labels_fake)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(out_fake).mean().item()

            # Discriminator ??Όλ―Έν° ??°?΄?Έ
            optimizer_D.step()
            
            ### (2) Generator ??°?΄?Έ
            generator.zero_grad()
            
            ###κΈ°μ‘΄ μ½λ(label.fill)? in-place ?°?°?΄?Ό λΆ???© ?κΈ? ? ??΄? ?΄? κ²? ?? 
            out_fake_for_G = discriminator(fake_imgs).view(-1)
            errG = criterion(out_fake_for_G, labels_real)
            errG.backward()
            optimizer_G.step()
            D_G_z2 = torch.sigmoid(out_fake_for_G).mean().item()
            
            # ??€ κΈ°λ‘
            epoch_D_loss += (errD_real + errD_fake).item()
            epoch_G_loss += errG.item()

            # μ§νλ₯? ??λ°? ??°?΄?Έ
            pbar.set_postfix({
                "D_loss": f"{(errD_real+errD_fake).item():.4f}",
                "G_loss": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.4f}",
                "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}"
            })

        ### ? ??­ ??΅ ?€ ?? ?€ GPU λ©λͺ¨λ¦? ? λ¦?
        torch.cuda.empty_cache()

        # ??­λ³? ?κ·? ??€ κ³μ°
        avg_D = epoch_D_loss / len(dataloader)
        avg_G = epoch_G_loss / len(dataloader)
        
        G_losses.append(avg_G)
        D_losses.append(avg_D)

        # λ§? ??­λ§λ€ ?? ??± ? "16κ°λ§ λ―Έλ¦¬λ³΄κΈ°" ????₯ (??λ§μ?? ? κ²?)
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

        # λͺ¨λΈ μ²΄ν¬?¬?Έ?Έ ????₯
        torch.save({
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
        }, f"./results/ckpt_epoch_{epoch+1}.pth")

        ### FID κ³μ° μ£ΌκΈ°?Έμ§? ??Έ 
        if (epoch + 1) % fid_every == 0:
            #FID ??± μ§μ  GPU λ©λͺ¨λ¦? ? λ¦?
            torch.cuda.empty_cache()

            #?? ? ?Όλ‘? κ²°κ³Ό ?»κΈ? ??΄ eval()λ‘? λ³?κ²?
            generator.eval()
            discriminator.eval()

            #?΄λ―Έμ?? ?΄? ??±
            fid_root = "./results/fid"
            real_dir = os.path.join(fid_root, "real")
            fake_dir = os.path.join(fid_root, "fake")

            if os.path.exists(fid_root):
                shutil.rmtree(fid_root)
            os.makedirs(real_dir)
            os.makedirs(fake_dir)

            # (1) Real ?΄λ―Έμ?? ????₯ (fid_num_images?₯)
            # shuffle=False ?΄κΈ? ?λ¬Έμ epochλ§λ€ ?Όκ΄??± ?κ²? FID κ³μ° κ°??₯
            real_loader_for_fid = DataLoader(
                fid_real_subset,
                batch_size=fid_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
            # ?΄λ―Έμ??(-1,1)λ₯? (0,1)λ‘? λ³?? ? ????₯
            real_idx = 0
            for real_batch in real_loader_for_fid:
                for img in real_batch:
                    img_01 = (img + 1) / 2.0
                    pil_img = TF.to_pil_image(img_01)
                    pil_img.save(os.path.join(real_dir, f"real_{real_idx:05d}.png"))
                    real_idx += 1

            # (2) Fake ?΄λ―Έμ?? ??± ? ????₯
            fake_idx = 0
            with torch.no_grad():
                for start in range(0, len(fid_real_indices), fid_batch_size):
                    end = min(start + fid_batch_size, len(fid_real_indices))
                    noise_batch = fixed_fid_noise[start:end]
                    #fake image mini batchλ₯? λ§λ  ? PILλ‘? ????₯?κΈ? ??΄ CPUλ‘? ?΄?
                    fake_batch = generator(noise_batch).cpu()

                    # ?΄λ―Έμ??(-1,1)λ₯? (0,1)λ‘? λ³?? ? ????₯
                    for img in fake_batch:
                        img_01 = (img + 1) / 2.0
                        pil_img = TF.to_pil_image(img_01)
                        pil_img.save(os.path.join(fake_dir, f"fake_{fake_idx:05d}.png"))
                        fake_idx += 1

                    del fake_batch, noise_batch
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            #FID κ³μ°
            paths = [real_dir, fake_dir]
            fid_value = fid_score.calculate_fid_given_paths(
                paths, batch_size=fid_batch_size, device=device, dims=2048
            )
            print(f"Epoch {epoch+1:03d} | FID: {fid_value:.4f}")

            # real/fake ?΄λ―Έμ?? ?? ? λ¦¬λ FID κ³μ° ? ?­? 
            shutil.rmtree(real_dir)
            shutil.rmtree(fake_dir)

            #eval() λͺ¨λ???? κ²μ ?€? train() λͺ¨λλ‘? λ³΅μ
            generator.train()
            discriminator.train()

        print(f"Epoch [{epoch+1}/{epochs}] D_loss: {avg_D:.4f}, G_loss: {avg_G:.4f}")

    # ??΅ ?λ£? ? ??€ κ·Έλ? ????₯
    torch.cuda.empty_cache()  # ??΅ μ’λ£ ??? λ©λͺ¨λ¦? ? λ¦?
    plt.figure(figsize=(6, 4))
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./results/training_loss.png")
    plt.close()

    print("??΅?΄ ?λ£λ??΅??€!")


if __name__ == "__main__":
    # λ°°μΉ ?¬κΈ°μ?? ?¬?©?  ?΄λ―Έμ?? ?(max_images) μ‘°μ 
    batch_size = 16       # ??? λ°°μΉ ?¬κΈ? (?: 4, 8, 16 ?±)
    max_images = 10000   # ?? 10000κ°λ§ ?¬?©
    epochs = 100         # ? μ²? ??­ ?
    lr = 0.00005     # ??΅λ₯?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"

    # ?°?΄?° λ‘λ©
    print("?°?΄?° λ‘λ© μ€?...")
    dataloader = load_data(batch_size, img_dir, max_images=max_images)
    print(f"?°?΄?°? (? {max_images}κ°?) ?¬κΈ?: {len(dataloader.dataset)}")

    # λͺ¨λΈ ??±
    print("λͺ¨λΈ ??± μ€?...")
    G = Generator().to(device)
    D = Discriminator().to(device)

    # λͺ¨λΈ ??Όλ―Έν° ? μΆλ ₯
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Generator ??Όλ―Έν°: {count_params(G):,}")
    print(f"Discriminator ??Όλ―Έν°: {count_params(D):,}")

    # ??΅ ??
    train(
        G, D, dataloader, epochs, lr, device,
        fid_batch_size=16,
        fid_num_images=1000,
        fid_every=1
    )
