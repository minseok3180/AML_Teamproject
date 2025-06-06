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
from pytorch_fid import fid_score   ###FID ?•Œë¬¸ì— ?ƒˆë¡? ?„¤ì¹˜í•œ ê²?


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
    FFHQ256 ?°?´?„°?…‹ ë¡œë” ë°˜í™˜ (?• max_imagesê°œë§Œ ?‚¬?š©)
    Args:
        batch_size (int): ë°°ì¹˜ ?¬ê¸?
        img_dir (str): ?´ë¯¸ì?? ?°?´?„° ?´?” ê²½ë¡œ
        max_images (int, optional): ?‚¬?š©?•  ìµœë?? ?´ë¯¸ì?? ?ˆ˜ (None?¼ ê²½ìš° ? „ì²?)
    """
    
    # FFHQ256 ?´ë¯¸ì?? ? „ì²˜ë¦¬ ?ŒŒ?´?”„?¼?¸
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
    ?•™?Šµ ë£¨í”„ ?‹¤?–‰ + ì£¼ê¸°? ?œ¼ë¡? FID ê³„ì‚°
    Args:
        generator (nn.Module): Generator ëª¨ë¸
        discriminator (nn.Module): Discriminator ëª¨ë¸
        dataloader (DataLoader): ?•™?Šµ?š© ?°?´?„°ë¡œë” (Subset ?¬?•¨, max_images ? œ?•œ?¨)
        epochs (int): ?—?­ ?ˆ˜
        lr (float): ?•™?Šµë¥?
        device (torch.device): ?•™?Šµ ?””ë°”ì´?Š¤
        ###fid_batch_size (int): FID ê³„ì‚° ?‹œ ?•œ ë²ˆì— ?ƒ?„±/?‹¤? œ ?´ë¯¸ì?? ëª? ê°œì”© ì²˜ë¦¬?• ì§?
        ###fid_num_images (int): FID ê³„ì‚°?— ?‚¬?š©?•  real/fake ?´ë¯¸ì?? ê°œìˆ˜ (?˜ˆ: 1000)
        ###fid_every (int): ëª? ?—?­ë§ˆë‹¤ FIDë¥? ê³„ì‚°?• ì§? (?˜ˆ: 5)
    """

    ### ?†?‹¤ ?•¨?ˆ˜ ë°? ?˜µ?‹°ë§ˆì´??? ?„¤? •
    criterion = nn.BCEWithLogitsLoss()
    optimizer_G = optim.Adam(generator.parameters(),
                             lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=lr, betas=(0.5, 0.999))

    ### ?…¸?´ì¦? ì°¨ì› (Generator ?…? ¥ ?¬ê¸°ì— ë§ì¶° ì¡°ì •)
    nz = generator.noise_dim
    real_dataset = dataloader.dataset
    num_real_images = len(real_dataset)   #max_images??? ?™?¼(10000)


    # fid
    fid_real_indices = list(range(min(fid_num_images, num_real_images))) 
    fid_real_subset = Subset(real_dataset, fid_real_indices) 
    fixed_fid_noise = torch.randn(len(fid_real_indices), nz, device=device)  

    ### ? ˆ?´ë¸? ?ƒ?„±
    real_label_val = 0.9 ###ë°œì‚° ?–µ? œ ?œ„?•´ 1->0.9ë¡? ?ˆ˜? •
    fake_label_val = 0.0

    # ?•™?Šµ ê¸°ë¡
    G_losses, D_losses = [], []
    
    # ê²°ê³¼ ????¥ ?””? ‰?† ë¦? ?ƒ?„±
    os.makedirs("./results", exist_ok=True)

    print("?•™?Šµ?„ ?‹œ?‘?•©?‹ˆ?‹¤...")
    for epoch in range(epochs):
        epoch_D_loss, epoch_G_loss = 0.0, 0.0
        # ì§„í–‰ë¥? ?‘œ?‹œë°?
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        # ?•™?Šµ ë£¨í”„
        for real_imgs in pbar:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # (1) Discriminator ?—…?°?´?Š¸
            discriminator.zero_grad()
            ### ?‹¤? œ ?´ë¯¸ì???— ????•œ ?•™?Šµ
            labels_real = torch.full((batch_size,), real_label_val, device=device)
            out_real = discriminator(real_imgs).view(-1)
            errD_real = criterion(out_real, labels_real)
            errD_real.backward()
            ### sigmoid ?•¨?ˆ˜ë¥? ?†µ?•´ raw logit ê°’ì„ ?™•ë¥ ë¡œ ë³??™˜
            D_x = torch.sigmoid(out_real).mean().item()

            # ê°?ì§? ?´ë¯¸ì?? ?ƒ?„±
            noise = torch.randn(batch_size, nz, device=device)
            fake_imgs = generator(noise)
            
            ### ê°?ì§? ?´ë¯¸ì???— ????•œ ?•™?Šµ
            ###ê¸°ì¡´ ì½”ë“œ(label.fill)?Š” in-place ?—°?‚°?´?¼ ë¶??‘?š© ?ƒê¸? ?ˆ˜ ?ˆ?–´?„œ ?´? ‡ê²? ?ˆ˜? •
            labels_fake = torch.full((batch_size,), fake_label_val, device=device)
            out_fake = discriminator(fake_imgs.detach()).view(-1)
            errD_fake = criterion(out_fake, labels_fake)
            errD_fake.backward()
            D_G_z1 = torch.sigmoid(out_fake).mean().item()

            # Discriminator ?ŒŒ?¼ë¯¸í„° ?—…?°?´?Š¸
            optimizer_D.step()
            
            ### (2) Generator ?—…?°?´?Š¸
            generator.zero_grad()
            
            ###ê¸°ì¡´ ì½”ë“œ(label.fill)?Š” in-place ?—°?‚°?´?¼ ë¶??‘?š© ?ƒê¸? ?ˆ˜ ?ˆ?–´?„œ ?´? ‡ê²? ?ˆ˜? •
            out_fake_for_G = discriminator(fake_imgs).view(-1)
            errG = criterion(out_fake_for_G, labels_real)
            errG.backward()
            optimizer_G.step()
            D_G_z2 = torch.sigmoid(out_fake_for_G).mean().item()
            
            # ?†?‹¤ ê¸°ë¡
            epoch_D_loss += (errD_real + errD_fake).item()
            epoch_G_loss += errG.item()

            # ì§„í–‰ë¥? ?‘œ?‹œë°? ?—…?°?´?Š¸
            pbar.set_postfix({
                "D_loss": f"{(errD_real+errD_fake).item():.4f}",
                "G_loss": f"{errG.item():.4f}",
                "D(x)": f"{D_x:.4f}",
                "D(G(z))": f"{D_G_z1:.4f}/{D_G_z2:.4f}"
            })

        ### ?•œ ?—?­ ?•™?Šµ ?‹¤ ??‚œ ?’¤ GPU ë©”ëª¨ë¦? ? •ë¦?
        torch.cuda.empty_cache()

        # ?—?­ë³? ?‰ê·? ?†?‹¤ ê³„ì‚°
        avg_D = epoch_D_loss / len(dataloader)
        avg_G = epoch_G_loss / len(dataloader)
        
        G_losses.append(avg_G)
        D_losses.append(avg_D)

        # ë§? ?—?­ë§ˆë‹¤ ?ƒ˜?”Œ ?ƒ?„± ?‹œ "16ê°œë§Œ ë¯¸ë¦¬ë³´ê¸°" ????¥ (?ƒ˜?”Œë§ì?? ? ê²?)
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

        # ëª¨ë¸ ì²´í¬?¬?¸?Š¸ ????¥
        torch.save({
            "epoch": epoch + 1,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'G_losses': G_losses,
            'D_losses': D_losses,
        }, f"./results/ckpt_epoch_{epoch+1}.pth")

        ### FID ê³„ì‚° ì£¼ê¸°?¸ì§? ?™•?¸ 
        if (epoch + 1) % fid_every == 0:
            #FID ?ƒ?„± ì§ì „ GPU ë©”ëª¨ë¦? ? •ë¦?
            torch.cuda.empty_cache()

            #?•ˆ? •? ?œ¼ë¡? ê²°ê³¼ ?–»ê¸? ?œ„?•´ eval()ë¡? ë³?ê²?
            generator.eval()
            discriminator.eval()

            #?´ë¯¸ì?? ?´?” ?ƒ?„±
            fid_root = "./results/fid"
            real_dir = os.path.join(fid_root, "real")
            fake_dir = os.path.join(fid_root, "fake")

            if os.path.exists(fid_root):
                shutil.rmtree(fid_root)
            os.makedirs(real_dir)
            os.makedirs(fake_dir)

            # (1) Real ?´ë¯¸ì?? ????¥ (fid_num_images?¥)
            # shuffle=False ?´ê¸? ?•Œë¬¸ì— epochë§ˆë‹¤ ?¼ê´??„± ?ˆê²? FID ê³„ì‚° ê°??Š¥
            real_loader_for_fid = DataLoader(
                fid_real_subset,
                batch_size=fid_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
            # ?´ë¯¸ì??(-1,1)ë¥? (0,1)ë¡? ë³??™˜ ?›„ ????¥
            real_idx = 0
            for real_batch in real_loader_for_fid:
                for img in real_batch:
                    img_01 = (img + 1) / 2.0
                    pil_img = TF.to_pil_image(img_01)
                    pil_img.save(os.path.join(real_dir, f"real_{real_idx:05d}.png"))
                    real_idx += 1

            # (2) Fake ?´ë¯¸ì?? ?ƒ?„± ?›„ ????¥
            fake_idx = 0
            with torch.no_grad():
                for start in range(0, len(fid_real_indices), fid_batch_size):
                    end = min(start + fid_batch_size, len(fid_real_indices))
                    noise_batch = fixed_fid_noise[start:end]
                    #fake image mini batchë¥? ë§Œë“  ?›„ PILë¡? ????¥?•˜ê¸? ?œ„?•´ CPUë¡? ?´?™
                    fake_batch = generator(noise_batch).cpu()

                    # ?´ë¯¸ì??(-1,1)ë¥? (0,1)ë¡? ë³??™˜ ?›„ ????¥
                    for img in fake_batch:
                        img_01 = (img + 1) / 2.0
                        pil_img = TF.to_pil_image(img_01)
                        pil_img.save(os.path.join(fake_dir, f"fake_{fake_idx:05d}.png"))
                        fake_idx += 1

                    del fake_batch, noise_batch
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

            #FID ê³„ì‚°
            paths = [real_dir, fake_dir]
            fid_value = fid_score.calculate_fid_given_paths(
                paths, batch_size=fid_batch_size, device=device, dims=2048
            )
            print(f"Epoch {epoch+1:03d} | FID: {fid_value:.4f}")

            # real/fake ?´ë¯¸ì?? ?””? ‰?† ë¦¬ëŠ” FID ê³„ì‚° ?›„ ?‚­? œ
            shutil.rmtree(real_dir)
            shutil.rmtree(fake_dir)

            #eval() ëª¨ë“œ????˜ ê²ƒì„ ?‹¤?‹œ train() ëª¨ë“œë¡? ë³µì›
            generator.train()
            discriminator.train()

        print(f"Epoch [{epoch+1}/{epochs}] D_loss: {avg_D:.4f}, G_loss: {avg_G:.4f}")

    # ?•™?Šµ ?™„ë£? ?›„ ?†?‹¤ ê·¸ë˜?”„ ????¥
    torch.cuda.empty_cache()  # ?•™?Šµ ì¢…ë£Œ ?‹œ?—?„ ë©”ëª¨ë¦? ? •ë¦?
    plt.figure(figsize=(6, 4))
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./results/training_loss.png")
    plt.close()

    print("?•™?Šµ?´ ?™„ë£Œë˜?—ˆ?Šµ?‹ˆ?‹¤!")


if __name__ == "__main__":
    # ë°°ì¹˜ ?¬ê¸°ì?? ?‚¬?š©?•  ?´ë¯¸ì?? ?ˆ˜(max_images) ì¡°ì •
    batch_size = 16       # ?›?•˜?Š” ë°°ì¹˜ ?¬ê¸? (?˜ˆ: 4, 8, 16 ?“±)
    max_images = 10000   # ?•?˜ 10000ê°œë§Œ ?‚¬?š©
    epochs = 100         # ? „ì²? ?—?­ ?ˆ˜
    lr = 0.00005     # ?•™?Šµë¥?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"

    # ?°?´?„° ë¡œë”©
    print("?°?´?„° ë¡œë”© ì¤?...")
    dataloader = load_data(batch_size, img_dir, max_images=max_images)
    print(f"?°?´?„°?…‹ (?• {max_images}ê°?) ?¬ê¸?: {len(dataloader.dataset)}")

    # ëª¨ë¸ ?ƒ?„±
    print("ëª¨ë¸ ?ƒ?„± ì¤?...")
    G = Generator().to(device)
    D = Discriminator().to(device)

    # ëª¨ë¸ ?ŒŒ?¼ë¯¸í„° ?ˆ˜ ì¶œë ¥
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print(f"Generator ?ŒŒ?¼ë¯¸í„°: {count_params(G):,}")
    print(f"Discriminator ?ŒŒ?¼ë¯¸í„°: {count_params(D):,}")

    # ?•™?Šµ ?‹œ?‘
    train(
        G, D, dataloader, epochs, lr, device,
        fid_batch_size=16,
        fid_num_images=1000,
        fid_every=1
    )
