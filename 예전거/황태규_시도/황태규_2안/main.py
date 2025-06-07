# main.py
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from model import Generator, Discriminator
import os
from PIL import Image

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
        if self.transform: img = self.transform(img)
        return img, 0

def load_data(batch_size: int, image_dir) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((128, 128)),   # 해상도 128로 맞춤
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = FlatImageDataset(image_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

def train(generator, discriminator, dataloader, epochs, device, lr=2e-4, out_dir='/home/elicer/AML_Teamproject/황태규_시도/황태규_2안/samples'):
    os.makedirs(out_dir, exist_ok=True)
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = torch.nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(16, 128, device=device)  # z_dim=128

    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            # --- Discriminator
            noise = torch.randn(batch_size, 128, device=device)
            fake_imgs = generator(noise)
            out_real = discriminator(real_imgs)
            out_fake = discriminator(fake_imgs.detach())
            real_labels = torch.ones_like(out_real)
            fake_labels = torch.zeros_like(out_fake)
            loss_D = 0.5 * (criterion(out_real, real_labels) + criterion(out_fake, fake_labels))
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()
            # --- Generator
            out_fake = discriminator(fake_imgs)
            loss_G = criterion(out_fake, real_labels)
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        # 에폭별 샘플 저장
        with torch.no_grad():
            samples = generator(fixed_noise).add(1).div(2)  # [0,1]로
            vutils.save_image(samples, f"{out_dir}/epoch_{epoch+1:03d}.png", nrow=4)
        print(f"Epoch {epoch+1}/{epochs}: D_loss={loss_D.item():.4f}, G_loss={loss_G.item():.4f}")

if __name__ == "__main__":
    batch_size = 32
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_dir = "/home/elicer/AML_Teamproject/ffhq256"
    dataloader = load_data(batch_size, image_dir)
    G = Generator().to(device)
    D = Discriminator().to(device)
    train(G, D, dataloader, epochs, device)
