import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

### FFHQ-256 ###
class FFHQ256Dataset(Dataset):
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
    

def load_data_ffhq256(batch_size: int, img_dir: str = './data/ffhq', max_images: int = None) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    full_dataset = FFHQ256Dataset(img_dir, transform)

    if max_images is not None and max_images < len(full_dataset):
        indices = list(range(max_images))
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    
    return dataloader


        # print("FFHQ256 data loading...")
        # img_dir = "./data/ffhq256"
        # dataloader = load_data_ffhq256(batch_size, img_dir, max_images=max_images)
        # gen_base_channels = [256, 128, 64, 32, 16, 8, 4]  # for 256x256 output