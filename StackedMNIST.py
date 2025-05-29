import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class StackedMNIST(Dataset):
    """
    Stacked MNIST dataset (RGB: 각 채널마다 서로 다른 MNIST digit)
    50,000장으로 제한하고 32x32 크기로 리사이징
    """
    def __init__(self, root, train=True, download=True, transform=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.transform = transform

        # 데이터셋 크기를 50,000장으로 제한
        self.length = min(len(self.mnist), 50000)
        
        # 기본 변환: 32x32 리사이징 추가
        self.resize = transforms.Resize(32, antialias=True)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 세 개의 서로 다른 MNIST digit을 쌓아서 R,G,B에 넣음
        imgs = []
        labels = []
        for _ in range(3):
            rand_idx = torch.randint(0, len(self.mnist), (1,)).item()
            img, label = self.mnist[rand_idx]
            imgs.append(img)
            labels.append(label)
        # (3, 32, 32)로 쌓기
        stacked_img = torch.cat([transforms.ToTensor()(img) for img in imgs], dim=0)
        # 32x32로 리사이징
        stacked_img = self.resize(stacked_img)
        stacked_label = tuple(labels)  # (R_label, G_label, B_label)
        
        if self.transform:
            stacked_img = self.transform(stacked_img)
        return stacked_img, stacked_label


# 데이터셋 생성
stacked_mnist = StackedMNIST(root='./data', train=True, download=True)

# DataLoader
loader = DataLoader(stacked_mnist, batch_size=64, shuffle=True)

# 배치 확인
for imgs, labels in loader:
    print(imgs.shape)       # torch.Size([64, 3, 32, 32])
    print(labels)           # e.g., (7, 1, 3)
    break