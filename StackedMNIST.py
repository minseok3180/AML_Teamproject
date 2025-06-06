import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

class StackedMNIST(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.mnist = datasets.MNIST(root=root, train=train, download=download)
        self.transform = transform
        self.length = min(len(self.mnist), 100)
        self.resize = transforms.Resize(32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        imgs = []
        labels = []
        for _ in range(3):
            rand_idx = torch.randint(0, len(self.mnist), (1,)).item()
            img, label = self.mnist[rand_idx]
            img = self.resize(img)
            img = transforms.ToTensor()(img)
            imgs.append(img)
            labels.append(label)
        stacked_img = torch.cat(imgs, dim=0)
        if self.transform:
            stacked_img = self.transform(stacked_img)
        return stacked_img, tuple(labels)

os.makedirs('stacked_MNIST', exist_ok=True)
stacked_mnist = StackedMNIST(root='.', train=True, download=True)
loader = DataLoader(stacked_mnist, batch_size=1, shuffle=False)

for i, (imgs, labels) in enumerate(loader):
    img = imgs[0]
    label = labels[0]  # label: torch.Tensor([R, G, B]) 또는 tuple(R, G, B)
    # mod = '751' 형태로 생성
    if isinstance(label, torch.Tensor):
        label = label.tolist()
    mod = ''.join([str(int(x)) for x in label])  # 반드시 각 값 int로 변환해서 붙임
    filename = f'stacked_MNIST/{mod}.png'        # 파일명은 오직 mod만!
    save_image(img, filename)
    if (i+1) % 1000 == 0:
        print(f"{i+1}장 저장 완료")
    if i+1 >= 10000:
        break
