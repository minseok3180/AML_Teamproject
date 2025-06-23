import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from PIL import Image
import gzip
from typing import Tuple, Optional
import PIL
from datasets import load_dataset


### CIFAR-10 ###
class CIFAR10Dataset(Dataset):
    """
    폴더 내에 있는 .png 이미지 파일을 모두 읽어
    RGB 텐서로 반환하는 Dataset
    """
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = root_dir
        # .png 확장자를 가진 파일만 선택, 정렬하여 일관된 순서 보장
        self.image_paths = sorted([
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.lower().endswith('.png')
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

def load_data_cifar10(
    batch_size: int,
    data_dir: str = './data/cifar-10',
    max_images: int = None
) -> DataLoader:
    """
    CIFAR-10 원본 .png 이미지(32×32) 전용 DataLoader 반환
      - data_dir: .png 파일들이 들어있는 디렉토리 경로
      - max_images: 전체 이미지 중 앞의 N개만 사용 (None이면 전체)
      - shuffle=True, drop_last=True로 설정
    """
    transform = transforms.Compose([
        transforms.ToTensor(),                               # [0,1] → 텐서
        transforms.Normalize((0.5, 0.5, 0.5),                # 채널별 평균
                             (0.5, 0.5, 0.5))                # 채널별 표준편차
    ])

    dataset = CIFAR10Dataset(root_dir=data_dir, transform=transform)

    # max_images 옵션 적용
    if max_images is not None and max_images < len(dataset):
        dataset = Subset(dataset, list(range(max_images)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    return loader

### ImageNet-32 ###

class ImageNet32Dataset(Dataset):
    def __init__(self, npz_path, transform=None):
        """
        npz_path: 경로/train_data_batch_1.npz
        transform: torchvision.transforms.Compose
        """
        data = np.load(npz_path, allow_pickle=True)  # allow loading of pickled data
        # data['data'] shape: (N, 3072), dtype=uint8
        self.images = data['data']
        # labels dropped; only images needed
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # flat array → (3, 32, 32)
        img = self.images[idx].reshape(3, 32, 32)
        # numpy uint8 [0–255] → PIL Image
        img = Image.fromarray(img.transpose(1, 2, 0), mode='RGB')
        if self.transform:
            img = self.transform(img)
        return img  # return only image


def load_data_imagenet32(
    batch_size: int,
    img_dir: str = './data/imagenet32',
    max_images: int = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    transform = transforms.Compose([
        transforms.ToTensor(),               # [0,255] → [0,1]
        transforms.Normalize([0.5]*3, [0.5]*3)  # → [–1,1]
    ])

    npz_path = os.path.join(img_dir, 'train_data_batch_1.npz')
    dataset = ImageNet32Dataset(npz_path, transform)

    if max_images is not None and max_images < len(dataset):
        dataset = Subset(dataset, list(range(max_images)))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )

    return dataloader


### FFHQ-64 ###

class FFHQ64Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img = self.dataset[idx]['image']
        img = Image.fromarray(np.array(img))  
        if self.transform:
            img = self.transform(img)
        
        return img

def load_data_ffhq64(batch_size: int, max_images: int = None):
    dataset = load_dataset("Dmini/FFHQ-64x64")['train']
    if max_images is not None:
        dataset = dataset.select(range(min(max_images, len(dataset))))
    
    transform = transforms.Compose([
        transforms.Resize(64),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
    ])
    
    ffhq_dataset = FFHQ64Dataset(dataset, transform)
    dataloader = DataLoader(
        ffhq_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  
        pin_memory=True,
        drop_last=True
    )
    return dataloader

### Stacked MNIST ###


class StackedMNISTDataset(Dataset):
    def __init__(self, mnist_dir: str, num_images: int = 1000000, random_seed: int = 123, transform=None):
        self.num_images = num_images
        self.transform = transform
        
        self.images, self.labels = self._load_mnist(mnist_dir)
        
        #self.stacked_images = self._create_stacked_data(random_seed)
        self.stacked_images, self.stacked_labels = self._create_stacked_data(random_seed)
        
    def _load_mnist(self, mnist_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        print(f'Loading MNIST from "{mnist_dir}"')
        
        with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
            images = np.frombuffer(file.read(), np.uint8, offset=16)
        
        with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)
        
        images = images.reshape(-1, 28, 28)
        images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
        return images, labels.astype(np.float64)
    
    def _create_stacked_data(self, random_seed: int) -> torch.Tensor:
        print(f'Creating {self.num_images} stacked images...')
        
        rnd = np.random.RandomState(random_seed)
        stacked_imgs = []
        stacked_labs = []
        for _ in range(self.num_images):
            idxs = rnd.randint(len(self.images), size=3)
            # R,G,B 채널로 합치기
            rgb = np.stack([
                self.images[idxs[0]],
                self.images[idxs[1]],
                self.images[idxs[2]],
            ], axis=0)  # (3,32,32)
            stacked_imgs.append(rgb)
            stacked_labs.append((
                int(self.labels[idxs[0]]),
                int(self.labels[idxs[1]]),
                int(self.labels[idxs[2]])
            ))
        imgs_tensor = torch.from_numpy(np.array(stacked_imgs, dtype=np.uint8))
        print(f'Successfully created {self.num_images} stacked images')

        return imgs_tensor, stacked_labs
        
        # for idx in range(self.num_images):
        #     indices = rnd.randint(self.images.shape[0], size=3)
            
        #     rgb_image = np.stack([
        #         self.images[indices[0]],  # R
        #         self.images[indices[1]],  # G
        #         self.images[indices[2]]   # B
        #     ], axis=0)
            
        #     stacked_images.append(rgb_image)
        
        # stacked_images = torch.from_numpy(np.array(stacked_images, dtype=np.uint8))
        # print(f'Successfully created {self.num_images} stacked images')
        # return stacked_images
    
    def __len__(self) -> int:
        return self.num_images
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        image = self.stacked_images[idx].float() / 255.0 # [0, 1] Normalization
        if self.transform:
            image = self.transform(image)
        label = self.stacked_labels[idx]
        return image, label
    
        # image = self.stacked_images[idx].float() / 255.0  # [0, 1] Normalization
        
        # if self.transform:
        #     image = self.transform(image)
            
        # return image


def load_data_StackMNIST(batch_size: int, img_dir: str = './data/mnist', max_images: int = None) -> DataLoader:
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    if max_images is None:
        max_images = 10000
    
    full_dataset = StackedMNISTDataset(img_dir, num_images=max_images, transform=transform)

    # DDP
    world_size = torch.cuda.device_count()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        sampler = sampler,
        num_workers=2, 
        pin_memory=True, 
        drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    import os

    # 데이터 로드
    dataloader = load_data_StackMNIST(batch_size=64, img_dir='./data/mnist', max_images=1000)
    images, labels = next(iter(dataloader))  # images: [64,3,32,32], labels: (digit1, digit2, digit3)

    # (만약 이미지가 [-1,1]로 정규화되어 있다면 복원)
    # images = images * 0.5 + 0.5

    # 8x8 그리드로 묶기
    grid = make_grid(images, nrow=8, padding=2)

    # 폴더가 없다면 생성
    os.makedirs('created_mnist', exist_ok=True)
    save_path = 'created_mnist/stacked_mnist_samples.png'

    # 그리드 출력 및 파일 저장
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(
        "Stacked MNIST Samples\n" +
        "\n".join(f"{i}: {tuple(lbl.tolist())}" for i, lbl in enumerate(labels[:8]))
    )
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved sample grid to {save_path}")

    # (원래의 확인용 루프)
    for images, labels in dataloader:
        print(images.shape)  # Expected: [64, 3, 32, 32]
        print(labels)        # Expected: (digit1, digit2, digit3)
        break
