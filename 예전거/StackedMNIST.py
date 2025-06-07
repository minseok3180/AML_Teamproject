import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import gzip
from typing import Tuple, Optional


class StackedMNISTDataset(Dataset):
    """
    Stacked MNIST Dataset for PyTorch
    
    3개의 MNIST 이미지를 RGB 채널에 각각 배치하여 스택된 이미지를 생성합니다.
    레이블은 각 채널의 숫자를 결합하여 3자리 수로 표현됩니다 (예: R=1, G=2, B=3 → 123).
    """
    
    def __init__(self, 
                 mnist_dir: str, 
                 num_images: int = 1000000, 
                 random_seed: int = 123,
                 transform: Optional[transforms.Compose] = None):
        """
        Args:
            mnist_dir: MNIST 데이터가 저장된 디렉토리 경로
            num_images: 생성할 이미지 수
            random_seed: 랜덤 시드
            transform: 이미지 변환 함수
        """
        self.num_images = num_images
        self.transform = transform
        
        # MNIST 데이터 로드
        self.images, self.labels = self._load_mnist(mnist_dir)
        
        # 스택된 이미지와 레이블 생성
        self.stacked_images, self.stacked_labels = self._create_stacked_data(random_seed)
        
    def _load_mnist(self, mnist_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """MNIST 데이터를 로드합니다."""
        print(f'Loading MNIST from "{mnist_dir}"')
        
        # 이미지 로드
        with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
            images = np.frombuffer(file.read(), np.uint8, offset=16)
        
        # 레이블 로드
        with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
            labels = np.frombuffer(file.read(), np.uint8, offset=8)
        
        # 이미지 reshape 및 패딩
        images = images.reshape(-1, 28, 28)
        images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
        
        # 검증
        assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
        assert labels.shape == (60000,) and labels.dtype == np.uint8
        assert np.min(images) == 0 and np.max(images) == 255
        assert np.min(labels) == 0 and np.max(labels) == 9
        
        return images, labels.astype(np.float64)
    
    def _create_stacked_data(self, random_seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """스택된 이미지와 레이블을 생성합니다."""
        print(f'Creating {self.num_images} stacked images...')
        
        rnd = np.random.RandomState(random_seed)
        stacked_images = []
        stacked_labels = []
        
        for idx in range(self.num_images):
            # 3개의 랜덤 인덱스 선택
            indices = rnd.randint(self.images.shape[0], size=3)
            
            # RGB 채널로 스택
            rgb_image = np.stack([
                self.images[indices[0]],  # R 채널
                self.images[indices[1]],  # G 채널
                self.images[indices[2]]   # B 채널
            ], axis=0)
            
            stacked_images.append(rgb_image)
            
            # 레이블 결합 (예: R=1, G=2, B=3 → 123)
            combined_label = (self.labels[indices[0]] + 
                            self.labels[indices[1]] * 10 + 
                            self.labels[indices[2]] * 100)
            stacked_labels.append(combined_label)
        
        # NumPy 배열을 PyTorch 텐서로 변환
        stacked_images = torch.from_numpy(np.array(stacked_images, dtype=np.uint8))
        stacked_labels = torch.from_numpy(np.array(stacked_labels, dtype=np.int64))
        
        # 검증
        assert stacked_images.shape == (self.num_images, 3, 32, 32)
        assert stacked_labels.shape == (self.num_images,)
        assert torch.min(stacked_labels) == 0 and torch.max(stacked_labels) == 999
        
        print(f'Successfully created {self.num_images} stacked images')
        return stacked_images, stacked_labels
    
    def __len__(self) -> int:
        return self.num_images
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.stacked_images[idx].float() / 255.0  # [0, 1]로 정규화
        label = self.stacked_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def get_onehot_labels(self) -> torch.Tensor:
        """원-핫 인코딩된 레이블을 반환합니다."""
        num_classes = 1000  # 0-999
        onehot = torch.zeros(self.num_images, num_classes, dtype=torch.float32)
        onehot.scatter_(1, self.stacked_labels.unsqueeze(1), 1.0)
        return onehot
    
    def save_dataset(self, save_path: str):
        """데이터셋을 파일로 저장합니다."""
        torch.save({
            'images': self.stacked_images,
            'labels': self.stacked_labels,
            'num_images': self.num_images
        }, save_path)
        print(f'Dataset saved to {save_path}')
    
    @classmethod
    def load_dataset(cls, load_path: str, transform: Optional[transforms.Compose] = None):
        """저장된 데이터셋을 로드합니다."""
        data = torch.load(load_path)
        dataset = cls.__new__(cls)
        dataset.stacked_images = data['images']
        dataset.stacked_labels = data['labels']
        dataset.num_images = data['num_images']
        dataset.transform = transform
        print(f'Dataset loaded from {load_path}')
        return dataset


def create_stacked_mnist_dataset(mnist_dir: str, 
                               num_images: int = 1000000, 
                               random_seed: int = 123,
                               save_path: Optional[str] = None) -> StackedMNISTDataset:
    """
    Stacked MNIST 데이터셋을 생성합니다.
    
    Args:
        mnist_dir: MNIST 데이터가 저장된 디렉토리 경로
        num_images: 생성할 이미지 수
        random_seed: 랜덤 시드
        save_path: 데이터셋을 저장할 경로 (선택사항)
    
    Returns:
        StackedMNISTDataset 객체
    """
    dataset = StackedMNISTDataset(mnist_dir, num_images, random_seed)
    
    if save_path:
        dataset.save_dataset(save_path)
    
    return dataset


# 사용 예제
if __name__ == "__main__":
    # 데이터셋 생성
    mnist_dir = "./data/mnist"  # MNIST 데이터 경로
    dataset = create_stacked_mnist_dataset(
        mnist_dir=mnist_dir,
        num_images=10000,  # 예제로 작은 수
        random_seed=123,
        save_path="stacked_mnist_dataset.pt"
    )
    
    # 데이터 로더 생성
    transform = transforms.Compose([
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]로 정규화
    ])
    
    dataset.transform = transform
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 첫 번째 배치 확인
    images, labels = next(iter(dataloader))
    print(f"Batch shape: {images.shape}")  # [32, 3, 32, 32]
    print(f"Labels shape: {labels.shape}")  # [32]
    print(f"Sample labels: {labels[:5]}")
    
    # 원-핫 레이블 확인
    onehot_labels = dataset.get_onehot_labels()
    print(f"One-hot labels shape: {onehot_labels.shape}")  # [num_images, 1000]