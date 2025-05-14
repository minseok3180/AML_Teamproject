import torch
from torchvision import datasets

datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        )

