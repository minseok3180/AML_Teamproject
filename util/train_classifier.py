import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def combine_labels(label_tuple):
    return label_tuple[0] * 100 + label_tuple[1] * 10 + label_tuple[2]

class Classifier(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*8*8, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device
):
    model = Classifier(num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            # labels is a tuple of three tensors (r,g,b), each shape (B,)
            r, g, b = labels
            r = r.to(device).long()
            g = g.to(device).long()
            b = b.to(device).long()
            # batch-wise combine
            targets = r * 100 + g * 10 + b  # shape (B,)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += (preds == targets).sum().item()
            total += imgs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        print(f"Epoch {epoch}/{epochs}  Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%")

    # save model
    os.makedirs('stacked_mnist_classifier', exist_ok=True)
    torch.save(model.state_dict(), 'stacked_mnist_classifier/stacked_mnist_classifier.pth')
