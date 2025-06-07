import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

use_mhsa = False
checkpoint_path = './results/ckpt_epoch_100.pth'          
classifier_path = './stacked_mnist_classifier/stacked_mnist_classifier.pth'
noise_dim = 100
batch_size = 128
num_samples = 10000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드
if use_mhsa:
    from model_mhsa import Generator
    print("Using MHSA-enhanced Generator")
else:
    from model import Generator
    print("Using baseline Generator")
G = Generator(NoiseDim=noise_dim).to(device)
# checkpoint 파일 구조에 맞춰 로드
ckpt = torch.load(checkpoint_path, map_location=device)
if 'generator_state_dict' in ckpt:
    G.load_state_dict(ckpt['generator_state_dict'])
else:
    G.load_state_dict(ckpt)
G.eval()

# 분류기 로드
from train_classifier import Classifier
clf = Classifier(num_classes=1000).to(device)
clf.load_state_dict(torch.load(classifier_path, map_location=device))
clf.eval()

# Mode Coverage & KL 계산
counts = np.zeros(1000, dtype=int)
prob_sum = np.zeros(1000, dtype=float)
processed = 0

torch.manual_seed(0)
while processed < num_samples:
    current_batch = min(batch_size, num_samples - processed)
    z = torch.randn(current_batch, noise_dim, device=device)
    with torch.no_grad():
        imgs = G(z)
        logits = clf(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (B,1000)
        preds = np.argmax(probs, axis=1)
    for c in preds:
        counts[c] += 1
    prob_sum += probs.sum(axis=0)
    processed += current_batch

p_y = prob_sum / processed
mode_coverage = int((counts > 0).sum())
q = np.full(1000, 1/1000)

# KL divergence to uniform
p_safe = np.where(p_y > 0, p_y, 1e-12)
kl_div = float(np.sum(p_safe * np.log(p_safe / q)))

print(f"Total generated samples: {processed}")
print(f"Mode coverage: {mode_coverage} / 1000")
print(f"KL divergence to uniform: {kl_div:.6f}")
