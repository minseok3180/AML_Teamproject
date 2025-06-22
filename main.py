# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math 
import copy

# source code
from dataloader import load_data_ffhq64, load_data_StackMNIST, load_data_cifar10, load_data_imagenet32
from loss import r1_penalty, r2_penalty, discriminator_rploss, generator_rploss, discriminator_hinge_rploss, generator_hinge_rploss
from metric import fid_scoring, NFETracker
from logger import Logger 
from train_classifier import Classifier, train_classifier

import numpy as np
import torch.nn.functional as F

# Change mhsa
use_mhsa = False
if use_mhsa:
    from model_mhsa import Generator, Discriminator
    print("Using MHSA-enhanced model (model_mhsa.py)")
else:
    from model import Generator, Discriminator
    print("Using baseline model (model.py)")

# Global variables 
duration_mimg   = 10.0
duration_images = duration_mimg * 1_000_000   

beta2_start  = 0.9      # inital β₂
beta2_end    = 0.99     # final β₂
burn_in_mimg = 2.0      # Mimg

gamma_start = 1.0    # initial γ
gamma_end   = 0.1    # final γ

ema_start     = 0.0      # initial half-life in Mimg
ema_end       = 0.5      # target  half-life in Mimg

def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: DataLoader,
    img_type: str,
    img_name,
    epochs: int,
    lr: float,
    r1_lambda: float,
    r2_lambda: float,
    device: torch.device,
    switch_loss: bool,
    switch_epoch: int,
    fid_batch_size: int = 32,
    fid_num_images: int = 1000,
    fid_every: int = 5,
    ):
    """
    Training loop
    Args:
        generator (nn.Module)
        discriminator (nn.Module)
        dataloader (DataLoader)
        epochs (int)
        lr (float) 
        device (torch.device)
    """
    # optimizer
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, beta2_start))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, beta2_start))

    ema_generator = copy.deepcopy(generator)
    for p in ema_generator.parameters():
        p.requires_grad_(False)

    # for burn-in 
    dataset_size   = len(dataloader.dataset)        # 예: 128_000
    images_seen = 0
    burn_in_images = burn_in_mimg * 1_000_000       # 2 Mimg → 2,000,000
    burn_in_epochs = math.ceil(burn_in_images / dataset_size)

    # logging
    logger = Logger(log_dir="./logs")
    logger.log_initial(epochs, fid_batch_size, r1_lambda, r2_lambda, device, img_name)

    nz = 100
    torch.manual_seed(42)
    fixed_noise = torch.randn(16, nz, device=device) # Every 50 epochs, feed the same noise into the Generator and compare the generated images.
    
    # Fid Setting
    real_dataset = dataloader.dataset
    num_real_images = len(real_dataset)
    fid_real_indices = list(range(min(fid_num_images, num_real_images))) 
    fid_real_subset = Subset(real_dataset, fid_real_indices) 
    fixed_fid_noise = torch.randn(len(fid_real_indices), nz, device=device)

    # NFE Setting 
    nfe_tracker = NFETracker()

    G_losses = [] # record mean_Generator loss for every epoch 
    D_losses = [] # record mean_Discriminator loss for every epoch
    Fid_list = []
    Mode_coverage_list = []
    KL_divergence_list = []
    os.makedirs('./results', exist_ok=True)
    
    if img_type == 'd1':
        clf = Classifier(num_classes=1000).to(device)
        clf.load_state_dict(torch.load('stacked_mnist_classifier/stacked_mnist_classifier.pth', map_location=device))
        clf.eval()

        mode_counts = np.zeros(1000, dtype=int)
        prob_sum = np.zeros(1000, dtype=float)
        total_samples = 0

    print(f"Training for {epochs} epochs (burn-in: {burn_in_epochs} epochs)...")
    for epoch in range(epochs):
                
        epoch_D_loss = 0.0 # one epoch - Discriminator loss
        epoch_G_loss = 0.0 # one epoch - Generator loss 
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, batch in enumerate(pbar):
            if img_type == 'd1':
                real_images, _ = batch
            else:
                real_images = batch

            batch_size = real_images.size(0) # the number of image in batch
            real_images = real_images.to(device)

            images_seen += batch_size
            t = min(images_seen, burn_in_images) / burn_in_images  # ∈ [0,1]

            
            # beta scheduling 
            beta2 = beta2_start + 0.5 * (beta2_end - beta2_start) * (1 - math.cos(math.pi * t))
            for opt in (optimizer_G, optimizer_D):
                for pg in opt.param_groups:
                    pg['betas'] = (0.5, beta2)

            # gamma scheduling
            gamma = gamma_start + 0.5*(gamma_end - gamma_start)*(1 - math.cos(math.pi * t))

            current_ema_hl_mimg   = ema_start + 0.5*(ema_end - ema_start)*(1 - math.cos(math.pi*t))
            current_ema_hl_images = current_ema_hl_mimg * 1_000_000

            if current_ema_hl_mimg == 0.0:
                ema_decay = 0.0
            else:
                H = current_ema_hl_mimg * 1_000_000 
                ema_decay = math.exp(-math.log(2) * batch_size / H)

            #######################
            # Discriminator Update
            #######################

            # gradient initialize 
            discriminator.zero_grad()
            
            # noise fake images generate
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)

            if img_type == 'd1':
                with torch.no_grad():
                    logits = clf(fake_images)
                    probs = F.softmax(logits, dim=1).cpu().numpy()  # (batch_size, 1000)
                preds = probs.argmax(axis=1)
                for c in preds:
                    mode_counts[c] += 1
                prob_sum += probs.sum(axis=0)
                total_samples += batch_size
            
            # Discriminator loss
            if not switch_loss:
                # 1) unwrap the DataParallel module so penalty runs on main GPU
                main_disc = discriminator.module 
                # 2) move images to cuda:0 for penalty
                real_images0 = real_images.clone().detach().to('cuda:0').requires_grad_(True)
                fake_images0 = fake_images.clone().detach().to('cuda:0').requires_grad_(True)
                # 3) compute RP‐loss using main_disc on cuda:0
                d_loss_basic = nn.functional.softplus(-(discriminator(real_images) - discriminator(fake_images).detach())).mean()
                penalty_r1 = r1_penalty(main_disc, real_images0, r1_lambda)
                penalty_r2 = r2_penalty(main_disc, fake_images0, r2_lambda)
                d_loss = d_loss_basic + gamma * (penalty_r1 + penalty_r2)
            else:
                if epoch < switch_epoch:
                    d_loss = discriminator_rploss(discriminator, real_images, fake_images, gamma, r1_lambda=r1_lambda, r2_lambda=r2_lambda)
                else:
                    d_loss = discriminator_hinge_rploss(discriminator, real_images, fake_images, gamma, r1_lambda=r1_lambda, r2_lambda=r2_lambda)

            d_loss.backward() # backprop 
            optimizer_D.step() # parameter update 
            

            ###################
            # Generator Update
            ###################

            # gradient initialize 
            generator.zero_grad()
            
            # noise fake images generate
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)
            
            # Generator loss 
            if not switch_loss:
                g_loss = generator_rploss(discriminator, real_images, fake_images)
            else:
                if epoch < switch_epoch:  #switch_epoch: epochs / 2
                    g_loss = generator_rploss(discriminator, real_images, fake_images)
                else:
                    g_loss = generator_hinge_rploss(discriminator, real_images, fake_images)
            
            nfe_tracker.increment()

            g_loss.backward() # backprop
            optimizer_G.step() # parameter update 

            # EMA
            with torch.no_grad():
               for p_ema, p in zip(ema_generator.parameters(), generator.parameters()):
                   p_ema.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

            
            # Loss
            epoch_D_loss += d_loss.item()
            epoch_G_loss += g_loss.item()
            
            pbar.set_postfix({
                'D_loss': f'{d_loss.item():.4f}',
                'G_loss': f'{g_loss.item():.4f}'
            })
            
        
        avg_D_loss = epoch_D_loss / len(dataloader) # Mean Discriminator loss for one epoch 
        avg_G_loss = epoch_G_loss / len(dataloader) # Mean Generator loss for one epoch
        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)

        
        # Save generated image
        if (epoch + 1) % 1 == 0:
            with torch.no_grad():
                fake_samples = ema_generator(fixed_noise).cpu()
                plt.figure(figsize=(12, 12))
                plt.axis("off")
                plt.title(f"Generated FFHQ Images - Epoch {epoch+1}")
                for idx in range(16):
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(fake_samples[idx].permute(1, 2, 0) * 0.5 + 0.5)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'./results/generated_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # Save check point
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, f'./results/checkpoint_epoch_{epoch+1}.pth')

        if img_type == 'd1':
            p_y = prob_sum / total_samples
            coverage = int((mode_counts > 0).sum())
            q = np.full(1000, 1/1000)
            p_safe = np.where(p_y > 0, p_y, 1e-12)
            inv_kl = float((q * np.log(q / p_safe)).sum())
            print(f"[Epoch {epoch}] Mode Coverage: {coverage}/1000, Inverse KL: {inv_kl:.6f}")
            
        # fid scoring
        fid_value = fid_scoring(epoch,
        fid_every, 
        generator, 
        discriminator, 
        fid_real_subset, 
        fid_batch_size,
        fid_real_indices,
        fixed_fid_noise,
        img_type,
        device)
        
        print(f'Epoch [{epoch+1}/{epochs}] - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')
        print(f"Total NFE in Epoch {epoch+1}: {nfe_tracker.get_nfe()}")
        
        Fid_list.append(fid_value)
        Mode_coverage_list.append(coverage if img_type == 'd1' else None)
        KL_divergence_list.append(inv_kl if img_type == 'd1' else None)

        if img_type == 'd1':
            logger.logd1(epoch, avg_G_loss, avg_D_loss, fid_value, coverage, inv_kl)
        else:
            logger.log(epoch, avg_G_loss, avg_D_loss, fid_value)

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./results/training_loss.png')
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("Fid During Training")
    plt.plot(Fid_list, label="FID")
    plt.xlabel("Epochs")
    plt.ylabel("FID")
    plt.legend()
    plt.savefig('./results/training_fid.png')
    plt.close()


    if img_type == 'd1':
        plt.figure(figsize=(10, 5))
        plt.title("Mode Coverage During Training")
        plt.plot(Mode_coverage_list, label="Mode Coverage")
        plt.xlabel("Epochs")
        plt.ylabel("Mode Coverage")
        plt.legend()
        plt.savefig('./results/training_mode_coverage.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.title("Inverse KL Divergence During Training")
        plt.plot(KL_divergence_list, label="Inverse KL Divergence")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('./results/training_kl.png')
        plt.close()

    
    print("Finished Training!")
    logger.log_final(epochs, avg_G_loss, avg_D_loss, fid_value)


if __name__ == "__main__":

    '''
    img_type
    dataset 1 : Stacked MNIST
    dataset 2 : FFHQ-64
    dataset 3 : CIFAR-10
    dataset 4 : ImageNet-32
    '''

    img_type = 'd1'
    batch_size = 512
    max_images = 128000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    

    if img_type == 'd1':
        print("Stacked MNIST data loading...")
        img_dir = "./data/mnist"
        dataloader = load_data_StackMNIST(batch_size, img_dir, max_images=max_images)
        gen_base_channels = [256, 256, 256, 256]        # for 32x32 output

        img_name = 'Stacked MNIST'
        lr = 0.0002
        r1_lambda = 10
        r2_lambda = 10
        clf_epochs = 10
        train_classifier(dataloader, clf_epochs, lr, device)


    elif img_type == 'd2' : 
        print("FFHQ64 data loading...")
        tfrecord_dir = "./data/ffhq64"  # Point to the tfrecord directory
        dataloader = load_data_ffhq64(batch_size, max_images=max_images)
        gen_base_channels = [128, 256, 256, 256, 256]

        img_name = 'FFHQ-64'
        lr = 0.0002
        r1_lambda = 2
        r2_lambda = 2

    elif img_type == 'd3' : 
        print("cifar-10 data loading...")
        img_dir = "./data/cifar-10"
        dataloader = load_data_cifar10(batch_size, img_dir, max_images=max_images)
        gen_base_channels = [256, 128, 64, 32]        # for 32x32 output 

        img_name = 'CIFAR-10'
        lr = 0.0002
        r1_lambda = 2
        r2_lambda = 2

    elif img_type == 'd4':
        print("ImageNet-32 data loading...")
        img_dir = "./data/imagenet32"
        dataloader = load_data_imagenet32(batch_size, img_dir, max_images=max_images)
        gen_base_channels = [1536, 1536, 1536, 1536]        # for 32x32 output 

        img_name = 'ImageNet-32'
        lr = 0.0002
        r1_lambda = 2
        r2_lambda = 2

    else : print('data type error!')

    dataset_size    = len(dataloader.dataset) 
    epochs = math.ceil(duration_images / dataset_size)

    # Discriminator channels: reverse of generator
    disc_base_channels = list(reversed(gen_base_channels))
    
    
    print("############################### model generating ###############################")
    print(f"max images : {max_images} --- dataset size: {len(dataloader.dataset)}")
    G = Generator(BaseChannels=gen_base_channels).to(device)
    D = Discriminator(BaseChannels=disc_base_channels).to(device)

    # DataParallel 추가
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    G = G.cuda()
    D = D.cuda()
    
    G.train()
    D.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Generator parameter: {count_parameters(G):,}")
    print(f"Discriminator parameter: {count_parameters(D):,}")

    print(f'Using device : {device}')
    print(f'epoch : {epochs}')
    print(f'batch size : {batch_size}')
    print(f'learning rate : {lr}')
    print(f'r1_lambda : {r1_lambda}')
    print(f'r2_lambda : {r2_lambda}')
    

    train(G, D, dataloader, img_type, img_name, epochs, lr, r1_lambda, r2_lambda, device,
        switch_loss=False,
        switch_epoch=int(epochs/2),
        fid_batch_size=batch_size,
        fid_num_images=12800,
        fid_every=1)