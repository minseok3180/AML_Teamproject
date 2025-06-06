# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# source code
from dataloader import load_data
from model import Generator, Discriminator
from loss import discriminator_rploss, generator_rploss
from fid import fid_scoring


def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    r1_lambda: float,
    r2_lambda: float,
    device: torch.device,
    fid_batch_size: int = 32,
    fid_num_images: int = 1000,
    fid_every: int = 5):
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
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999)) 
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    nz = generator.noise_dim
    fixed_noise = torch.randn(16, nz, device=device) # Every 50 epochs, feed the same noise into the Generator and compare the generated images.
    
    # fid
    real_dataset = dataloader.dataset
    num_real_images = len(real_dataset)
    fid_real_indices = list(range(min(fid_num_images, num_real_images))) 
    fid_real_subset = Subset(real_dataset, fid_real_indices) 
    fixed_fid_noise = torch.randn(len(fid_real_indices), nz, device=device)

    G_losses = [] # record mean_Generator loss for every epoch 
    D_losses = [] # record mean_Discriminator loss for every epoch
    os.makedirs('./results', exist_ok=True)
    
    print("Training...")
    for epoch in range(epochs):
        epoch_D_loss = 0.0 # one epoch - Discriminator loss
        epoch_G_loss = 0.0 # one epoch - Generator loss 
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for i, real_images in enumerate(pbar): # i : batch index 
            batch_size = real_images.size(0) # the number of image in batch
            real_images = real_images.to(device)
            
            #######################
            # Discriminator Update
            #######################

            # gradient initialize 
            discriminator.zero_grad()
            
            # noise fake images generate
            noise = torch.randn(batch_size, nz, device=device)
            fake_images = generator(noise)
            
            # RpGAN Discriminator loss + R1 + R2
            d_loss = discriminator_rploss(
                discriminator,
                real_images,
                fake_images,
                r1_lambda=r1_lambda,
                r2_lambda=r2_lambda
            )
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
            
            # RpGAN Generator loss 
            g_loss = generator_rploss(
                discriminator,
                real_images,
                fake_images
            )
            g_loss.backward() # backprop
            optimizer_G.step() # parameter update 
            
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
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise).detach().cpu()
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
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
            }, f'./results/checkpoint_epoch_{epoch+1}.pth')

        # fid scoring
        fid_scoring(epoch,
        fid_every, 
        generator, 
        discriminator, 
        fid_real_subset, 
        fid_batch_size,
        fid_real_indices,
        fixed_fid_noise,
        device)
        
        print(f'Epoch [{epoch+1}/{epochs}] - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')
    
    # Loss graph
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('./results/training_loss.png')
    plt.close()
    
    print("Finished Training!")


if __name__ == "__main__":

    ### parameter
    batch_size = 64
    max_images = 10000
    epochs = 1000
    img_dir = "/home/elicer/AML_Teamproject/ffhq256"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("FFHQ256 data loading...")
    dataloader = load_data(batch_size, img_dir, max_images=max_images)
    print(f"max images : {max_images} --- dataset size: {len(dataloader.dataset)}")

    print("model generating...")
    G = Generator().to(device)
    D = Discriminator().to(device)

    G.train()
    D.train()
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Generator parameter: {count_parameters(G):,}")
    print(f"Discriminator parameter: {count_parameters(D):,}")

    lr = 0.0002
    r1_lambda = 10.0
    r2_lambda = 10.0
    train(G, D, dataloader, epochs, lr, r1_lambda, r2_lambda, device,
        fid_batch_size=64,
        fid_num_images=1000,
        fid_every=1)