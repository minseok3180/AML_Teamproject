# main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import math 
import copy
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import warnings
import traceback 

# source code
from dataloader import load_data_ffhq64, load_data_StackMNIST, load_data_cifar10, load_data_imagenet32
from util.loss import r1_penalty, r2_penalty, discriminator_rploss, generator_rploss, discriminator_hinge_rploss, generator_hinge_rploss
from util.metric import fid_scoring, NFETracker
from util.logger import Logger 
from util.train_classifier import Classifier, train_classifier



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

beta2_start  = 0.9      # inital β₂
beta2_end    = 0.99     # final β₂
burn_in_mimg = 2.0      # Mimg

# warm-up span in Mimg
warmup_mimg  = 0.5      # warm-up over first 0.5 Mimg

gamma_start = 10.0    # initial γ
gamma_end   = 1    # final γ

ema_start     = 0.0      # initial half-life in Mimg
ema_end       = 0.5      # target  half-life in Mimg

lazy_interval = 8

def setup_directories():
    """Create necessary directories"""
    dirs = ['./results', './logs']
    for dir_path in dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create directory {dir_path}: {e}")

def is_main_process():
    """Check if current process is the main process in DDP"""
    try:
        return dist.get_rank() == 0
    except:
        return True  # Single GPU case

def train(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    dataloader: DataLoader,
    img_type: str,
    img_name,
    epochs: int,
    lr: float,
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
    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, beta2_start))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.5, beta2_start))

    # implement AMP
    # scaler = GradScaler()

    ema_generator = copy.deepcopy(generator)
    ema_generator.eval()

    for p in ema_generator.parameters():
        p.requires_grad_(False)

    # for burn-in 
    dataset_size   = len(dataloader.dataset)        
    images_seen = 0
    burn_in_images = burn_in_mimg * 1_000_000       # 2 Mimg → 2,000,000
    warmup_images  = warmup_mimg * 1_000_000        # linear warm-up span
    burn_in_epochs = math.ceil(burn_in_images / dataset_size)
    warmup_epochs = math.ceil(warmup_images / dataset_size)

    # logging
    logger = Logger(log_dir="./logs") if is_main_process() else None
    if logger:
        logger.log_initial(epochs, fid_batch_size, device, img_name)

    nz = 100
    torch.manual_seed(42)
    fixed_noise = torch.randn(16, nz, device=device) 
    
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
    r1_penalty_list = []
    r2_penalty_list = []
    
    setup_directories()

    clf = None
    mode_counts = None
    total_samples = 0
    coverage = 0
    rev_kl = 0.0
    
    if img_type == 'd1':
        clf = Classifier(num_classes=1000).to(device)

        classifier_path = 'stacked_mnist_classifier/stacked_mnist_classifier.pth'
        if os.path.exists(classifier_path):
            try: 
                clf.load_state_dict(torch.load(classifier_path, map_location=device))
                clf.eval()
                if is_main_process():
                    print(f"Loaded classifier from {classifier_path}")
            except Exception as e:
                if is_main_process():
                    print(f"Error loading classifier: {e}")
        else:
            if is_main_process():
                print(f"Warning: Classifier file {classifier_path} not found!")

        mode_counts = np.zeros(1000, dtype=int)

    if is_main_process():
        print(f"Training for {epochs} epochs (warm-up: {warmup_epochs} epochs, burn-in: {burn_in_epochs} epochs)")
    
    for epoch in range(epochs):
        if isinstance(dataloader.sampler, torch.utils.data.DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        if img_type == 'd1':
            mode_counts.fill(0)
            total_samples= 0
                
        epoch_D_loss = 0.0 # one epoch - Discriminator loss
        epoch_G_loss = 0.0 # one epoch - Generator loss 
        
        if is_main_process():
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        else:
            pbar = dataloader

        for i, batch in enumerate(pbar):
            if img_type == 'd1':
                real_images, _ = batch
            else:
                real_images = batch

            batch_size = real_images.size(0) # the number of image in batch
            real_images = real_images.to(device)

            images_seen += batch_size
            # compute scheduling parameter
            if images_seen < warmup_images:
                # linear warm-up
                w = images_seen / warmup_images
                beta2 = beta2_start + w * (beta2_end - beta2_start)
                gamma = gamma_start + w * (gamma_end - gamma_start)
                current_ema_hl_mimg = ema_start + w * (ema_end - ema_start)
            else:
                # cosine decay after warm-up
                t = min(images_seen, burn_in_images) - warmup_images
                t /= (burn_in_images - warmup_images)
                beta2 = beta2_start + 0.5*(beta2_end - beta2_start)*(1 - math.cos(math.pi * t))
                gamma = gamma_start + 0.5*(gamma_end - gamma_start)*(1 - math.cos(math.pi * t))
                current_ema_hl_mimg = ema_start + 0.5*(ema_end - ema_start)*(1 - math.cos(math.pi * t))

            # update optimizer betas
            for opt in (optimizer_G, optimizer_D):
                for pg in opt.param_groups:
                    pg['betas'] = (0.5, beta2)

            # compute ema decay
            H = max(current_ema_hl_mimg * 1_000_000, 1e-6)
            ema_decay = math.exp(-math.log(2) * batch_size / H) if H > 0 else 0.0
                
            calc_penalty = (i % lazy_interval == 0)
            
            #######################
            # Discriminator Update
            #######################

            discriminator.zero_grad()
            noise = torch.randn(batch_size, nz, device=device)

            generator.eval() 
            with torch.no_grad():
                fake_images = generator(noise)
            generator.train()

            if img_type == 'd1' and clf is not None:       
                with torch.no_grad():
                    logits = clf(fake_images)
                    probs  = F.softmax(logits, dim=1).cpu().numpy()
                pred_classes = probs.argmax(axis=1)
                pred_maxvals = probs.max(axis=1)
                for c, conf in zip(pred_classes, pred_maxvals):
                    if conf > 0.5:
                        mode_counts[c] += 1
                        total_samples += 1
            
            # Discriminator loss
            if calc_penalty:
                r1_val = r1_penalty(discriminator, real_images)
                r2_val = r2_penalty(discriminator, fake_images)
                penalty_r1 = r1_val * lazy_interval
                penalty_r2 = r2_val * lazy_interval

                r1_penalty_list.append(r1_val.item())
                r2_penalty_list.append(r2_val.item())
            else:
                penalty_r1 = penalty_r2 = 0.0     # float
            # --------- (switch_loss 분기 단순화) ----------
            if epoch < switch_epoch or not switch_loss:
                d_loss = discriminator_rploss(
                    discriminator, real_images, fake_images,
                    gamma, penalty_r1, penalty_r2)
            else:
                d_loss = discriminator_hinge_rploss(
                    discriminator, real_images, fake_images,
                    gamma, penalty_r1, penalty_r2)



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
                   p_ema.data.mul_(ema_decay).add_(p.detach().data * (1.0 - ema_decay))
            
            # Loss
            epoch_D_loss += d_loss.item()
            epoch_G_loss += g_loss.item()
            
            if is_main_process() and hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })
            
        
    
        avg_D_loss = epoch_D_loss / len(dataloader)
        avg_G_loss = epoch_G_loss / len(dataloader)

    
        if dist.is_initialized() and dist.get_world_size() > 1:
            d_tensor = torch.tensor(avg_D_loss, device=device)
            g_tensor = torch.tensor(avg_G_loss, device=device)
            dist.all_reduce(d_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(g_tensor, op=dist.ReduceOp.SUM)
            ws = dist.get_world_size()
            avg_D_loss = d_tensor.detach().item() / ws
            avg_G_loss = g_tensor.detach().item() / ws

        D_losses.append(avg_D_loss)
        G_losses.append(avg_G_loss)
        
        # Save generated image
        if (epoch + 1) % 1 == 0 and is_main_process():
            with torch.no_grad():
                ema_generator.eval()
                fake_samples = ema_generator(fixed_noise).cpu()
                

                plt.figure(figsize=(12, 12))
                plt.axis("off")
                plt.title(f"Generated Images - Epoch {epoch+1}")
                for idx in range(16):
                    plt.subplot(4, 4, idx + 1)
                    plt.imshow(fake_samples[idx].permute(1, 2, 0) * 0.5 + 0.5)
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig(f'./results/generated_epoch_{epoch+1}.png', dpi=150, bbox_inches='tight')
                plt.close()
        
        # Save check point
        if (epoch + 1) % 5 == 0 and is_main_process():
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
            if total_samples > 0:
                counts = mode_counts  # shape (1000,)
                total  = total_samples
                p = counts / total            

                q = np.full_like(p, 1 / p.size)
                coverage = int((counts > 0).sum())

                p_theta = np.clip(p, 1e-12, None)

                rev_kl = float((p_theta * np.log(p_theta / q)).sum())
            else:
                coverage = 0
                rev_kl = 0.0
            if is_main_process():
                print(f"[Epoch {epoch+1}] Mode Coverage: {coverage}/1000, Reverse KL: {rev_kl:.6f}")
        
        # fid scoring
        fid_value = fid_scoring(epoch,
        fid_every, 
        ema_generator, 
        discriminator, 
        fid_real_subset, 
        fid_batch_size,
        fid_real_indices,
        fixed_fid_noise,
        img_type,
        device)

        #eval 모드에서 다시 train 모드로
        generator.train()
        discriminator.train()
        
        if is_main_process():
            print(f'Epoch [{epoch+1}/{epochs}] - D_loss: {avg_D_loss:.4f}, G_loss: {avg_G_loss:.4f}')
            print(f"Total NFE in Epoch {epoch+1}: {nfe_tracker.get_nfe()}")
        
        Fid_list.append(fid_value)
        Mode_coverage_list.append(coverage if img_type == 'd1' else None)
        KL_divergence_list.append(rev_kl if img_type == 'd1' else None)

        if logger:
            if img_type == 'd1':
                logger.logd1(epoch, avg_G_loss, avg_D_loss, fid_value, coverage, rev_kl)
            else:
                logger.log(epoch, avg_G_loss, avg_D_loss, fid_value)
    if is_main_process():
        metrics = {
            'G_losses': G_losses,
            'D_losses': D_losses,
            'FID': Fid_list,
            'Mode_coverage': Mode_coverage_list,
            'KL_divergence': KL_divergence_list,
            'R1_penalty': r1_penalty_list,
            'R2_penalty': r2_penalty_list,
        }
    
        # 각 리스트의 길이가 다를 수 있으니 pandas.Series 로 변환
        df_metrics = pd.DataFrame({k: pd.Series(v) for k, v in metrics.items()})
        df_metrics.to_csv('./results/training_metrics.csv', index=False)

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
            plt.title("Reverse KL Divergence During Training")
            plt.plot(KL_divergence_list, label="Reverse KL Divergence")
            plt.xlabel("Epochs")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig('./results/training_kl.png')
            plt.close()
            
            plt.figure(figsize=(10,5))
            plt.plot(r1_penalty_list, label="R1 penalty")
            plt.plot(r2_penalty_list, label="R2 penalty")
            plt.xlabel("Iteration")
            plt.ylabel("Penalty value")
            plt.legend()
            plt.tight_layout()
            plt.savefig("./results/penalties.png")
            plt.close()
    
        print("Finished Training!")
        if logger:
            logger.log_final(epochs, avg_G_loss, avg_D_loss, fid_value)


if __name__ == "__main__":

    '''
    img_type
    dataset 1 : Stacked MNIST
    dataset 2 : FFHQ-64
    dataset 3 : CIFAR-10
    dataset 4 : ImageNet-32
    '''


    #DDP 초기화
    if 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        is_distributed = True    
        if is_main_process():
            print(f"Initialized DDP on rank {dist.get_rank()}/{dist.get_world_size()}")
    else:                                            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0                            
        is_distributed = False                      
        print(f"Running on single device: {device}")

   
    img_type = 'd1'
    batch_size = 128
    max_images = 128000
 

    if img_type == 'd1':
       
        dataloader = load_data_StackMNIST(
            batch_size, "./data/mnist", max_images)
        dataset = dataloader.dataset

        # if is_main_process():
        #     print("Stacked MNIST data loading...")
        # img_dir = "./data/mnist"
        # dataloader = load_data_StackMNIST(batch_size, img_dir, max_images=max_images)

        gen_base_channels = [256, 256, 256, 256]        # for 32x32 output
        # gen_base_channels = [128, 128, 128, 128]


        img_name = 'Stacked MNIST'
        lr = 0.0002
        clf_epochs = 30


    elif img_type == 'd2' : 
        if is_main_process():
            print("FFHQ64 data loading...")
        tfrecord_dir = "./data/ffhq64"  # Point to the tfrecord directory
        dataloader = load_data_ffhq64(batch_size, max_images=max_images)
        dataset = dataloader.dataset
        gen_base_channels = [128, 256, 256, 256, 256]


        img_name = 'FFHQ-64'
        lr = 0.0002
        


    elif img_type == 'd3' : 
        if is_main_process():
            print("cifar-10 data loading...")
        img_dir = "./data/cifar-10"
        dataloader = load_data_cifar10(batch_size, img_dir, max_images=max_images)
        dataset = dataloader.dataset
        gen_base_channels = [256, 128, 64, 32]        # for 32x32 output 

        img_name = 'CIFAR-10'
        lr = 0.0002
        




    elif img_type == 'd4':
        if is_main_process():
            print("ImageNet-32 data loading...")
        img_dir = "./data/imagenet32"
        dataloader = load_data_imagenet32(batch_size, img_dir, max_images=max_images)
        dataset = dataloader.dataset
        gen_base_channels = [1536, 1536, 1536, 1536]        # for 32x32 output 

        img_name = 'ImageNet-32'
        lr = 0.0002

    else : print('data type error!')

    if is_distributed and img_type != 'd1':
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=True)

    if img_type == 'd1' and is_main_process():
        train_classifier(dataloader, clf_epochs, lr, device)


    dataset_size = len(dataloader.dataset)
    duration_images = duration_mimg * 1_000_000
    epochs = math.ceil(duration_images / dataset_size)

    if is_main_process():
        print("############################### model generating ###############################")
        print(f"max images : {max_images} --- dataset size: {dataset_size}")

    # ───────── 모델 생성 ─────────
    G = Generator(BaseChannels=gen_base_channels).to(device)
    D = Discriminator(BaseChannels=list(reversed(gen_base_channels))).to(device)

    if is_distributed:
        G = DDP(G, device_ids=[local_rank], output_device=local_rank)
        D = DDP(D, device_ids=[local_rank], output_device=local_rank)

    def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    if is_main_process():
        gm = G.module if hasattr(G, "module") else G
        dm = D.module if hasattr(D, "module") else D
        print(f"Generator parameter : {count_parameters(gm):,}")
        print(f"Discriminator parameter : {count_parameters(dm):,}")
        print(f'Using device : {device}')
        print(f'epoch : {epochs}')
        print(f'batch size : {batch_size}')
        print(f'learning rate : {lr}')

    # ───────── 학습 ─────────

    train(
        generator      = G,
        discriminator  = D,
        dataloader     = dataloader,
        img_type       = img_type,
        img_name       = img_name,
        epochs         = epochs,
        lr             = lr,
        device         = device,
        switch_loss    = False,
        switch_epoch   = int(epochs // 2),
        fid_batch_size = batch_size,
        fid_num_images = 12_800,
        fid_every      = 1
    )



















    # dataset_size    = len(dataloader.dataset) 
    # epochs = math.ceil(duration_images / dataset_size)

    # # Discriminator channels: reverse of generator
    # disc_base_channels = list(reversed(gen_base_channels))
    
    # if is_main_process():
    #     print("############################### model generating ###############################")
    #     print(f"max images : {max_images} --- dataset size: {len(dataloader.dataset)}")


    # G = Generator(BaseChannels=gen_base_channels).to(device)
    # D = Discriminator(BaseChannels=disc_base_channels).to(device)
    
    # if 'WORLD_SIZE' in os.environ:
    #     G = DDP(G, device_ids=[local_rank], output_device=local_rank)
    #     D = DDP(D, device_ids=[local_rank], output_device=local_rank)

    # G.train()
    # D.train()
    
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # if is_main_process():
    #     g_model = G.module if hasattr(G, "module") else G
    #     d_model = D.module if hasattr(D, "module") else D
    #     print(f"Generator parameter: {count_parameters(g_model):,}")
    #     print(f"Discriminator parameter: {count_parameters(d_model):,}")
    #     print(f'Using device : {device}')
    #     print(f'epoch : {epochs}')
    #     print(f'batch size : {batch_size}')
    #     print(f'learning rate : {lr}')


    


    # train(G, D, dataloader, img_type, img_name, epochs, lr, device,
    #     switch_loss=False,
    #     switch_epoch=int(epochs/2),
    #     fid_batch_size=batch_size,
    #     fid_num_images=12800,
    #     fid_every=1)

