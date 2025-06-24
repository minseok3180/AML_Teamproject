import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pytorch_fid import fid_score
import torch.distributed as dist

class NFETracker:
    def __init__(self):
        self.nfe_count = 0  # 함수 평가 횟수 초기화

    def increment(self):
        """함수 평가 횟수를 1 증가시킵니다."""
        self.nfe_count += 1

    def get_nfe(self):
        return self.nfe_count

def nfe_scoring(model, data, loss_fn, optimizer, nfe_tracker):
    model.train()
    optimizer.zero_grad()

    predictions = model(data)
    loss = loss_fn(predictions)

    

    loss.backward()
    optimizer.step()

    return loss.item()


def fid_scoring(epoch,
        fid_every, 
        generator, 
        discriminator, 
        fid_real_subset, 
        fid_batch_size,
        fid_real_indices,
        fixed_fid_noise,
        img_type,
        device):

    if (epoch + 1) % fid_every == 0 and dist.get_rank() == 0:
                torch.cuda.empty_cache()

                generator.eval()
                discriminator.eval()

                fid_root = "./results/fid"
                real_dir = os.path.join(fid_root, "real")
                fake_dir = os.path.join(fid_root, "fake")

                if os.path.exists(fid_root):
                    shutil.rmtree(fid_root)
                os.makedirs(real_dir)
                os.makedirs(fake_dir)

                real_loader_for_fid = DataLoader(
                    fid_real_subset,
                    batch_size=fid_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    drop_last=False
                )
                
                real_idx = 0
                for real_batch in real_loader_for_fid:

                    if img_type == 'd1':
                        img_batch = real_batch[0]
                    else:
                        img_batch = real_batch

                    for img in img_batch:
                        img_01 = (img + 1) / 2.0
                        pil_img = TF.to_pil_image(img_01)
                        pil_img.save(os.path.join(real_dir, f"real_{real_idx:05d}.png"))
                        real_idx += 1

                fake_idx = 0
                with torch.no_grad():
                    for start in range(0, len(fid_real_indices), fid_batch_size):
                        end = min(start + fid_batch_size, len(fid_real_indices))
                        noise_batch = fixed_fid_noise[start:end]
                        fake_batch = generator(noise_batch).cpu()

                        for img in fake_batch:
                            img_01 = (img + 1) / 2.0
                            pil_img = TF.to_pil_image(img_01)
                            pil_img.save(os.path.join(fake_dir, f"fake_{fake_idx:05d}.png"))
                            fake_idx += 1

                        del fake_batch, noise_batch
                        torch.cuda.empty_cache()

                torch.cuda.empty_cache()

                paths = [real_dir, fake_dir]
                fid_value = fid_score.calculate_fid_given_paths(
                    paths, batch_size=fid_batch_size, device=device, dims=2048
                )
            
                print(f"Epoch {epoch+1:03d} | FID: {fid_value:.4f}")

                shutil.rmtree(real_dir, ignore_errors=True)
                shutil.rmtree(fake_dir, ignore_errors=True)

                return fid_value

    return None
