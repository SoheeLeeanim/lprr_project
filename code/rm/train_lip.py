import os
import sys

# Add project root to sys.path to allow running as script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils
from code.rm.datasets.patch_dataset import LipPatchDataset
from code.rm.models.autoencoder import RM_AutoEncoder
import random


# -----------------------
# SSIM (simple implementation)
# -----------------------
def _gaussian(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g

def _create_window(window_size=11, channel=3, device="cpu"):
    g = _gaussian(window_size, 1.5, device=device).unsqueeze(1)
    window_2d = g @ g.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # (1,1,ws,ws)
    window = window_2d.repeat(channel, 1, 1, 1)      # (C,1,ws,ws)
    return window

def ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    :param img1, img2: Tensors of shape (B, C, H, W) with values in [0, 1].
    :return: Mean SSIM value over the batch.
    """
    assert img1.shape == img2.shape
    B, C, H, W = img1.shape
    device = img1.device
    window = _create_window(window_size, C, device=device)

    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size//2, groups=C)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size//2, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size//2, groups=C) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size//2, groups=C) - mu2_sq
    sigma12   = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size//2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# -----------------------
# Train
# -----------------------
def main():
    # ====== directory setting ======
    # e.g., data/source/patch_lip, data/target/patch_lip
    source_root = os.path.join("data", "source", "patch_lip") 
    target_root = os.path.join("data", "target", "patch_lip")
    source_base = os.path.join("data", "source")
    target_base = os.path.join("data", "target")

    frame_offset_target = -10

    # ====== Hyperparameters ======
    batch_size = 16
    num_workers = 0 # set to 0 for Windows compatibility
    lr = 1e-4
    epochs = 10
 
    # loss weights
    w_l1 = 1.0
    w_ssim = 1.0

    device = "cuda"
    print("[Device]", device)

    # ====== Dataset / Loader ======
    ds = LipPatchDataset(source_root, target_root, frame_offset_target=frame_offset_target)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    print("[Dataset length]", len(ds))

    # ====== Model ======
    model = RM_AutoEncoder(in_ch=3, base_ch=64, latent_ch=256).to(device)

    # ====== Optimizer ======
    opt = optim.Adam(model.parameters(), lr=lr)

    # L1 Loss
    l1 = nn.L1Loss()

    #  Create directories for saving checkpoints and samples
    ckpt_dir = os.path.join("checkpoints", "rm_lip")
    os.makedirs(ckpt_dir, exist_ok=True)
    sample_dir = os.path.join("samples", "rm_lip")
    os.makedirs(sample_dir, exist_ok=True)


    # ====== Train Loop ======
    model.train()
    global_step = 0

    for epoch in range(1, epochs + 1):
        running = 0.0

        pbar = tqdm(dl, desc=f"Epoch {epoch}/{epochs}", total=len(dl))
        for (src, tgt) in pbar: 
            src = src.to(device)  # Shape: (B, 3, 128, 128)
            tgt = tgt.to(device)

            src_rec, tgt_pred = model(src)
            loss_src = w_l1 * l1(src_rec, src) + w_ssim * (1.0 - ssim(src_rec, src))
            loss_tgt = w_l1 * l1(tgt_pred, tgt) + w_ssim * (1.0 - ssim(tgt_pred, tgt))

            loss = loss_src + loss_tgt

            opt.zero_grad()
            loss.backward()
            opt.step()


            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(running / (pbar.n + 1)):.4f}")

            global_step += 1


        avg = running / max(1, len(dl))
        print(f"[Epoch {epoch}/{epochs}] loss={avg:.6f}")

        # ---- Save sample images ----
        dl_sample = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        # ---- Save sample images (once per epoch): RANDOM ----
        model.eval()
        with torch.no_grad():
            rand_indices = random.sample(range(len(ds)), k=4)

            src_lips, tgt_lips = [], []

            for i in rand_indices:
                src_lip, tgt_lip = ds[i]
                src_lips.append(src_lip); tgt_lips.append(tgt_lip)

            src_lip = torch.stack(src_lips).to(device)
            tgt_lip = torch.stack(tgt_lips).to(device)
            src_rec_lip, tgt_pred_lip = model(src_lip)

            n = src_lip.size(0)
            out_lip = os.path.join(sample_dir, f"epoch_{epoch:03d}_lip_rand.png")
            vutils.save_image(torch.cat([src_lip, src_rec_lip, tgt_lip, tgt_pred_lip], dim=0), out_lip, nrow=n)

            print("  saved sample:", out_lip)

        model.train()

        
        # ---- Save checkpoint ----
        ckpt_path = os.path.join(ckpt_dir, f"rm_epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
        }, ckpt_path)
        print("  saved:", ckpt_path)

    print("Done.")


if __name__ == "__main__":
    main()
