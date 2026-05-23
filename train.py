import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from data_prep import get_dataloaders
from models import CNNAutoencoder, AttentionUNetGenerator, CNNPatchDiscriminator

# ==========================================
# 1. Metric Calculation Functions
# ==========================================

def calculate_mse(img1, img2):
    """
    Computes Mean Squared Error.
    Images should be numpy arrays with shape (H, W, C) in range [0.0, 1.0].
    """
    return np.mean((img1 - img2) ** 2)

def calculate_psnr(img1, img2):
    """
    Computes Peak Signal-to-Noise Ratio.
    Images should be numpy arrays with shape (H, W, C) in range [0.0, 1.0].
    """
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(img1, img2):
    """
    Computes Structural Similarity Index (SSIM) between two images.
    Images should be numpy arrays in range [0.0, 1.0].
    This is a robust custom implementation using OpenCV Gaussian blur,
    avoiding the need for the scikit-image dependency.
    """
    # Standard constants
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2

    # Convert to grayscale if color
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        g1 = cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
        g2 = cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    else:
        g1 = img1.astype(np.float64)
        g2 = img2.astype(np.float64)

    kernel_size = 11
    sigma = 1.5

    # Means
    mu1 = cv2.GaussianBlur(g1, (kernel_size, kernel_size), sigma)
    mu2 = cv2.GaussianBlur(g2, (kernel_size, kernel_size), sigma)

    # Squares of means
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Variances and Covariance
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (kernel_size, kernel_size), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (kernel_size, kernel_size), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (kernel_size, kernel_size), sigma) - mu1_mu2

    # SSIM formula
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = num / (den + 1e-8)
    return np.mean(ssim_map)

# ==========================================
# 2. Evaluation / Validation Loop
# ==========================================

def evaluate_model(model, val_loader, device):
    model.eval()
    psnr_list, ssim_list, mse_list = [], [], []
    
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            enhanced = model(noisy)
            
            # Convert tensors to numpy arrays of shape (H, W, C) in range [0, 1]
            for i in range(noisy.size(0)):
                enh_np = enhanced[i].permute(1, 2, 0).cpu().numpy()
                cln_np = clean[i].permute(1, 2, 0).cpu().numpy()
                
                # Clip to secure [0, 1] range
                enh_np = np.clip(enh_np, 0.0, 1.0)
                cln_np = np.clip(cln_np, 0.0, 1.0)
                
                mse_list.append(calculate_mse(enh_np, cln_np))
                psnr_list.append(calculate_psnr(enh_np, cln_np))
                ssim_list.append(calculate_ssim(enh_np, cln_np))
                
    return np.mean(psnr_list), np.mean(ssim_list), np.mean(mse_list)

# ==========================================
# 3. Main Training Script
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Train ANN and TransGAN models for Low-Dose CT scan enhancement")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--noise_level", type=float, default=0.4, help="Low dose noise level (0.0 to 1.0)")
    parser.add_argument("--subset_limit", type=int, default=100, help="Limit dataset size for fast CPU demo training")
    parser.add_argument("--data_dir", type=str, default=r"d:\ANN Project\raw-890", help="Path to image dataset")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")
    
    # 1. Load Data
    print(f"Loading data from {args.data_dir}...")
    train_loader, val_loader, _, _ = get_dataloaders(
        image_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=(128, 128),
        noise_level=args.noise_level,
        subset_limit=args.subset_limit
    )
    
    # 2. Instantiate Models
    ann_model = CNNAutoencoder().to(device)
    
    # TransGAN components (Generator + Discriminator)
    transgan_gen = AttentionUNetGenerator().to(device)
    transgan_disc = CNNPatchDiscriminator().to(device)
    
    # 3. Define Optimizers & Losses
    ann_optimizer = optim.Adam(ann_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    gen_optimizer = optim.Adam(transgan_gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    disc_optimizer = optim.Adam(transgan_disc.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    criterion_reconstruction = nn.MSELoss() # L2 loss for baseline Autoencoder
    criterion_gan = nn.BCEWithLogitsLoss() # Adversarial loss for GAN
    criterion_l1 = nn.L1Loss() # Pixel level alignment loss for Generator
    
    best_ann_psnr = -1.0
    best_gen_psnr = -1.0
    
    print("\nStarting Training Execution...")
    for epoch in range(1, args.epochs + 1):
        ann_model.train()
        transgan_gen.train()
        transgan_disc.train()
        
        ann_epoch_loss = 0.0
        gen_epoch_loss = 0.0
        disc_epoch_loss = 0.0
        
        for batch_idx, (noisy, clean) in enumerate(train_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            batch_sz = noisy.size(0)
            
            # ------------------------------------------
            # Train ANN (CNN Autoencoder)
            # ------------------------------------------
            ann_optimizer.zero_grad()
            ann_output = ann_model(noisy)
            ann_loss = criterion_reconstruction(ann_output, clean)
            ann_loss.backward()
            ann_optimizer.step()
            ann_epoch_loss += ann_loss.item()
            
            # ------------------------------------------
            # Train TransGAN (Attention U-Net Generator + CNN Discriminator)
            # ------------------------------------------
            # (A) Train Discriminator
            disc_optimizer.zero_grad()
            
            # Real images classification
            disc_real_out = transgan_disc(clean)
            disc_real_labels = torch.ones_like(disc_real_out)
            loss_disc_real = criterion_gan(disc_real_out, disc_real_labels)
            
            # Fake images classification
            fake_clean = transgan_gen(noisy)
            disc_fake_out = transgan_disc(fake_clean.detach())
            disc_fake_labels = torch.zeros_like(disc_fake_out)
            loss_disc_fake = criterion_gan(disc_fake_out, disc_fake_labels)
            
            # Backprop Discriminator
            loss_disc = (loss_disc_real + loss_disc_fake) * 0.5
            loss_disc.backward()
            disc_optimizer.step()
            disc_epoch_loss += loss_disc.item()
            
            # (B) Train Generator
            gen_optimizer.zero_grad()
            
            # Adversarial Loss (wants Discriminator to believe fake is real)
            disc_fake_out_g = transgan_disc(fake_clean)
            loss_adv = criterion_gan(disc_fake_out_g, torch.ones_like(disc_fake_out_g))
            
            # Content / Reconstruction L1 Loss
            loss_content = criterion_l1(fake_clean, clean)
            
            # Combined Loss: adversarial + lambda * content (standard GAN translation training)
            loss_gen = loss_adv + 50.0 * loss_content
            loss_gen.backward()
            gen_optimizer.step()
            gen_epoch_loss += loss_gen.item()
            
        # End of Epoch Evaluation
        ann_epoch_loss /= len(train_loader)
        gen_epoch_loss /= len(train_loader)
        disc_epoch_loss /= len(train_loader)
        
        # Validation PSNR and SSIM
        ann_psnr, ann_ssim, ann_mse = evaluate_model(ann_model, val_loader, device)
        gen_psnr, gen_ssim, gen_mse = evaluate_model(transgan_gen, val_loader, device)
        
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        print(f"ANN Model   | Train Loss: {ann_epoch_loss:.5f} | Val PSNR: {ann_psnr:.2f} dB | Val SSIM: {ann_ssim:.4f}")
        print(f"TransGAN    | Gen Loss:   {gen_epoch_loss:.5f} | Val PSNR: {gen_psnr:.2f} dB | Val SSIM: {gen_ssim:.4f}")
        print(f"            | Disc Loss:  {disc_epoch_loss:.5f}")
        
        # Save checkpoints
        if ann_psnr > best_ann_psnr:
            best_ann_psnr = ann_psnr
            torch.save(ann_model.state_dict(), "ann_model.pth")
            print("=> Saved best ANN model weights!")
            
        if gen_psnr > best_gen_psnr:
            best_gen_psnr = gen_psnr
            torch.save(transgan_gen.state_dict(), "transgan_model.pth")
            print("=> Saved best TransGAN Generator weights!")
            
    print("\nTraining completed successfully!")
    print(f"Best ANN PSNR: {best_ann_psnr:.2f} dB")
    print(f"Best TransGAN PSNR: {best_gen_psnr:.2f} dB")

if __name__ == "__main__":
    main()
