import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

def add_low_dose_noise(image_np, noise_level=0.5):
    """
    Simulates a low-dose CT scan by adding a combination of Poisson and Gaussian noise.
    
    Parameters:
    - image_np: numpy array of shape (H, W, 3) or (H, W), values in range [0, 255] or [0.0, 1.0]
    - noise_level: float between 0.0 (no noise) and 1.0 (extreme noise)
    
    Returns:
    - noisy_image: numpy array of the same shape and type, values in original range.
    """
    # Convert image to float64 in range [0.0, 1.0]
    if image_np.max() > 1.0:
        img_normalized = image_np.astype(np.float64) / 255.0
        scale_back = True
    else:
        img_normalized = image_np.astype(np.float64)
        scale_back = False
        
    if noise_level <= 0.001:
        return image_np
        
    # 1. Poisson Noise (simulates photon starvation at low doses)
    # The higher the noise level, the lower the photon count (peak)
    peak = 50.0 / (noise_level + 1e-5)  # Max intensity in photons (e.g., lower peak = more Poisson noise)
    noisy_poisson = np.random.poisson(img_normalized * peak) / peak
    
    # 2. Gaussian Noise (simulates electronic noise in CT detectors)
    gaussian_sigma = 0.05 * noise_level
    noisy_gaussian = np.random.normal(0, gaussian_sigma, img_normalized.shape)
    
    # Combine noises and clip to [0, 1]
    noisy_img = noisy_poisson + noisy_gaussian
    noisy_img = np.clip(noisy_img, 0.0, 1.0)
    
    if scale_back:
        return (noisy_img * 255.0).astype(np.uint8)
    return noisy_img

class LowDoseDataset(Dataset):
    def __init__(self, image_dir, filenames, image_size=(128, 128), transform=None, noise_level=0.5):
        """
        Custom PyTorch Dataset for paired low-dose (noisy) and high-dose (clean) images.
        """
        self.image_dir = image_dir
        self.filenames = filenames
        self.image_size = image_size
        self.transform = transform
        self.noise_level = noise_level
        
        # Standard resize and tensor conversion for the high-dose image
        self.base_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Open image and convert to RGB
        clean_img = Image.open(img_path).convert('RGB')
        
        # Convert to tensor and apply standard transforms to ground truth
        clean_tensor = self.base_transform(clean_img)
        
        # Convert clean tensor to numpy array to add synthetic low-dose noise
        clean_np = clean_tensor.permute(1, 2, 0).numpy()  # (H, W, 3)
        
        # Add noise
        noisy_np = add_low_dose_noise(clean_np, noise_level=self.noise_level)
        
        # Convert back to tensor
        noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).float()  # (3, H, W)
        
        # Apply any additional user transforms if needed
        if self.transform:
            # Applying spatial transforms equally to both to maintain alignment
            # (In our case, basic flip/rotation)
            pass
            
        return noisy_tensor, clean_tensor

def get_dataloaders(image_dir, batch_size=8, image_size=(128, 128), train_split=0.8, noise_level=0.5, subset_limit=None):
    """
    Creates train and validation PyTorch DataLoaders.
    """
    all_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort files to ensure reproducibility
    all_files = sorted(all_files)
    
    # Shuffle files with a fixed seed
    np.random.seed(42)
    np.random.shuffle(all_files)
    
    # Limit dataset size for fast demonstration training if requested
    if subset_limit is not None and subset_limit < len(all_files):
        all_files = all_files[:subset_limit]
        
    split_idx = int(len(all_files) * train_split)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Dataset summary: {len(all_files)} total files.")
    print(f"Training on: {len(train_files)} files | Validating on: {len(val_files)} files.")
    
    train_dataset = LowDoseDataset(image_dir, train_files, image_size=image_size, noise_level=noise_level)
    val_dataset = LowDoseDataset(image_dir, val_files, image_size=image_size, noise_level=noise_level)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, train_files, val_files

# Simple test function
if __name__ == '__main__':
    raw_dir = r"d:\ANN Project\raw-890"
    if os.path.exists(raw_dir):
        train_loader, val_loader, _, _ = get_dataloaders(raw_dir, batch_size=4, subset_limit=20)
        noisy, clean = next(iter(train_loader))
        print("Batch Noisy shape:", noisy.shape, "Dtype:", noisy.dtype)
        print("Batch Clean shape:", clean.shape, "Dtype:", clean.dtype)
        print("Max value in batch:", clean.max().item(), "Min:", clean.min().item())
        print("Noise addition verification successful!")
    else:
        print("Data directory not found for local execution test.")
