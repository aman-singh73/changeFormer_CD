# fine_tune_custom.py - Save this as a separate file
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
import rasterio
from PIL import Image
from torchvision import transforms
import cv2
from models.ChangeFormer import ChangeFormerV6
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings('ignore')

class CustomInfrastructureDataset(Dataset):
    def __init__(self, data_root, split='train', img_size=256, augment=True):
        self.img_size = img_size
        self.augment = augment and split == 'train'
        
        # Adjust paths based on your data structure
        self.img_A_paths = sorted(glob.glob(f"{data_root}/A/*.tif"))
        self.img_B_paths = sorted(glob.glob(f"{data_root}/B/*.tif"))
        self.label_paths = sorted(glob.glob(f"{data_root}/label/*.tif"))
        
        assert len(self.img_A_paths) == len(self.img_B_paths) == len(self.label_paths), \
            "Mismatch in number of files!"
        
        print(f"Found {len(self.img_A_paths)} samples for {split}")
        
        # OSCD-style normalization (CRITICAL!)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_preprocess_image(self, img_path, bands=[4, 3, 2]):
        with rasterio.open(img_path) as src:
            img_data = []
            for band_idx in bands:
                if band_idx <= src.count:
                    band = src.read(band_idx).astype(np.float32)
                    band = np.nan_to_num(band, nan=0.0)
                    img_data.append(band)
            img = np.stack(img_data, axis=-1)
        
        # Robust preprocessing (same as inference)
        p1, p99 = np.percentile(img, [1, 99])
        img_clipped = np.clip(img, p1, p99)
        
        # Per-channel normalization
        img_normalized = np.zeros_like(img_clipped)
        for c in range(img.shape[2]):
            channel = img_clipped[:, :, c]
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                img_normalized[:, :, c] = (channel - channel_min) / (channel_max - channel_min)
            else:
                img_normalized[:, :, c] = 0.5
        
        # Convert to uint8 for PIL
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        return img_uint8
    
    def __len__(self):
        return len(self.img_A_paths)
    
    def __getitem__(self, idx):
        # Load images
        img_A = self.load_and_preprocess_image(self.img_A_paths[idx])
        img_B = self.load_and_preprocess_image(self.img_B_paths[idx])
        
        # Load label
        with rasterio.open(self.label_paths[idx]) as src:
            label = src.read(1).astype(np.uint8)
        
        # Random augmentation for training
        if self.augment and np.random.random() < 0.5:
            if np.random.random() < 0.5:  # Horizontal flip
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
                label = np.fliplr(label)
            
            if np.random.random() < 0.5:  # Vertical flip
                img_A = np.flipud(img_A)
                img_B = np.flipud(img_B)
                label = np.flipud(label)
            
            if np.random.random() < 0.3:  # 90-degree rotation
                k = np.random.choice([1, 2, 3])
                img_A = np.rot90(img_A, k=k)
                img_B = np.rot90(img_B, k=k)
                label = np.rot90(label, k=k)
        
        # Convert to PIL and resize
        img_A_pil = Image.fromarray(img_A).resize((self.img_size, self.img_size), Image.LANCZOS)
        img_B_pil = Image.fromarray(img_B).resize((self.img_size, self.img_size), Image.LANCZOS)
        label_pil = Image.fromarray(label).resize((self.img_size, self.img_size), Image.NEAREST)
        
        # Apply transforms
        img_A_tensor = self.transform(img_A_pil)
        img_B_tensor = self.transform(img_B_pil)
        label_tensor = torch.from_numpy(np.array(label_pil) > 0).long()
        
        return img_A_tensor, img_B_tensor, label_tensor

class FocalLoss(nn.Module):
    """Focal Loss for handling extreme class imbalance in change detection"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def calculate_class_weights(dataset):
    """Calculate class weights for imbalanced data"""
    total_pixels = 0
    positive_pixels = 0
    
    for i in range(len(dataset)):
        _, _, label = dataset[i]
        total_pixels += label.numel()
        positive_pixels += label.sum().item()
    
    if positive_pixels == 0:
        return torch.tensor([1.0, 1.0])
    
    neg_weight = 1.0
    pos_weight = (total_pixels - positive_pixels) / positive_pixels
    # Limit extreme weights
    pos_weight = min(pos_weight, 50.0)
    
    print(f"Class weights: [1.0, {pos_weight:.1f}] (positive class weight)")
    return torch.tensor([neg_weight, pos_weight])

def progressive_fine_tune(data_root, pretrained_path, output_dir="./fine_tuned_models", 
                         epochs=100, batch_size=8):
    """
    Progressive fine-tuning: Phase 1 (head only) + Phase 2 (full model)
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    train_dataset = CustomInfrastructureDataset(data_root, split='train', augment=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2 if torch.cuda.is_available() else 0)
    
    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # Load pretrained model
    print("Loading pretrained ChangeFormer...")
    model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(device)
    checkpoint = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(checkpoint['model_G_state_dict'], strict=True)
    
    # ============ PHASE 1: HEAD-ONLY TRAINING ============
    print("\n" + "="*50)
    print("PHASE 1: Training classification head only (20 epochs)")
    print("="*50)
    
    # Freeze all parameters except final layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze final classification layers (adjust based on your model architecture)
    # You might need to check your model structure and adjust these layer names
    trainable_layers = []
    for name, module in model.named_modules():
        if 'final' in name or 'classifier' in name or 'head' in name or 'output' in name:
            for param in module.parameters():
                param.requires_grad = True
            trainable_layers.append(name)
    
    # If no specific layers found, unfreeze last few layers
    if not trainable_layers:
        print("Warning: No final layers found, unfreezing last 2 modules")
        modules = list(model.modules())
        for module in modules[-2:]:
            for param in module.parameters():
                param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")
    
    # Phase 1 optimizer and loss
    optimizer_phase1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=1e-3, weight_decay=1e-4)
    criterion = FocalLoss(alpha=2.0, gamma=2.0)  # Focal loss for imbalanced data
    scheduler_phase1 = CosineAnnealingLR(optimizer_phase1, T_max=20)
    
    # Phase 1 training loop
    phase1_losses = []
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (img_A, img_B, labels) in enumerate(train_loader):
            img_A, img_B, labels = img_A.to(device), img_B.to(device), labels.to(device)
            
            optimizer_phase1.zero_grad()
            
            # Forward pass
            outputs = model(img_A, img_B)
            if isinstance(outputs, list):
                outputs = outputs[-1]  # Take final output
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer_phase1.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Phase 1 Epoch {epoch+1}/20, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        phase1_losses.append(avg_loss)
        scheduler_phase1.step()
        
        print(f"Phase 1 Epoch {epoch+1}/20 completed. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_G_state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss
            }, f"{output_dir}/phase1_epoch_{epoch+1}.pt")
    
    # ============ PHASE 2: FULL MODEL FINE-TUNING ============
    print("\n" + "="*50)
    print("PHASE 2: Fine-tuning entire model (40 epochs)")
    print("="*50)
    
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True
    
    print(f"All {total_params:,} parameters are now trainable")
    
    # Phase 2 optimizer with lower learning rate
    optimizer_phase2 = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler_phase2 = CosineAnnealingLR(optimizer_phase2, T_max=40)
    
    # Phase 2 training loop
    phase2_losses = []
    best_loss = float('inf')
    
    for epoch in range(40):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (img_A, img_B, labels) in enumerate(train_loader):
            img_A, img_B, labels = img_A.to(device), img_B.to(device), labels.to(device)
            
            optimizer_phase2.zero_grad()
            
            outputs = model(img_A, img_B)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer_phase2.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Phase 2 Epoch {epoch+1}/40, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        phase2_losses.append(avg_loss)
        scheduler_phase2.step()
        
        print(f"Phase 2 Epoch {epoch+1}/40 completed. Avg Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_G_state_dict': model.state_dict(),
                'epoch': epoch + 21,
                'loss': avg_loss
            }, f"{output_dir}/best_custom_model.pt")
            print(f"New best model saved! Loss: {avg_loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_G_state_dict': model.state_dict(),
                'epoch': epoch + 21,
                'loss': avg_loss
            }, f"{output_dir}/phase2_epoch_{epoch+21}.pt")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 21), phase1_losses, 'b-', label='Phase 1 (Head Only)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase 1: Head-Only Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(21, 61), phase2_losses, 'r-', label='Phase 2 (Full Model)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Phase 2: Full Model Fine-tuning')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=200)
    plt.close()
    
    print(f"\n" + "="*50)
    print("FINE-TUNING COMPLETED!")
    print(f"Best model saved as: {output_dir}/best_custom_model.pt")
    print(f"Final loss: {best_loss:.4f}")
    print("="*50)
    
    return model

if __name__ == "__main__":
    # USAGE EXAMPLE
    data_root = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom" 
    pretrained_path = r"D:\Aman_kr_Singh\NEW_OSCD\best_ckpt.pt" 
    
    # Create output directory
    import os
    output_dir = "./fine_tuned_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Start fine-tuning
    print("Starting Progressive Fine-tuning...")
    print(f"Data root: {data_root}")
    print(f"Pretrained model: {pretrained_path}")
    print(f"Output directory: {output_dir}")
    
    model = progressive_fine_tune(
        data_root=data_root,
        pretrained_path=pretrained_path,
        output_dir=output_dir,
        epochs=100,
        batch_size=8 
    )
    
    print("\nFine-tuning complete! Use 'best_custom_model.pt' for inference.")