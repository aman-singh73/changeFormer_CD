
import torch
import numpy as np
from datasets.CD_dataset import CDDataset
import torchvision.utils as utils
import matplotlib.pyplot as plt
import os
from PIL import Image

def create_dataset(args, split='train', is_train=True):
    """Universal dataset creation for both TIF and standard formats."""
    
    rgb_bands = getattr(args, 'rgb_bands', [3, 2, 1])
    normalize_method = getattr(args, 'normalize_method', 'adaptive_percentile')  # Changed default
    data_format = getattr(args, 'data_format', 'auto')
    
    # Auto-detect TIF if not explicitly set
    if data_format == 'auto' and hasattr(args, 'data_name'):
        if 'tif' in args.data_name.lower() or 'custom' in args.data_name.lower():
            data_format = 'tif'
    
    print(f"Creating dataset: split={split}, format={data_format}, rgb_bands={rgb_bands}")
    
    return CDDataset(
        root_dir=args.data_root,
        img_size=args.img_size,
        split=split,
        is_train=is_train,
        label_transform='binary',
        to_tensor=True,
        rgb_bands=rgb_bands,
        normalize_method=normalize_method,
        data_format=data_format
    )

def get_loaders(args):
    """Train/val loaders"""
    from datasets.CD_dataset import CDDataset
    from torch.utils.data import DataLoader

    print(f"Creating loaders with data_root: {args.data_root}")
    
    # Ensure integer parameters
    batch_size = int(args.batch_size)
    num_workers = int(args.num_workers)
    img_size = int(args.img_size)
    augment_factor = int(args.augment_factor)

    # Create training dataset
    print(f"Creating dataset: split=train, format={args.data_format}, rgb_bands={args.rgb_bands}")
    train_dataset = CDDataset(
        root_dir=args.data_root,
        img_size=img_size,
        split='train',
        is_train=True,
        rgb_bands=args.rgb_bands,
        normalize_method=args.normalize_method,
        data_format=args.data_format,
        augment_factor=augment_factor
    )
    
    # Create validation dataset
    print(f"Creating dataset: split=val, format={args.data_format}, rgb_bands={args.rgb_bands}")
    val_dataset = CDDataset(
        root_dir=args.data_root,
        img_size=img_size,
        split='val',
        is_train=False,
        rgb_bands=args.rgb_bands,
        normalize_method=args.normalize_method,
        data_format=args.data_format,
        augment_factor=1  # No augmentation for validation
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Created train loader with {len(train_dataset)} samples and val loader with {len(val_dataset)} samples")
    
    return {'train': train_loader, 'val': val_loader}

def get_test_loader(args):
    """Test loader"""
    test_dataset = create_dataset(args, split='test', is_train=False)
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False
    )

def make_numpy_grid(tensor_data, pad_value=0, padding=2, nrow=4):
    """
    Enhanced make_numpy_grid with better handling for different tensor types
    """
    if isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.detach().cpu()
        
        # Handle different tensor dimensions
        if tensor_data.dim() == 3:
            tensor_data = tensor_data.unsqueeze(0)  # Add batch dimension
        
        # Handle grayscale (single channel) tensors
        if tensor_data.dim() == 4 and tensor_data.size(1) == 1:
            # Replicate to 3 channels for visualization
            tensor_data = tensor_data.repeat(1, 3, 1, 1)
        
        # Create grid using torchvision
        vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding, nrow=nrow)
        vis = vis.cpu().numpy().transpose((1, 2, 0))
        
        # Handle grayscale output
        if vis.shape[2] == 1:
            vis = np.stack([vis[:,:,0], vis[:,:,0], vis[:,:,0]], axis=-1)
    else:
        # Handle numpy arrays
        if isinstance(tensor_data, np.ndarray):
            if len(tensor_data.shape) == 2:
                vis = np.stack([tensor_data, tensor_data, tensor_data], axis=-1)
            else:
                vis = tensor_data
        else:
            vis = tensor_data
    
    return np.clip(vis, 0, 1)

def de_norm(tensor_data):
    """
    Enhanced denormalization function
    Handles different normalization schemes
    """
    if isinstance(tensor_data, torch.Tensor):
        tensor_data = tensor_data.clone()
        
        # Check the range to determine normalization type
        min_val = tensor_data.min().item()
        max_val = tensor_data.max().item()
        
        if min_val >= -1.1 and max_val <= 1.1:
            # Likely normalized to [-1, 1]
            tensor_data = tensor_data * 0.5 + 0.5
        elif min_val >= -0.1 and max_val <= 1.1:
            # Already in [0, 1] range
            pass
        else:
            # Unknown range, normalize to [0, 1]
            tensor_data = (tensor_data - min_val) / (max_val - min_val + 1e-8)
    
    return torch.clamp(tensor_data, 0, 1)

def get_device(args):
    """Enhanced device setup"""
    if hasattr(args, 'gpu_ids') and isinstance(args.gpu_ids, str):
        str_ids = args.gpu_ids.split(',')
        args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpu_ids.append(id)
    
    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_ids[0])
        print(f"Using GPU: {args.gpu_ids}")
    else:
        print("Using CPU")

def save_change_detection_result(pred_tensor, gt_tensor, img_a, img_b, save_path):
    """Enhanced save function with better visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Convert tensors to numpy with enhanced denormalization
    if isinstance(img_a, torch.Tensor):
        img_a = de_norm(img_a).cpu().numpy().transpose(1, 2, 0)
    
    if isinstance(img_b, torch.Tensor):
        img_b = de_norm(img_b).cpu().numpy().transpose(1, 2, 0)
        
    if isinstance(gt_tensor, torch.Tensor):
        gt = gt_tensor.cpu().numpy()
        if gt.ndim == 3:
            gt = gt[0]  # Take first channel if needed
        
    if isinstance(pred_tensor, torch.Tensor):
        if pred_tensor.dim() > 2:
            pred = torch.argmax(pred_tensor, dim=0).cpu().numpy()
        else:
            pred = pred_tensor.cpu().numpy()
    
    # Plot images with better formatting
    axes[0, 0].imshow(img_a)
    axes[0, 0].set_title('Image A (Before)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_b)
    axes[0, 1].set_title('Image B (After)', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(gt, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 0].set_title('Ground Truth', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction', fontsize=12)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# =============== NEW ENHANCED FUNCTIONS ===============

def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization"""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def calculate_class_weights(dataloader):
    """Calculate class weights for imbalanced datasets"""
    total_pixels = 0
    change_pixels = 0
    
    print("Calculating class weights from dataloader...")
    sample_count = 0
    max_samples = 50  # Limit to avoid long computation
    
    for batch in dataloader:
        labels = batch['L']
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        
        total_pixels += labels.size
        change_pixels += np.sum(labels > 0.5)  # Threshold for binary
        
        sample_count += 1
        if sample_count >= max_samples:
            break
    
    no_change_pixels = total_pixels - change_pixels
    
    # Calculate weights
    change_ratio = change_pixels / total_pixels if total_pixels > 0 else 0.5
    no_change_ratio = no_change_pixels / total_pixels if total_pixels > 0 else 0.5
    
    print(f"Dataset statistics (from {sample_count} batches):")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Change pixels: {change_pixels:,} ({change_ratio:.2%})")
    print(f"  No-change pixels: {no_change_pixels:,} ({no_change_ratio:.2%})")
    
    # Inverse frequency weighting
    if no_change_ratio > 0 and change_ratio > 0:
        weights = [1.0 / no_change_ratio, 1.0 / change_ratio]
        # Normalize so that minority class weight is reasonable
        weights = [w / max(weights) for w in weights]
        weights = [max(w, 1.0) for w in weights]  # Ensure weights >= 1.0
    else:
        weights = [1.0, 1.0]
    
    print(f"  Calculated class weights: [no-change: {weights[0]:.2f}, change: {weights[1]:.2f}]")
    
    return weights

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth.tar', is_best=False):
    """Save checkpoint with proper error handling"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    
    try:
        torch.save(state, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        if is_best:
            best_filepath = os.path.join(checkpoint_dir, 'model_best.pth.tar')
            torch.save(state, best_filepath)
            print(f"Best model saved: {best_filepath}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint with proper error handling"""
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_G_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_G_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            if optimizer and 'optimizer_G_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])
            elif optimizer and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            epoch = checkpoint.get('epoch_id', checkpoint.get('epoch', 0))
            print(f"Loaded checkpoint '{checkpoint_path}' (epoch {epoch})")
            return checkpoint
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None
    else:
        print(f"No checkpoint found at '{checkpoint_path}'")
        return None

def create_training_summary(results_dir, train_acc, val_acc, train_losses=None):
    """Create comprehensive training summary plots"""
    os.makedirs(results_dir, exist_ok=True)
    
    epochs = range(1, len(train_acc) + 1)
    
    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    axes[0].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (mF1)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add best accuracy annotations
    best_val_idx = np.argmax(val_acc)
    axes[0].annotate(f'Best: {val_acc[best_val_idx]:.4f}',
                    xy=(best_val_idx + 1, val_acc[best_val_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Loss plot (if available)
    if train_losses is not None and len(train_losses) > 0:
        loss_epochs = range(1, len(train_losses) + 1)
        axes[1].plot(loss_epochs, train_losses, 'g-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Loss data not available', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        axes[1].set_title('Training Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training summary saved to {results_dir}/training_summary.png")

def visualize_predictions(images_A, images_B, predictions, ground_truth, save_path, 
                         max_samples=8, title_prefix=""):
    """Enhanced prediction visualization"""
    
    batch_size = min(max_samples, images_A.shape[0] if hasattr(images_A, 'shape') else len(images_A))
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, batch_size * 3))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        try:
            # Image A
            if isinstance(images_A, torch.Tensor):
                img_a = de_norm(images_A[i]).cpu().numpy().transpose(1, 2, 0)
            else:
                img_a = images_A[i]
            img_a = np.clip(img_a, 0, 1)
            axes[i, 0].imshow(img_a)
            axes[i, 0].set_title(f'Image A ({i+1})', fontsize=10)
            axes[i, 0].axis('off')
            
            # Image B
            if isinstance(images_B, torch.Tensor):
                img_b = de_norm(images_B[i]).cpu().numpy().transpose(1, 2, 0)
            else:
                img_b = images_B[i]
            img_b = np.clip(img_b, 0, 1)
            axes[i, 1].imshow(img_b)
            axes[i, 1].set_title(f'Image B ({i+1})', fontsize=10)
            axes[i, 1].axis('off')
            
            # Ground Truth
            if isinstance(ground_truth, torch.Tensor):
                gt = ground_truth[i].cpu().numpy()
                if gt.ndim > 2:
                    gt = gt[0]  # Take first channel if multi-channel
            else:
                gt = ground_truth[i]
            axes[i, 2].imshow(gt, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Ground Truth ({i+1})', fontsize=10)
            axes[i, 2].axis('off')
            
            # Prediction
            if isinstance(predictions, torch.Tensor):
                if predictions[i].dim() > 2:
                    pred = torch.argmax(predictions[i], dim=0).cpu().numpy()
                else:
                    pred = predictions[i].cpu().numpy()
            else:
                pred = predictions[i]
            axes[i, 3].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Prediction ({i+1})', fontsize=10)
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error visualizing sample {i}: {e}")
            # Fill with error message
            for j in range(4):
                axes[i, j].text(0.5, 0.5, f'Error\nprocessing\nsample {i}', 
                              ha='center', va='center', transform=axes[i, j].transAxes,
                              fontsize=8)
                axes[i, j].axis('off')
    
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=16)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def print_model_summary(model, input_shape=(3, 256, 256)):
    """Print model summary with parameter count"""
    try:
        # Try using torchsummary if available
        from torchsummary import summary
        print("="*50)
        print("MODEL SUMMARY")
        print("="*50)
        summary(model, input_shape)
    except ImportError:
        print("torchsummary not installed. Install with: pip install torchsummary")
        
        # Basic parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("="*50)
        print("MODEL SUMMARY")
        print("="*50)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("="*50)

def setup_logging(log_dir):
    """Setup logging directory structure"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create subdirectories
    subdirs = ['checkpoints', 'visualizations', 'results']
    for subdir in subdirs:
        os.makedirs(os.path.join(log_dir, subdir), exist_ok=True)
    
    print(f"Logging setup complete: {log_dir}")
    return log_dir



