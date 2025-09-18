import os
import torch
import numpy as np
from argparse import ArgumentParser
from models.trainer import CDTrainer
import utils
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime
import cv2
from PIL import Image
from scipy import ndimage
from skimage import morphology, measure, filters
import torch.nn.functional as F
import warnings
import logging
import rasterio
import time
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Tuple, Dict, Any, List
import shutil
import sys, io

# Force stdout/stderr to use UTF-8 (fixes charmap crashes on Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

logging.getLogger().setLevel(logging.ERROR)

class MultispectralHeavyAugmentationPipeline:
    """Heavy augmentation pipeline for multispectral Cartosat-3 imagery"""
    
    def __init__(self, augment_factor=5.0, n_bands=4, preserve_original=True):
        self.augment_factor = augment_factor
        self.n_bands = n_bands
        self.preserve_original = preserve_original
        
        # Geometric transformations (channel-agnostic)
        self.geometric_heavy = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.Transpose(p=1.0),
            ], p=0.9),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=8, alpha_affine=8, p=1.0),
                A.GridDistortion(num_steps=4, distort_limit=0.2, p=1.0),
                A.OpticalDistortion(distort_limit=0.15, shift_limit=0.08, p=1.0),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.4),
        ], additional_targets={'imageB': 'image', 'mask': 'mask'})
        
        # Intensity transformations (multispectral-safe)
        self.intensity_heavy = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
            A.RandomGamma(gamma_limit=(70, 130), p=0.6),
            A.ChannelShuffle(p=0.3),
        ], additional_targets={'imageB': 'image'})
        
        # Noise transformations (multispectral-safe)
        self.noise_heavy = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(8.0, 40.0), per_channel=True, p=1.0),
                A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True, p=1.0),
            ], p=0.7),
            A.GaussianBlur(blur_limit=3, p=0.3),
        ], additional_targets={'imageB': 'image'})
        
        # Pixel-level transformations (multispectral-safe)
        self.pixel_heavy = A.Compose([
            A.OneOf([
                A.Sharpen(alpha=(0.1, 0.4), lightness=(0.6, 1.0), p=1.0),
                A.Emboss(alpha=(0.1, 0.3), strength=(0.1, 0.5), p=1.0),
            ], p=0.5),
        ], additional_targets={'imageB': 'image'})
        
        # Mixed transformations
        self.mixed_heavy = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.GaussNoise(var_limit=(5.0, 20.0), per_channel=True, p=0.5),
        ], additional_targets={'imageB': 'image', 'mask': 'mask'})
        
        print(f"Multispectral augmentation pipeline initialized for {n_bands}-band imagery with {augment_factor}x factor")
    
    def get_augmentation_types(self):
        return ['geometric_heavy', 'intensity_heavy', 'noise_heavy', 'pixel_heavy', 'mixed_heavy']
    
    def apply_heavy_augmentation(self, image_a: np.ndarray, image_b: np.ndarray, 
                                mask: np.ndarray, aug_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply heavy augmentation of specified type for multi-spectral data"""
        
        # Ensure correct format - convert to uint8
        if image_a.dtype != np.uint8:
            image_a = np.clip(image_a * 255, 0, 255).astype(np.uint8) if image_a.max() <= 1.0 else np.clip(image_a, 0, 255).astype(np.uint8)
        if image_b.dtype != np.uint8:
            image_b = np.clip(image_b * 255, 0, 255).astype(np.uint8) if image_b.max() <= 1.0 else np.clip(image_b, 0, 255).astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)
        
        try:
            compose = getattr(self, aug_type)
            augmented = compose(image=image_a, imageB=image_b, mask=mask)
            return augmented['image'], augmented['imageB'], augmented['mask']
        except Exception as e:
            print(f"Heavy augmentation failed for type {aug_type}: {e}")
            return image_a.copy(), image_b.copy(), mask.copy()

class InMemoryCrossValidationManager:
    """Memory-efficient cross-validation manager that doesn't save fold data to disk"""
    
    def __init__(self, data_root: str, n_folds: int = 3, augment_factor: int = 5, 
                 random_state: int = 42, n_bands: int = 4):
        self.data_root = Path(data_root)
        self.n_folds = n_folds
        self.augment_factor = augment_factor
        self.random_state = random_state
        self.n_bands = n_bands
        self.cv_results = []
        
        # Read original file list
        train_list_path = self.data_root / 'list' / 'train.txt'
        with open(train_list_path, 'r') as f:
            self.original_files = [line.strip() for line in f.readlines()]
        
        print(f"InMemory CrossValidation Manager initialized:")
        print(f"  Original samples: {len(self.original_files)}")
        print(f"  Folds: {n_folds}")
        print(f"  Augmentation factor: {augment_factor}x")
        print(f"  Spectral bands: {n_bands}")
        print(f"  MEMORY EFFICIENT: No fold data saved to disk")
    
    def create_cv_folds(self):
        """Create cross-validation folds"""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.cv_folds = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.original_files)):
            train_files = [self.original_files[i] for i in train_idx]
            val_files = [self.original_files[i] for i in val_idx]
            
            self.cv_folds.append({
                'fold': fold_idx + 1,
                'train_files': train_files,
                'val_files': val_files,
                'train_count': len(train_files),
                'val_count': len(val_files)
            })
            
            print(f"Fold {fold_idx + 1}: {len(train_files)} train, {len(val_files)} val")
        
        return self.cv_folds
    
    def get_fold_data_info(self, fold_info: Dict, augment_factor: int = None):
        """Get fold data information without creating physical files"""
        if augment_factor is None:
            augment_factor = self.augment_factor
            
        fold_num = fold_info['fold']
        train_files = fold_info['train_files']
        val_files = fold_info['val_files']
        
        print(f"\nPreparing in-memory data for Fold {fold_num}...")
        print(f"Original training samples: {len(train_files)}")
        
        # Calculate target augmented samples
        target_samples = len(train_files) * augment_factor
        additional_needed = target_samples - len(train_files)
        
        print(f"Target samples: {target_samples}")
        print(f"Will create {additional_needed} augmentations in-memory during training")
        
        return {
            'fold_num': fold_num,
            'train_files': train_files,
            'val_files': val_files,
            'original_train_count': len(train_files),
            'val_count': len(val_files),
            'target_augmented_count': target_samples,
            'augmentations_needed': additional_needed
        }

class InMemoryAugmentedDataset(torch.utils.data.Dataset):
    """In-memory augmented dataset that generates augmentations on-the-fly"""
    
    def __init__(self, data_root, file_list, augment_factor=5, n_bands=4, is_training=True):
        self.data_root = Path(data_root)
        self.original_files = file_list
        self.augment_factor = augment_factor
        self.n_bands = n_bands
        self.is_training = is_training
        
        # Initialize augmentation pipeline
        if is_training and augment_factor > 1:
            self.aug_pipeline = MultispectralHeavyAugmentationPipeline(
                augment_factor=augment_factor, 
                n_bands=n_bands
            )
            self.aug_types = self.aug_pipeline.get_augmentation_types()
        else:
            self.aug_pipeline = None
            self.aug_types = []
        
        # Calculate total dataset size including augmentations
        if is_training and augment_factor > 1:
            self.total_size = len(self.original_files) * augment_factor
        else:
            self.total_size = len(self.original_files)
        
        print(f"InMemoryAugmentedDataset created:")
        print(f"  Original files: {len(self.original_files)}")
        print(f"  Total size with augmentation: {self.total_size}")
        print(f"  Augmentation factor: {augment_factor}")
        print(f"  Training mode: {is_training}")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Determine if this is an original or augmented sample
        original_idx = idx % len(self.original_files)
        is_augmented = idx >= len(self.original_files)
        
        filename = self.original_files[original_idx]
        base_name = filename.replace('.tif', '').replace('.png', '')
        
        try:
            # Load original images
            img_a_path = self.data_root / 'A' / filename
            img_b_path = self.data_root / 'B' / filename
            
            # Load images with proper multi-spectral handling
            with rasterio.open(img_a_path) as src:
                img_a = src.read().transpose(1, 2, 0).astype(np.float32)
                # Select only RGB bands [2,1,0] from 4-band data for model compatibility
                if img_a.shape[2] >= 3:
                    img_a = img_a[:, :, [2, 1, 0]]  # Use bands 2,1,0 as RGB
                if img_a.max() > 1.0:
                    img_a = (img_a / img_a.max()).astype(np.float32)
            
            with rasterio.open(img_b_path) as src:
                img_b = src.read().transpose(1, 2, 0).astype(np.float32)
                # Select only RGB bands [2,1,0] from 4-band data for model compatibility
                if img_b.shape[2] >= 3:
                    img_b = img_b[:, :, [2, 1, 0]]  # Use bands 2,1,0 as RGB
                if img_b.max() > 1.0:
                    img_b = (img_b / img_b.max()).astype(np.float32)
            
            # Load mask
            mask_paths = [self.data_root / 'label' / f"{base_name}.tif", 
                         self.data_root / 'label' / f"{base_name}.png"]
            mask_path = next((p for p in mask_paths if p.exists()), None)
            
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {filename}")
            
            if mask_path.suffix == '.tif':
                with rasterio.open(mask_path) as src:
                    mask = src.read(1).astype(np.float32)
                    mask = (mask > 0).astype(np.uint8)
            else:
                mask = np.array(Image.open(mask_path), dtype=np.uint8)
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
                mask = (mask > 0).astype(np.uint8)
            
            # Apply augmentation if needed
            if is_augmented and self.aug_pipeline and self.is_training:
                # Determine augmentation type based on index
                aug_idx = (idx - len(self.original_files)) % len(self.aug_types)
                aug_type = self.aug_types[aug_idx]
                
                # Apply augmentation
                img_a, img_b, mask = self.aug_pipeline.apply_heavy_augmentation(
                    img_a, img_b, mask, aug_type
                )
            
            # Convert to tensors
            img_a = torch.from_numpy(img_a.transpose(2, 0, 1)).float()  # HWC to CHW
            img_b = torch.from_numpy(img_b.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask).long()
            
            return {
                'A': img_a,
                'B': img_b,
                'L': mask,
                'filename': f"{base_name}_{'aug_' + str(idx) if is_augmented else 'orig'}"
            }
            
        except Exception as e:
            print(f"Error loading sample {idx} (file: {filename}): {e}")
            # Return a dummy sample to avoid training interruption
            dummy_img = torch.zeros(3, 256, 256).float()
            dummy_mask = torch.zeros(256, 256).long()
            return {
                'A': dummy_img,
                'B': dummy_img,
                'L': dummy_mask,
                'filename': f"dummy_{idx}"
            }

class ImprovedPostProcessor:
    """Enhanced postprocessing with confidence-aware filtering"""
    
    def __init__(self, 
                 min_component_size=20,
                 morphology_kernel_size=5,
                 confidence_threshold=0.7,
                 adaptive_threshold=True,
                 gaussian_sigma=1.0,
                 edge_preservation=True):
        self.min_component_size = min_component_size
        self.morphology_kernel_size = morphology_kernel_size
        self.confidence_threshold = confidence_threshold
        self.adaptive_threshold = adaptive_threshold
        self.gaussian_sigma = gaussian_sigma
        self.edge_preservation = edge_preservation
        
    def postprocess_prediction(self, prediction, confidence=None, original_size=None):
        """Apply enhanced postprocessing to prediction"""
        if prediction.max() <= 1.0:
            binary_pred = (prediction > 0.5).astype(np.uint8)
        else:
            binary_pred = (prediction > 0).astype(np.uint8)
            
        if confidence is not None:
            high_conf_mask = confidence > self.confidence_threshold
            binary_pred = binary_pred & high_conf_mask
            
        processed = self._apply_enhanced_morphological_ops(binary_pred)
        processed = self._multi_scale_component_filtering(processed)
        if self.edge_preservation:
            processed = self._edge_preserving_smooth(processed)
        processed = self._final_size_filtering(processed)
        
        return processed
    
    def _apply_enhanced_morphological_ops(self, binary_mask):
        """Apply enhanced morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morphology_kernel_size, self.morphology_kernel_size))
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed
    
    def _multi_scale_component_filtering(self, binary_mask):
        """Multi-scale connected component filtering"""
        labeled_mask, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return binary_mask
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        img_area = binary_mask.shape[0] * binary_mask.shape[1]
        adaptive_min_size = max(self.min_component_size, int(img_area * 0.0001))
        valid_components = np.where(component_sizes >= adaptive_min_size)[0] + 1
        filtered_mask = np.isin(labeled_mask, valid_components).astype(np.uint8)
        return filtered_mask
    
    def _edge_preserving_smooth(self, binary_mask):
        """Edge-preserving smoothing using bilateral filter approach"""
        if self.gaussian_sigma > 0:
            float_mask = binary_mask.astype(np.float32)
            smoothed = ndimage.gaussian_filter(float_mask, sigma=self.gaussian_sigma)
            return (smoothed > 0.6).astype(np.uint8)
        return binary_mask
    
    def _final_size_filtering(self, binary_mask):
        """Final aggressive size filtering for small noise"""
        labeled_mask, num_features = ndimage.label(binary_mask)
        if num_features == 0:
            return binary_mask
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
        final_min_size = max(15, self.min_component_size // 2)
        valid_components = np.where(component_sizes >= final_min_size)[0] + 1
        filtered_mask = np.isin(labeled_mask, valid_components).astype(np.uint8)
        return filtered_mask

class FocalTverskyLoss(torch.nn.Module):
    """Focal Tversky Loss for extreme class imbalance"""
    
    def __init__(self, alpha=0.2, beta=0.8, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[-1]
        
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Expected inputs to be a tensor, got {type(inputs)}")
        
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        if inputs.shape[1] == 1:
            inputs = torch.sigmoid(inputs.squeeze(1))
        else:
            inputs = torch.softmax(inputs, dim=1)[:, 1]
        targets = targets.float()
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        TP = (inputs * targets).sum()
        FP = (inputs * (1 - targets)).sum()
        FN = ((1 - inputs) * targets).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky) ** self.gamma

class EnhancedCombinedLoss(torch.nn.Module):
    """Enhanced combined loss for extreme class imbalance"""
    
    def __init__(self, 
                 focal_tversky_weight=0.7,
                 weighted_bce_weight=0.3,
                 class_weights=None,
                 alpha=0.2,
                 beta=0.8,
                 gamma=2.0):
        super(EnhancedCombinedLoss, self).__init__()
        self.focal_tversky_weight = focal_tversky_weight
        self.weighted_bce_weight = weighted_bce_weight
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.bce_weights = class_weights[1] / class_weights[0] if class_weights is not None else 20.0
        print(f"Using BCE weight for change class: {self.bce_weights}")
        
    def forward(self, predictions, targets):
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[-1]
        
        if not isinstance(predictions, torch.Tensor):
            raise ValueError(f"Expected predictions to be a tensor, got {type(predictions)}")
        
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        
        ft_loss = self.focal_tversky(predictions, targets)
        if predictions.shape[1] == 1:
            pred_prob = torch.sigmoid(predictions.squeeze(1))
        else:
            pred_prob = torch.softmax(predictions, dim=1)[:, 1]
        weight_tensor = torch.ones_like(targets.float())
        weight_tensor[targets == 1] = self.bce_weights
        bce_loss = torch.nn.functional.binary_cross_entropy(pred_prob, targets.float(), weight=weight_tensor)
        total_loss = (self.focal_tversky_weight * ft_loss + 
                    self.weighted_bce_weight * bce_loss)
        return total_loss, ft_loss, bce_loss

class ImprovedTrainingVisualizer:
    """Enhanced visualizer with better metrics tracking"""
    
    def __init__(self, vis_dir, exp_name):
        self.vis_dir = Path(vis_dir) / exp_name
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.vis_dir / "plots"
        self.predictions_dir = self.vis_dir / "predictions"
        self.postprocessed_dir = self.vis_dir / "postprocessed"
        self.plots_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        self.postprocessed_dir.mkdir(exist_ok=True)
        self.postprocessor = ImprovedPostProcessor()
        
        self.train_losses = []
        self.val_losses = []
        self.train_mf1_scores = []
        self.val_mf1_scores = []
        self.train_miou_scores = []
        self.val_miou_scores = []
        self.train_acc_scores = []
        self.val_acc_scores = []
        self.learning_rates = []
        self.epochs = []
        
        self.train_f1_0 = []
        self.train_f1_1 = []
        self.val_f1_0 = []
        self.val_f1_1 = []
        self.train_iou_0 = []
        self.train_iou_1 = []
        self.val_iou_0 = []
        self.val_iou_1 = []
        
        plt.style.use('default')
        print(f"Enhanced Visualizer initialized. Saving to: {self.vis_dir}")
    
    def update_metrics(self, epoch, train_metrics=None, val_metrics=None, lr=None):
        """Update enhanced metrics"""
        self.epochs.append(epoch)
        
        if train_metrics:
            self.train_losses.append(train_metrics.get('loss', 0.0))
            self.train_mf1_scores.append(train_metrics.get('mf1', 0.0))
            self.train_miou_scores.append(train_metrics.get('miou', 0.0))
            self.train_acc_scores.append(train_metrics.get('acc', 0.0))
            self.train_f1_0.append(train_metrics.get('F1_0', 0.0))
            self.train_f1_1.append(train_metrics.get('F1_1', 0.0))
            self.train_iou_0.append(train_metrics.get('iou_0', 0.0))
            self.train_iou_1.append(train_metrics.get('iou_1', 0.0))
        
        if val_metrics:
            self.val_losses.append(val_metrics.get('loss', 0.0))
            self.val_mf1_scores.append(val_metrics.get('mf1', 0.0))
            self.val_miou_scores.append(val_metrics.get('miou', 0.0))
            self.val_acc_scores.append(val_metrics.get('acc', 0.0))
            self.val_f1_0.append(val_metrics.get('F1_0', 0.0))
            self.val_f1_1.append(val_metrics.get('F1_1', 0.0))
            self.val_iou_0.append(val_metrics.get('iou_0', 0.0))
            self.val_iou_1.append(val_metrics.get('iou_1', 0.0))
        
        if lr is not None:
            self.learning_rates.append(float(lr))
    
    def plot_enhanced_training_curves(self, save_plots=True):
        """Create comprehensive training curves"""
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        train_color = '#2E86AB'
        val_color = '#A23B72'
        
        # Loss curves
        ax1 = fig.add_subplot(gs[0, 0])
        if self.train_losses:
            ax1.plot(self.epochs[:len(self.train_losses)], self.train_losses, 
                    label='Train Loss', color=train_color, linewidth=2)
        if self.val_losses:
            ax1.plot(self.epochs[:len(self.val_losses)], self.val_losses, 
                    label='Val Loss', color=val_color, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # mF1 curves
        ax2 = fig.add_subplot(gs[0, 1])
        if self.train_mf1_scores:
            ax2.plot(self.epochs[:len(self.train_mf1_scores)], self.train_mf1_scores, 
                    label='Train mF1', color=train_color, linewidth=2)
        if self.val_mf1_scores:
            ax2.plot(self.epochs[:len(self.val_mf1_scores)], self.val_mf1_scores, 
                    label='Val mF1', color=val_color, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean F1 Score')
        ax2.set_title('Mean F1 Score Progress')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Change class F1
        ax3 = fig.add_subplot(gs[0, 2])
        if self.train_f1_1:
            ax3.plot(self.epochs[:len(self.train_f1_1)], self.train_f1_1, 
                    label='Train F1_1 (Change)', color=train_color, linewidth=2)
        if self.val_f1_1:
            ax3.plot(self.epochs[:len(self.val_f1_1)], self.val_f1_1, 
                    label='Val F1_1 (Change)', color=val_color, linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Change Class F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        plt.suptitle(f'Training Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                     fontsize=16, fontweight='bold')
        
        if save_plots:
            plt.savefig(self.plots_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def save_metrics_json(self):
        """Save metrics to JSON file"""
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mf1_scores': self.train_mf1_scores,
            'val_mf1_scores': self.val_mf1_scores,
            'val_f1_1': self.val_f1_1,
            'timestamp': datetime.now().isoformat(),
            'best_val_mf1': max(self.val_mf1_scores) if self.val_mf1_scores else 0.0
        }
        with open(self.vis_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

class EnhancedCDTrainer(CDTrainer):
    """Enhanced CD Trainer with detailed per-class metrics and in-memory augmentation support"""
    
    def __init__(self, args, dataloaders):
        print("Initializing Enhanced CD Trainer with in-memory augmentation...")
        
        self.enhanced_loss_fn = EnhancedCombinedLoss(
            focal_tversky_weight=args.focal_tversky_weight,
            weighted_bce_weight=args.weighted_bce_weight,
            class_weights=args.class_weights,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.focal_gamma
        )
        
        super().__init__(args, dataloaders)
        self.loss_fn = self.enhanced_loss_fn
        
        self.running_mf1 = 0.0
        self.batch_count = 0
        self.epoch_start_time = time.time()
        
        print("Enhanced CD Trainer initialized with in-memory augmentation support")
    
    def _extract_tensors_from_batch(self, batch):
        """Extract tensors from batch"""
        try:
            if isinstance(batch, dict):
                if 'A' not in batch or 'B' not in batch or 'L' not in batch:
                    raise ValueError(f"Batch dictionary missing required keys. Found: {list(batch.keys())}")
                img_A = batch['A'].to(self.device)
                img_B = batch['B'].to(self.device)
                mask = batch['L'].to(self.device)
                filenames = batch.get('filename', [])
                return img_A, img_B, mask, filenames
            else:
                raise ValueError(f"Unsupported batch type: {type(batch)}")
        except Exception as e:
            print(f"Error extracting tensors from batch: {e}")
            raise e
    
    def set_input(self, batch):
        """Set input tensors for the model"""
        try:
            img_A, img_B, mask, filenames = self._extract_tensors_from_batch(batch)
            self.real_A = img_A
            self.real_B = img_B
            self.L = mask
            self.filenames = filenames
            self.batch = batch
        except Exception as e:
            print(f"Error in set_input: {e}")
            raise e
    
    def _forward_G(self, batch):
        """Forward pass through the generator"""
        try:
            if batch is None:
                raise ValueError("Batch must be provided to _forward_G")
            
            self.set_input(batch)
            self.G_pred = self.net_G(self.real_A, self.real_B)
            
            if isinstance(self.G_pred, (list, tuple)):
                self.G_final_pred = self.G_pred[-1]
            else:
                self.G_final_pred = self.G_pred
            
            if not isinstance(self.G_final_pred, torch.Tensor):
                raise ValueError(f"Expected G_final_pred to be a tensor, got {type(self.G_final_pred)}")
            
            return self.G_final_pred
        except Exception as e:
            print(f"Forward pass failed: {e}")
            raise e
    
    def _backward_G(self, training=True):
        """Compute loss and detailed metrics, perform backward pass if training"""
        try:
            if not hasattr(self, 'L') or self.L is None:
                raise ValueError("Target mask (self.L) is not set. Ensure set_input was called successfully.")
            
            loss, ft_loss, bce_loss = self.enhanced_loss_fn(self.G_final_pred, self.L)
            
            if self.G_final_pred.shape[1] == 1:
                pred_prob = torch.sigmoid(self.G_final_pred.squeeze(1))
            else:
                pred_prob = torch.softmax(self.G_final_pred, dim=1)[:, 1]
            
            pred = (pred_prob > 0.5).cpu().numpy()
            target = self.L.squeeze(1).cpu().numpy() if self.L.dim() == 4 else self.L.cpu().numpy()
            
            detailed_metrics = calculate_detailed_metrics(pred, target)
            detailed_metrics['loss'] = loss.item()
            detailed_metrics['ft_loss'] = ft_loss.item()
            detailed_metrics['bce_loss'] = bce_loss.item()
            
            if training:
                self.G_loss = loss
                self.G_loss.backward()
                if hasattr(self.args, 'gradient_clip_val') and self.args.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.net_G.parameters(), self.args.gradient_clip_val)
            
            return detailed_metrics
        except Exception as e:
            print(f"Backward pass failed: {e}")
            raise e
    
    def train_models(self):
        """Enhanced training loop with detailed metrics"""
        print("Starting enhanced training with in-memory augmentation...")
        
        self.current_epoch = 0
        
        for epoch in range(self.args.max_epochs):
            self.current_epoch = epoch
            self.net_G.train()
            self.epoch_start_time = time.time()
            self.batch_count = 0
            self.running_mf1 = 0.0
            
            epoch_metrics = {
                'loss': 0.0, 'acc': 0.0, 'miou': 0.0, 'mf1': 0.0,
                'iou_0': 0.0, 'iou_1': 0.0, 'F1_0': 0.0, 'F1_1': 0.0,
                'precision_0': 0.0, 'precision_1': 0.0, 
                'recall_0': 0.0, 'recall_1': 0.0
            }
            
            for batch_idx, batch in enumerate(self.dataloaders['train']):
                try:
                    self.optimizer_G.zero_grad()
                    self._forward_G(batch)
                    metrics = self._backward_G(training=True)
                    self.optimizer_G.step()
                    
                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key] += metrics[key]
                    
                    self.batch_count += 1
                    self.running_mf1 = epoch_metrics['mf1'] / self.batch_count
                    
                    if batch_idx % 10 == 0:
                        elapsed = time.time() - self.epoch_start_time
                        total_batches = len(self.dataloaders['train'])
                        eta = elapsed * (total_batches - batch_idx) / max(batch_idx + 1, 1)
                        imps = self.batch_count * self.args.batch_size / elapsed if elapsed > 0 else 0
                        
                        print(f"Is_training: True. [{epoch+1},{self.args.max_epochs}][{batch_idx+1},{total_batches}], "
                              f"imps: {imps:.2f}, est: {eta/3600:.2f}h, G_loss: {metrics['loss']:.5f}, "
                              f"running_mf1: {self.running_mf1:.5f}")
                    
                except Exception as e:
                    print(f"Error in training batch: {e}")
                    raise e
            
            num_batches = self.batch_count
            for key in epoch_metrics:
                epoch_metrics[key] = epoch_metrics[key] / num_batches if num_batches > 0 else 0.0
            
            print(f"Is_training: True. Epoch {epoch+1} / {self.args.max_epochs}, epoch_mF1= {epoch_metrics['mf1']:.5f}")
            
            val_metrics = None
            if epoch % self.args.val_freq == 0 and 'val' in self.dataloaders:
                print("\nBegin evaluation...")
                val_metrics = self.validate(epoch)
            
            if hasattr(self.args, 'visualizer'):
                lr = self.optimizer_G.param_groups[0]['lr']
                self.args.visualizer.update_metrics(epoch, epoch_metrics, val_metrics, lr)
                if epoch % self.args.vis_freq == 0:
                    self.args.visualizer.plot_enhanced_training_curves()
                    self.args.visualizer.save_metrics_json()
            
            if epoch % self.args.save_epoch_freq == 0:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch+1}.pt')
                torch.save(self.net_G.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: epoch_{epoch+1}.pt")
            
            if val_metrics and hasattr(self, 'best_val_mf1'):
                if val_metrics['mf1'] > self.best_val_mf1:
                    self.best_val_mf1 = val_metrics['mf1']
                    best_path = os.path.join(self.args.checkpoint_dir, 'best_ckpt.pt')
                    torch.save(self.net_G.state_dict(), best_path)
                    print(f"New best validation mF1: {self.best_val_mf1:.5f} - saved best_ckpt.pt")
            elif not hasattr(self, 'best_val_mf1'):
                self.best_val_mf1 = val_metrics['mf1'] if val_metrics else 0.0
        
        print("Enhanced in-memory training completed!")
    
    def validate(self, epoch):
        """Enhanced validation with detailed metrics"""
        self.net_G.eval()
        val_start_time = time.time()
        batch_count = 0
        running_mf1 = 0.0
        
        val_metrics = {
            'loss': 0.0, 'acc': 0.0, 'miou': 0.0, 'mf1': 0.0,
            'iou_0': 0.0, 'iou_1': 0.0, 'F1_0': 0.0, 'F1_1': 0.0,
            'precision_0': 0.0, 'precision_1': 0.0, 
            'recall_0': 0.0, 'recall_1': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloaders['val']):
                try:
                    self._forward_G(batch)
                    metrics = self._backward_G(training=False)
                    
                    for key in val_metrics:
                        if key in metrics:
                            val_metrics[key] += metrics[key]
                    
                    batch_count += 1
                    running_mf1 = val_metrics['mf1'] / batch_count
                    
                    if batch_idx % 5 == 0:
                        elapsed = time.time() - val_start_time
                        total_batches = len(self.dataloaders['val'])
                        eta = elapsed * (total_batches - batch_idx) / max(batch_idx + 1, 1)
                        imps = batch_count * self.args.batch_size / elapsed if elapsed > 0 else 0
                        
                        print(f"Is_training: False. [{epoch+1},{self.args.max_epochs}][{batch_idx+1},{total_batches}], "
                              f"imps: {imps:.2f}, est: {eta/3600:.2f}h, G_loss: {metrics['loss']:.5f}, "
                              f"running_mf1: {running_mf1:.5f}")
                
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    raise e
        
        for key in val_metrics:
            val_metrics[key] = val_metrics[key] / batch_count if batch_count > 0 else 0.0
        
        print(f"Is_training: False. Epoch {epoch+1} / {self.args.max_epochs}, epoch_mF1= {val_metrics['mf1']:.5f}")
        print(f"acc: {val_metrics['acc']:.5f} miou: {val_metrics['miou']:.5f} mf1: {val_metrics['mf1']:.5f} "
              f"F1_1: {val_metrics['F1_1']:.5f}")
        
        return val_metrics

class InMemoryCrossValidationTrainer:
    """Memory-efficient trainer for cross-validation without saving fold data"""
    
    def __init__(self, base_args, cv_manager: InMemoryCrossValidationManager):
        self.base_args = base_args
        self.cv_manager = cv_manager
        self.cv_results = []
    
    def create_fold_dataloaders(self, fold_data: Dict):
        """Create dataloaders for a specific fold using in-memory datasets"""
        train_files = fold_data['train_files']
        val_files = fold_data['val_files']
        fold_num = fold_data['fold_num']
        
        print(f"Creating in-memory dataloaders for Fold {fold_num}...")
        
        # Create in-memory datasets
        train_dataset = InMemoryAugmentedDataset(
            data_root=self.base_args.data_root,
            file_list=train_files,
            augment_factor=self.cv_manager.augment_factor,
            n_bands=self.cv_manager.n_bands,
            is_training=True
        )
        
        val_dataset = InMemoryAugmentedDataset(
            data_root=self.base_args.data_root,
            file_list=val_files,
            augment_factor=1,  # No augmentation for validation
            n_bands=self.cv_manager.n_bands,
            is_training=False
        )
        
        # Create dataloaders
        try:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.base_args.batch_size,
                shuffle=True,
                num_workers=self.base_args.num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=True,
                persistent_workers=True if self.base_args.num_workers > 0 else False
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.base_args.batch_size,
                shuffle=False,
                num_workers=self.base_args.num_workers,
                pin_memory=True if torch.cuda.is_available() else False,
                drop_last=False,
                persistent_workers=True if self.base_args.num_workers > 0 else False
            )
        except Exception as e:
            print(f"Error creating dataloaders: {e}")
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.base_args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.base_args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False
            )
        
        dataloaders = {
            'train': train_loader,
            'val': val_loader
        }
        
        print(f"Dataloaders created:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")
        
        return dataloaders
    
    def train_fold(self, fold_info: Dict, fold_data: Dict):
        """Train a single fold with in-memory augmentation"""
        fold_num = fold_info['fold']
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold_num}/{self.cv_manager.n_folds} (IN-MEMORY)")
        print(f"{'='*60}")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create fold-specific arguments
            fold_args = self._create_fold_args(fold_info, fold_data)
            fold_args.n_bands = self.cv_manager.n_bands
            
            # Initialize fold-specific visualizer
            fold_visualizer = ImprovedTrainingVisualizer(
                fold_args.vis_dir, f"{fold_args.exp_name}_fold_{fold_num}"
            )
            fold_args.visualizer = fold_visualizer
            
            # Create in-memory dataloaders for this fold
            print(f"Creating in-memory dataloaders for fold {fold_num}...")
            dataloaders = self.create_fold_dataloaders(fold_data)
            
            # Initialize enhanced trainer for this fold
            print(f"Initializing enhanced trainer for fold {fold_num}...")
            model = EnhancedCDTrainer(args=fold_args, dataloaders=dataloaders)
            
            # Train the model
            print(f"Starting training for fold {fold_num}...")
            model.train_models()
            
            # Get final validation metrics
            final_val_metrics = model.validate(fold_args.max_epochs - 1)
            
            # Store fold results
            fold_result = {
                'fold': fold_num,
                'train_samples': fold_data['target_augmented_count'],
                'original_train_samples': fold_data['original_train_count'],
                'val_samples': fold_data['val_count'],
                'augmentation_factor': self.cv_manager.augment_factor,
                'final_metrics': final_val_metrics,
                'best_val_mf1': getattr(model, 'best_val_mf1', 0.0),
                'memory_efficient': True
            }
            
            self.cv_results.append(fold_result)
            
            print(f"Fold {fold_num} completed successfully!")
            print(f"Final validation mF1: {final_val_metrics['mf1']:.5f}")
            print(f"Best validation mF1: {fold_result['best_val_mf1']:.5f}")
            print(f"Memory usage: IN-MEMORY (no disk storage)")
            
            return fold_result
            
        except Exception as e:
            print(f"Training failed for fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            raise e
    
    def _create_fold_args(self, fold_info: Dict, fold_data: Dict):
        """Create fold-specific arguments"""
        import copy
        fold_args = copy.deepcopy(self.base_args)
        
        # Update paths for this fold (only for checkpoints and visualization)
        fold_num = fold_info['fold']
        fold_args.exp_name = f"{self.base_args.exp_name}_Fold{fold_num}_InMemory_Heavy{self.cv_manager.augment_factor}x"
        fold_args.checkpoint_dir = os.path.join(
            self.base_args.checkpoint_root,
            f"ChangeFormer_CV_InMemory_MultiSpectral",
            f"Fold_{fold_num}_{fold_args.exp_name}"
        )
        fold_args.vis_dir = os.path.join(self.base_args.vis_dir, "cv_inmemory_augmentation_multispectral")
        
        # Create directories
        os.makedirs(fold_args.checkpoint_dir, exist_ok=True)
        os.makedirs(fold_args.vis_dir, exist_ok=True)
        
        # Keep original data root (no fold-specific data directories)
        fold_args.data_root = self.base_args.data_root
        
        # Add missing attributes that the trainer expects
        if not hasattr(fold_args, 'shuffle_AB'):
            fold_args.shuffle_AB = False
        if not hasattr(fold_args, 'multi_scale_train'):
            fold_args.multi_scale_train = False
        if not hasattr(fold_args, 'multi_scale_infer'):
            fold_args.multi_scale_infer = False
        if not hasattr(fold_args, 'multi_pred_weights'):
            fold_args.multi_pred_weights = [1.0]
        
        return fold_args
    
    def run_cross_validation(self):
        """Run complete 3-fold cross-validation with in-memory heavy augmentation"""
        print("STARTING 3-FOLD CROSS-VALIDATION WITH IN-MEMORY MULTISPECTRAL AUGMENTATION")
        print("=" * 80)
        print("MEMORY EFFICIENT MODE: No fold data saved to disk")
        print("=" * 80)
        
        # Create CV folds
        cv_folds = self.cv_manager.create_cv_folds()
        
        # Train each fold
        for fold_info in cv_folds:
            fold_num = fold_info['fold']
            
            # Get fold data information (no physical files created)
            fold_data = self.cv_manager.get_fold_data_info(fold_info)
            
            # Train this fold
            try:
                fold_result = self.train_fold(fold_info, fold_data)
                
                # Save fold results
                self._save_fold_results(fold_result)
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Fold {fold_num} failed: {e}")
                continue
        
        # Compute and display final CV results
        self._compute_cv_summary()
        
        return self.cv_results
    
    def _save_fold_results(self, fold_result: Dict):
        """Save individual fold results"""
        fold_num = fold_result['fold']
        results_dir = Path(self.base_args.vis_dir) / "cv_inmemory_augmentation_multispectral" / "fold_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"fold_{fold_num}_results.json", 'w') as f:
            json.dump(fold_result, f, indent=2, default=str)
    
    def _compute_cv_summary(self):
        """Compute and display cross-validation summary"""
        if not self.cv_results:
            print("No fold results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY (IN-MEMORY MULTISPECTRAL AUGMENTATION)")
        print("=" * 80)
        
        # Collect metrics
        metrics = ['acc', 'miou', 'mf1', 'F1_0', 'F1_1', 'iou_0', 'iou_1']
        cv_summary = defaultdict(list)
        
        for fold_result in self.cv_results:
            fold_metrics = fold_result['final_metrics']
            for metric in metrics:
                if metric in fold_metrics:
                    cv_summary[metric].append(fold_metrics[metric])
        
        # Compute statistics
        summary_stats = {}
        for metric in metrics:
            if metric in cv_summary and cv_summary[metric]:
                values = cv_summary[metric]
                summary_stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Display results
        print(f"Number of folds completed: {len(self.cv_results)}")
        print(f"Heavy augmentation factor: {self.cv_manager.augment_factor}x (IN-MEMORY)")
        print(f"Spectral bands: {self.cv_manager.n_bands}")
        print(f"Memory efficient: No fold data saved to disk")
        print("\nPER-FOLD RESULTS:")
        
        for i, fold_result in enumerate(self.cv_results):
            fold_num = fold_result['fold']
            metrics = fold_result['final_metrics']
            print(f"  Fold {fold_num}: mF1={metrics['mf1']:.5f}, mIoU={metrics['miou']:.5f}, "
                  f"Change_F1={metrics['F1_1']:.5f}, Change_IoU={metrics['iou_1']:.5f}, "
                  f"Train samples={fold_result['train_samples']} (from {fold_result['original_train_samples']} originals)")
        
        print("\nCROSS-VALIDATION STATISTICS:")
        for metric in ['mf1', 'miou', 'F1_1', 'iou_1']:
            if metric in summary_stats:
                stats = summary_stats[metric]
                print(f"  {metric:>12}: {stats['mean']:.5f} ± {stats['std']:.5f} "
                      f"(min: {stats['min']:.5f}, max: {stats['max']:.5f})")
        
        # Save CV summary
        cv_summary_data = {
            'cv_results': self.cv_results,
            'summary_statistics': summary_stats,
            'augmentation_factor': self.cv_manager.augment_factor,
            'n_folds': self.cv_manager.n_folds,
            'n_bands': self.cv_manager.n_bands,
            'memory_efficient': True,
            'disk_usage': 'No fold data saved to disk',
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = Path(self.base_args.vis_dir) / "cv_inmemory_augmentation_multispectral" / "cv_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(cv_summary_data, f, indent=2, default=str)
        
        print(f"\nFull CV summary saved to: {summary_path}")
        
        # Highlight best results
        if 'mf1' in summary_stats:
            best_mf1 = summary_stats['mf1']['max']
            best_fold = None
            for fold_result in self.cv_results:
                if fold_result['final_metrics']['mf1'] == best_mf1:
                    best_fold = fold_result['fold']
                    break
            
            print(f"\nBest Fold: Fold {best_fold} with mF1: {best_mf1:.5f}")
            print(f"MEMORY EFFICIENCY: No disk space used for fold data storage!")
        
        # Plot metrics across folds
        try:
            plt.figure(figsize=(12, 6))
            for metric in ['mf1', 'miou', 'F1_1', 'iou_1']:
                if metric in summary_stats:
                    values = summary_stats[metric]['values']
                    plt.plot(range(1, len(values) + 1), values, marker='o', label=metric)
            
            plt.xlabel('Fold')
            plt.ylabel('Metric Value')
            plt.title(f"Cross-Validation Metrics - {self.base_args.exp_name}")
            plt.legend()
            plt.grid(True)
            plt.savefig(Path(self.base_args.vis_dir) / "cv_inmemory_augmentation_multispectral" / "cv_metrics_plot.png")
            plt.close()
        except Exception as e:
            print(f"Failed to save CV metrics plot: {e}")

def calculate_detailed_metrics(pred, target):
    """Calculate detailed per-class metrics like in reference"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    iou_0 = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0
    iou_1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    miou = (iou_0 + iou_1) / 2
    
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0
    f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    mf1 = (f1_0 + f1_1) / 2
    
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'acc': acc, 'miou': miou, 'mf1': mf1,
        'iou_0': iou_0, 'iou_1': iou_1,
        'F1_0': f1_0, 'F1_1': f1_1,
        'precision_0': precision_0, 'precision_1': precision_1,
        'recall_0': recall_0, 'recall_1': recall_1
    }

def fix_dataset_nan_values(data_root):
    """Fix NaN values in the dataset"""
    print("Fixing NaN values in dataset...")
    data_path = Path(data_root)
    nan_files_fixed = 0
    
    for split in ['A', 'B']:
        img_dir = data_path / split
        if not img_dir.exists():
            print(f"Directory {img_dir} not found!")
            continue
            
        for img_path in img_dir.glob('*.tif'):
            try:
                with rasterio.open(img_path, 'r+') as src:
                    data = src.read()
                    if np.isnan(data).any():
                        print(f"   Fixing NaN in {img_path.name}")
                        nan_files_fixed += 1
                        for band_idx in range(data.shape[0]):
                            band_data = data[band_idx]
                            if np.isnan(band_data).any():
                                valid_pixels = band_data[~np.isnan(band_data)]
                                replacement_value = np.median(valid_pixels) if len(valid_pixels) > 0 else 0.0
                                band_data[np.isnan(band_data)] = replacement_value
                        src.write(data)
            except Exception as e:
                print(f"   Error fixing {img_path.name}: {e}")
    
    print(f"Fixed NaN values in {nan_files_fixed} files")
    return nan_files_fixed > 0

def calculate_enhanced_class_weights(data_root, split_file='train.txt'):
    """Calculate enhanced class weights for extreme class imbalance"""
    print("Calculating enhanced class weights...")
    list_path = Path(data_root) / 'list' / split_file
    if not list_path.exists():
        print(f"Training list not found: {list_path}")
        return [1.0, 30.0]
    
    with open(list_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    
    total_pixels = 0
    change_pixels = 0
    label_dir = Path(data_root) / 'label'
    
    print(f"Base training samples: {len(file_names)}")
    
    for file_name in file_names:
        base_name = file_name.replace('.tif', '')
        label_paths = [label_dir / f"{base_name}.tif", label_dir / f"{base_name}.png"]
        label_path = next((path for path in label_paths if path.exists()), None)
        
        if label_path is None:
            print(f"   Warning: Label not found for {file_name}")
            continue
            
        try:
            if label_path.suffix == '.tif':
                with rasterio.open(label_path) as src:
                    label_data = src.read(1)
            else:
                label_data = np.array(Image.open(label_path))
                if len(label_data.shape) > 2:
                    label_data = label_data[:, :, 0]
            
            label_data = (label_data > 0).astype(np.uint8)
            total_pixels += label_data.size
            change_pixels += np.sum(label_data > 0)
                
        except Exception as e:
            print(f"   Error reading {label_path}: {e}")
    
    if total_pixels == 0:
        print("   Could not calculate class weights: no valid pixels found")
        return [1.0, 30.0]
    
    change_ratio = change_pixels / total_pixels
    no_change_ratio = 1 - change_ratio
    
    # Enhanced class weight calculation for extreme imbalance
    beta = 0.9999
    effective_num_change = (1.0 - beta ** change_pixels) / (1.0 - beta) if change_pixels > 0 else 1.0
    effective_num_no_change = (1.0 - beta ** (total_pixels - change_pixels)) / (1.0 - beta)
    
    change_weight = 1.0 / effective_num_change
    no_change_weight = 1.0 / effective_num_no_change
    
    # Normalize weights
    total = change_weight + no_change_weight
    change_weight = change_weight / total * 2
    no_change_weight = no_change_weight / total * 2
    
    # Ensure minimum weight for change class
    change_weight = max(change_weight, 20.0)
    
    print(f"   Change ratio: {change_ratio:.6f} ({change_ratio*100:.4f}%)")
    print(f"   Computed class weights - No change: {no_change_weight:.2f}, Change: {change_weight:.2f}")
    
    return [float(no_change_weight), float(change_weight)]

def validate_args(args):
    """Validate and fix argument values"""
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"Data root directory does not exist: {args.data_root}")
        return False
    
    required_dirs = ['A', 'B', 'label', 'list']
    missing_dirs = [req_dir for req_dir in required_dirs if not (data_path / req_dir).exists()]
    if missing_dirs:
        print(f"Missing required directories: {missing_dirs}")
        return False
    
    train_list = data_path / 'list' / 'train.txt'
    if not train_list.exists():
        print(f"Training list not found: {train_list}")
        return False
    
    if len(args.rgb_bands) != 3:
        print("RGB bands should be exactly 3 values. Using default for Cartosat-3")
        args.rgb_bands = [2, 1, 0]
    
    if args.focal_tversky_weight + args.weighted_bce_weight != 1.0:
        print("Loss weights should sum to 1. Normalizing...")
        total = args.focal_tversky_weight + args.weighted_bce_weight
        args.focal_tversky_weight = args.focal_tversky_weight / total
        args.weighted_bce_weight = args.weighted_bce_weight / total
    
    if args.tversky_alpha + args.tversky_beta != 1.0:
        print("Tversky alpha and beta should sum to 1. Normalizing...")
        total = args.tversky_alpha + args.tversky_beta
        args.tversky_alpha = args.tversky_alpha / total
        args.tversky_beta = args.tversky_beta / total
    
    print("Arguments validated successfully")
    return True

def enhanced_inmemory_cross_validation_train(args):
    """Enhanced cross-validation training with in-memory heavy multispectral augmentation"""
    print("ENHANCED IN-MEMORY CROSS-VALIDATION CHANGEFORMER TRAINING (MULTISPECTRAL)")
    print("=" * 70)
    print("MEMORY EFFICIENT: No fold data saved to disk!")
    print("=" * 70)
    
    if not validate_args(args):
        print("Argument validation failed. Exiting.")
        exit(1)
    
    if args.fix_nan_values:
        fix_dataset_nan_values(args.data_root)
    
    if args.auto_class_weights:
        # Calculate class weights from original dataset
        class_weights = calculate_enhanced_class_weights(args.data_root)
        args.class_weights = [float(w) for w in class_weights]
        print(f"Using calculated class weights: {args.class_weights}")
    
    # Ensure parameters are correct type
    args.batch_size = int(args.batch_size)
    args.num_workers = int(args.num_workers)
    args.max_epochs = int(args.max_epochs)
    args.img_size = int(args.img_size)
    args.vis_freq = int(args.vis_freq)
    args.save_epoch_freq = int(args.save_epoch_freq)
    
    # Set number of bands for Cartosat-3 (adjust if different)
    args.n_bands = 4  # Pan + RGB
    
    # Initialize in-memory cross-validation manager
    cv_manager = InMemoryCrossValidationManager(
        data_root=args.data_root,
        n_folds=args.n_folds,
        augment_factor=args.cv_augment_factor,
        random_state=args.random_state,
        n_bands=args.n_bands
    )
    
    # Initialize cross-validation trainer
    cv_trainer = InMemoryCrossValidationTrainer(base_args=args, cv_manager=cv_manager)
    
    # Run cross-validation
    try:
        cv_results = cv_trainer.run_cross_validation()
        print(f"\nIn-memory cross-validation completed! Results: {len(cv_results)} successful folds")
        print("MEMORY EFFICIENCY: No disk space consumed by fold data!")
        return cv_results
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    parser = ArgumentParser(description="Memory-Efficient ChangeFormer Training with In-Memory Cross-Validation")

    # Data setting
    parser.add_argument('--data_name', type=str, default='cartoCustom')
    parser.add_argument('--data_root', type=str, default=r'D:/Aman_kr_Singh/NEW_OSCD/ChangeFormer/data/cartoCustom')
    parser.add_argument('--data_format', type=str, default='tif')
    parser.add_argument('--split', type=str, default='list')
    parser.add_argument('--dataset', type=str, default='CDDataset')
    parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_InMemory_CV_Heavy_MultiSpectral')

    # Cartosat-3 specific parameters
    parser.add_argument('--satellite_type', type=str, default='cartosat')
    parser.add_argument('--rgb_bands', type=int, nargs=3, default=[2, 1, 0])
    parser.add_argument('--normalize_method', type=str, default='adaptive_percentile')
    parser.add_argument('--n_bands', type=int, default=4, help='Number of spectral bands')

    # Model config
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--net_G', type=str, default='ChangeFormerV6')
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--n_class', type=int, default=2)

    # Training params
    parser.add_argument('--max_epochs', type=int, default=120)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--img_size', type=int, default=256)    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--lr_policy', type=str, default='cosine_warm_restarts')
    
    # Enhanced loss for extreme imbalance
    parser.add_argument('--loss', type=str, default='combined')
    parser.add_argument('--auto_class_weights', type=bool, default=True)
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1.0, 30.0])
    parser.add_argument('--focal_tversky_weight', type=float, default=0.7)
    parser.add_argument('--weighted_bce_weight', type=float, default=0.3)
    parser.add_argument('--tversky_alpha', type=float, default=0.2)
    parser.add_argument('--tversky_beta', type=float, default=0.8)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Checkpointing and visualization
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/ChangeFormer_InMemory/ChangeFormerV6_InMemory_CV_Heavy_MultiSpectral')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--vis_freq', type=int, default=5)
    parser.add_argument('--save_epoch_freq', type=int, default=10)

    # Pretraining
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', type=str, 
                       default=r'D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_LEVIR\ChangeFormerV6_LEVIR\LEVIR_WEIGHT\best_ckpt.pt')
    
    # Training strategy
    parser.add_argument('--early_stopping_patience', type=int, default=40)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Validation and testing
    parser.add_argument('--val_freq', type=int, default=3)
    
    # Dataset preprocessing
    parser.add_argument('--fix_nan_values', type=bool, default=True)
    parser.add_argument('--label_transform', type=str, default='binary')
    
    # Data augmentation settings
    parser.add_argument('--augment_factor', type=int, default=2)
    parser.add_argument('--multi_scale_train', type=bool, default=False)
    parser.add_argument('--multi_scale_infer', type=bool, default=False)
    parser.add_argument('--multi_pred_weights', type=float, nargs='+', default=[1.0])
    parser.add_argument('--shuffle_AB', type=bool, default=False)

    # Cross-validation specific parameters
    parser.add_argument('--n_folds', type=int, default=3)
    parser.add_argument('--cv_augment_factor', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--mode', type=str, default='cv', choices=['single', 'cv'])
    
    # GPU setup
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()

    # Parse GPU ids
    args.gpu_ids = [int(i) for i in args.gpu.split(',')] if torch.cuda.is_available() else []
    
    # Setup device
    if torch.cuda.is_available() and args.gpu_ids:
        args.device = f'cuda:{args.gpu_ids[0]}'
        print(f"Using GPU: {args.device}")
        gpu_memory = torch.cuda.get_device_properties(args.gpu_ids[0]).total_memory
        print(f"GPU Memory: {gpu_memory / 1024**3:.1f} GB")
    else:
        args.device = 'cpu'
        print("Using CPU")

    print("MEMORY-EFFICIENT CHANGEFORMER WITH IN-MEMORY CROSS-VALIDATION")
    print("=" * 80)
    print("KEY MEMORY OPTIMIZATIONS:")
    print("- NO fold data saved to disk")
    print("- In-memory augmentation on-the-fly")
    print("- Memory cleanup between folds")
    print("- Efficient tensor management")
    print(f"Enhanced Loss: Focal Tversky ({args.focal_tversky_weight}) + Weighted BCE ({args.weighted_bce_weight})")
    print(f"Cross-validation: {args.n_folds}-fold with {args.cv_augment_factor}x in-memory augmentation")
    print("=" * 80)
    print(f"Dataset: {args.data_name} (Cartosat-3)")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    try:
        if args.mode == 'cv':
            enhanced_inmemory_cross_validation_train(args)
            print("\nMEMORY-EFFICIENT IN-MEMORY CROSS-VALIDATION TRAINING COMPLETED!")
            print("NO DISK SPACE CONSUMED BY FOLD DATA!")
        else:
            print("Single training mode not implemented in this memory-efficient version")
            print("Use mode='cv' for in-memory cross-validation")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()