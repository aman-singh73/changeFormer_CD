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

warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

class AdvancedAugmentationPipeline:
    """Advanced augmentation pipeline for satellite imagery change detection"""
    
    def __init__(self, target_multiplier=2.7, preserve_original=True):
        """
        Initialize augmentation pipeline
        
        Args:
            target_multiplier: Factor to multiply original dataset size (2.7 for 194->525)
            preserve_original: Whether to keep original images in augmented dataset
        """
        self.target_multiplier = target_multiplier
        self.preserve_original = preserve_original
        
        # Define augmentation transforms that preserve spatial relationships
        self.geometric_transforms = A.Compose([
            A.OneOf([
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.Transpose(p=1.0),
            ], p=0.8),
        ], additional_targets={'imageB': 'image', 'mask': 'mask'})
        
        self.intensity_transforms = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.7),
        ], additional_targets={'imageB': 'image'})
        
        self.noise_transforms = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 25.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.1, 0.3), p=1.0),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=1.0),
            ], p=0.4),
        ], additional_targets={'imageB': 'image'})
        
        self.weather_transforms = A.Compose([
            A.OneOf([
                A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, alpha_coef=0.1, p=1.0),
                A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 1), angle_lower=0, angle_upper=1, 
                               num_flare_circles_lower=1, num_flare_circles_upper=3, p=1.0),
            ], p=0.2),
        ], additional_targets={'imageB': 'image'})
        
        self.mixed_transforms = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=1.0),
                A.GridDistortion(num_steps=3, distort_limit=0.1, p=1.0),
                A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
            ], p=0.3),
        ], additional_targets={'imageB': 'image', 'mask': 'mask'})
        
        print(f"Augmentation pipeline initialized with target multiplier: {target_multiplier}")
    
    def apply_augmentation_set(self, image_a: np.ndarray, image_b: np.ndarray, 
                              mask: np.ndarray, aug_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply specific augmentation type"""
        
        # Ensure images are in correct format
        if image_a.dtype != np.uint8:
            image_a = (image_a * 255).astype(np.uint8) if image_a.max() <= 1.0 else image_a.astype(np.uint8)
        if image_b.dtype != np.uint8:
            image_b = (image_b * 255).astype(np.uint8) if image_b.max() <= 1.0 else image_b.astype(np.uint8)
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        
        try:
            if aug_type == 'geometric':
                augmented = self.geometric_transforms(image=image_a, imageB=image_b, mask=mask)
                return augmented['image'], augmented['imageB'], augmented['mask']
            
            elif aug_type == 'intensity':
                augmented = self.intensity_transforms(image=image_a, imageB=image_b)
                return augmented['image'], augmented['imageB'], mask
            
            elif aug_type == 'noise':
                augmented = self.noise_transforms(image=image_a, imageB=image_b)
                return augmented['image'], augmented['imageB'], mask
            
            elif aug_type == 'weather':
                augmented = self.weather_transforms(image=image_a, imageB=image_b)
                return augmented['image'], augmented['imageB'], mask
            
            elif aug_type == 'mixed':
                augmented = self.mixed_transforms(image=image_a, imageB=image_b, mask=mask)
                return augmented['image'], augmented['imageB'], augmented['mask']
            
            elif aug_type == 'combined':
                # Apply multiple augmentations
                aug1 = self.geometric_transforms(image=image_a, imageB=image_b, mask=mask)
                aug2 = self.intensity_transforms(image=aug1['image'], imageB=aug1['imageB'])
                if random.random() < 0.3:
                    aug3 = self.noise_transforms(image=aug2['image'], imageB=aug2['imageB'])
                    return aug3['image'], aug3['imageB'], aug1['mask']
                return aug2['image'], aug2['imageB'], aug1['mask']
            
        except Exception as e:
            print(f"Augmentation failed for type {aug_type}: {e}")
            return image_a, image_b, mask
        
        return image_a, image_b, mask

class SingleFoldCrossValidator:
    """Single-fold cross-validation manager for change detection"""
    
    def __init__(self, data_root: str, validation_split: float = 0.2, random_seed: int = 42):
        """
        Initialize single-fold cross-validator
        
        Args:
            data_root: Root directory of the dataset
            validation_split: Fraction of data to use for validation (0.2 = 20%)
            random_seed: Random seed for reproducible splits
        """
        self.data_root = Path(data_root)
        self.validation_split = validation_split
        self.random_seed = random_seed
        self.original_train_list_path = self.data_root / 'list' / 'train.txt'
        self.backup_dir = self.data_root / 'cv_backup'
        
        print(f"Single-fold CV initialized with {validation_split*100:.1f}% validation split")
        
    def create_single_fold_split(self, stratify_by_change_ratio: bool = True):
        """
        Create a single train/validation split with stratification
        
        Args:
            stratify_by_change_ratio: Whether to stratify by change pixel ratio
        """
        print("Creating single-fold train/validation split...")
        
        try:
            if not self.original_train_list_path.exists():
                raise FileNotFoundError(f"Training list not found: {self.original_train_list_path}")
            
            with open(self.original_train_list_path, 'r') as f:
                all_files = [line.strip() for line in f.readlines()]
            
            print(f"Total samples: {len(all_files)}")
            
            if not self.backup_dir.exists():
                self.backup_dir.mkdir(parents=True)
                shutil.copy2(self.original_train_list_path, self.backup_dir / 'original_train.txt')
                print("Created backup of original train list")
            
            if stratify_by_change_ratio:
                change_ratios = self._calculate_change_ratios(all_files)
                stratify_labels = self._create_stratification_bins(change_ratios)
            else:
                stratify_labels = None
            
            np.random.seed(self.random_seed)
            indices = np.arange(len(all_files))
            
            if stratify_labels is not None:
                train_indices, val_indices = self._stratified_split(indices, stratify_labels)
            else:
                np.random.shuffle(indices)
                val_size = int(len(indices) * self.validation_split)
                val_indices = indices[:val_size]
                train_indices = indices[val_size:]
            
            train_files = [all_files[i] for i in train_indices]
            val_files = [all_files[i] for i in val_indices]
            
            train_list_path = self.data_root / 'list' / 'train.txt'
            val_list_path = self.data_root / 'list' / 'val.txt'
            
            with open(train_list_path, 'w') as f:
                for file in train_files:
                    f.write(f"{file}\n")
            
            with open(val_list_path, 'w') as f:
                for file in val_files:
                    f.write(f"{file}\n")
            
            print(f"Single-fold split created:")
            print(f"  Training samples: {len(train_files)}")
            print(f"  Validation samples: {len(val_files)}")
            print(f"  Validation ratio: {len(val_files)/len(all_files):.3f}")
            
            split_info = {
                'total_samples': len(all_files),
                'train_samples': len(train_files),
                'val_samples': len(val_files),
                'validation_split': self.validation_split,
                'random_seed': self.random_seed,
                'stratified': stratify_by_change_ratio,
                'train_files': train_files,
                'val_files': val_files,
                'created_at': datetime.now().isoformat()
            }
            
            with open(self.data_root / 'list' / 'single_fold_split_info.json', 'w') as f:
                json.dump(split_info, f, indent=2)
            
            return train_files, val_files
        
        except Exception as e:
            print(f"Error creating single-fold split: {e}")
            raise e
    
    def _calculate_change_ratios(self, file_list: List[str]) -> List[float]:
        """Calculate change pixel ratios for each file"""
        print("Calculating change ratios for stratification...")
        change_ratios = []
        label_dir = self.data_root / 'label'
        
        for filename in file_list:
            base_name = filename.replace('.tif', '').replace('.png', '')
            label_paths = [label_dir / f"{base_name}.tif", label_dir / f"{base_name}.png"]
            label_path = next((p for p in label_paths if p.exists()), None)
            
            if label_path is None:
                change_ratios.append(0.0)
                continue
            
            try:
                if label_path.suffix == '.tif':
                    with rasterio.open(label_path) as src:
                        label_data = src.read(1)
                else:
                    label_data = np.array(Image.open(label_path))
                    if len(label_data.shape) > 2:
                        label_data = label_data[:, :, 0]
                
                total_pixels = label_data.size
                change_pixels = np.sum(label_data > 0)
                change_ratio = change_pixels / total_pixels if total_pixels > 0 else 0.0
                change_ratios.append(change_ratio)
                
            except Exception as e:
                print(f"Error calculating change ratio for {filename}: {e}")
                change_ratios.append(0.0)
        
        print(f"Change ratio stats: min={min(change_ratios):.6f}, max={max(change_ratios):.6f}, mean={np.mean(change_ratios):.6f}")
        return change_ratios
    
    def _create_stratification_bins(self, change_ratios: List[float]) -> List[int]:
        """Create stratification bins based on change ratios"""
        ratios = np.array(change_ratios)
        percentiles = np.percentile(ratios, [33, 67])
        
        bins = []
        for ratio in ratios:
            if ratio <= percentiles[0]:
                bins.append(0)
            elif ratio <= percentiles[1]:
                bins.append(1)
            else:
                bins.append(2)
        
        print(f"Stratification bins: Low={bins.count(0)}, Medium={bins.count(1)}, High={bins.count(2)}")
        return bins
    
    def _stratified_split(self, indices: np.ndarray, stratify_labels: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stratified split maintaining class distribution"""
        unique_labels = np.unique(stratify_labels)
        train_indices = []
        val_indices = []
        
        for label in unique_labels:
            label_indices = indices[np.array(stratify_labels) == label]
            np.random.shuffle(label_indices)
            
            val_size = int(len(label_indices) * self.validation_split)
            val_indices.extend(label_indices[:val_size])
            train_indices.extend(label_indices[val_size:])
        
        return np.array(train_indices), np.array(val_indices)
    
    def restore_original_split(self):
        """Restore original training split"""
        try:
            if (self.backup_dir / 'original_train.txt').exists():
                shutil.copy2(self.backup_dir / 'original_train.txt', self.original_train_list_path)
                print("Restored original training split")
            else:
                print("No backup found to restore")
        except Exception as e:
            print(f"Error restoring original split: {e}")

def create_augmented_dataset(data_root: str, target_samples: int = 525, 
                           preserve_validation: bool = True):
    """Create augmented dataset to reach target number of samples"""
    
    print(f"Creating augmented dataset with target: {target_samples} samples")
    data_path = Path(data_root)
    
    train_list_path = data_path / 'list' / 'train.txt'
    try:
        if not train_list_path.exists():
            raise FileNotFoundError(f"Training list not found: {train_list_path}")
        
        with open(train_list_path, 'r') as f:
            original_files = [line.strip() for line in f.readlines()]
        
        original_count = len(original_files)
        print(f"Original training samples: {original_count}")
        
        if original_count >= target_samples:
            print("Dataset already has enough samples")
            return original_files
        
        additional_needed = target_samples - original_count
        augmentations_per_image = int(np.ceil(additional_needed / original_count))
        
        print(f"Need {additional_needed} additional samples")
        print(f"Creating {augmentations_per_image} augmentations per original image")
        
        aug_pipeline = AdvancedAugmentationPipeline()
        aug_types = ['geometric', 'intensity', 'combined', 'noise', 'mixed', 'weather']
        
        backup_dir = data_path / 'original_backup'
        if not backup_dir.exists():
            print("Creating backup of original data...")
            backup_dir.mkdir()
            for subdir in ['A', 'B', 'label']:
                shutil.copytree(data_path / subdir, backup_dir / subdir)
            shutil.copy2(train_list_path, backup_dir / 'train.txt')
            print("Backup created successfully")
        
        new_files = []
        augmented_count = 0
        
        for i, filename in enumerate(original_files):
            base_name = filename.replace('.tif', '').replace('.png', '')
            
            try:
                img_a_path = data_path / 'A' / filename
                if img_a_path.suffix == '.tif':
                    with rasterio.open(img_a_path) as src:
                        img_a = src.read().transpose(1, 2, 0)
                else:
                    img_a = np.array(Image.open(img_a_path))
                
                img_b_path = data_path / 'B' / filename
                if img_b_path.suffix == '.tif':
                    with rasterio.open(img_b_path) as src:
                        img_b = src.read().transpose(1, 2, 0)
                else:
                    img_b = np.array(Image.open(img_b_path))
                
                mask_paths = [data_path / 'label' / f"{base_name}.tif", 
                            data_path / 'label' / f"{base_name}.png"]
                mask_path = next((p for p in mask_paths if p.exists()), None)
                
                if mask_path is None:
                    print(f"Warning: Mask not found for {filename}")
                    continue
                
                if mask_path.suffix == '.tif':
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1)
                else:
                    mask = np.array(Image.open(mask_path))
                    if len(mask.shape) > 2:
                        mask = mask[:, :, 0]
                
                for aug_idx in range(augmentations_per_image):
                    if augmented_count >= additional_needed:
                        break
                    
                    aug_type = aug_types[aug_idx % len(aug_types)]
                    
                    aug_img_a, aug_img_b, aug_mask = aug_pipeline.apply_augmentation_set(
                        img_a, img_b, mask, aug_type
                    )
                    
                    new_filename = f"{base_name}_aug_{aug_type}_{aug_idx:02d}.tif"
                    new_files.append(new_filename)
                    
                    aug_a_path = data_path / 'A' / new_filename
                    if len(aug_img_a.shape) == 3:
                        with rasterio.open(
                            aug_a_path, 'w',
                            driver='GTiff',
                            height=aug_img_a.shape[0],
                            width=aug_img_a.shape[1],
                            count=aug_img_a.shape[2],
                            dtype=aug_img_a.dtype
                        ) as dst:
                            for band_idx in range(aug_img_a.shape[2]):
                                dst.write(aug_img_a[:, :, band_idx], band_idx + 1)
                    else:
                        with rasterio.open(
                            aug_a_path, 'w',
                            driver='GTiff',
                            height=aug_img_a.shape[0],
                            width=aug_img_a.shape[1],
                            count=1,
                            dtype=aug_img_a.dtype
                        ) as dst:
                            dst.write(aug_img_a, 1)
                    
                    aug_b_path = data_path / 'B' / new_filename
                    if len(aug_img_b.shape) == 3:
                        with rasterio.open(
                            aug_b_path, 'w',
                            driver='GTiff',
                            height=aug_img_b.shape[0],
                            width=aug_img_b.shape[1],
                            count=aug_img_b.shape[2],
                            dtype=aug_img_b.dtype
                        ) as dst:
                            for band_idx in range(aug_img_b.shape[2]):
                                dst.write(aug_img_b[:, :, band_idx], band_idx + 1)
                    else:
                        with rasterio.open(
                            aug_b_path, 'w',
                            driver='GTiff',
                            height=aug_img_b.shape[0],
                            width=aug_img_b.shape[1],
                            count=1,
                            dtype=aug_img_b.dtype
                        ) as dst:
                            dst.write(aug_img_b, 1)
                    
                    aug_mask_path = data_path / 'label' / new_filename
                    with rasterio.open(
                        aug_mask_path, 'w',
                        driver='GTiff',
                        height=aug_mask.shape[0],
                        width=aug_mask.shape[1],
                        count=1,
                        dtype=aug_mask.dtype
                    ) as dst:
                        dst.write(aug_mask, 1)
                    
                    augmented_count += 1
                    
                    if augmented_count % 50 == 0:
                        print(f"Created {augmented_count}/{additional_needed} augmented samples")
                
                if augmented_count >= additional_needed:
                    break
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        all_files = original_files + new_files
        with open(train_list_path, 'w') as f:
            for filename in all_files:
                f.write(f"{filename}\n")
        
        print(f"Dataset augmentation completed!")
        print(f"Original samples: {original_count}")
        print(f"Augmented samples: {augmented_count}")
        print(f"Total samples: {len(all_files)}")
        print(f"Updated training list: {train_list_path}")
        
        return all_files
    
    except Exception as e:
        print(f"Error in dataset augmentation: {e}")
        raise e

def debug_batch_structure(dataloader, num_batches=2):
    """Debug function to understand batch structure with detailed inspection"""
    print("=== DEBUGGING BATCH STRUCTURE ===")
    try:
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i}:")
            print(f"  Total items in batch: {len(batch)}")
            
            for j, item in enumerate(batch):
                if isinstance(item, torch.Tensor):
                    print(f"  Item {j}: Tensor, shape={item.shape}, dtype={item.dtype}")
                elif isinstance(item, str):
                    print(f"  Item {j}: String = '{item}'")
                elif isinstance(item, (list, tuple)):
                    print(f"  Item {j}: {type(item).__name__} with {len(item)} elements")
                    if len(item) > 0:
                        print(f"    First element type: {type(item[0])}")
                        if isinstance(item[0], torch.Tensor):
                            print(f"    First element shape: {item[0].shape}")
                else:
                    print(f"  Item {j}: {type(item)}")
            
            if i >= num_batches - 1:
                break
        print("=== END DEBUG ===")
    except Exception as e:
        print(f"Error debugging batch structure: {e}")

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
        try:
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
        except Exception as e:
            print(f"Error in postprocessing: {e}")
            return prediction
    
    def _apply_enhanced_morphological_ops(self, binary_mask):
        """Apply enhanced morphological operations"""
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                             (self.morphology_kernel_size, self.morphology_kernel_size))
            opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
            return closed
        except Exception as e:
            print(f"Error in morphological operations: {e}")
            return binary_mask
    
    def _multi_scale_component_filtering(self, binary_mask):
        """Multi-scale connected component filtering"""
        try:
            labeled_mask, num_features = ndimage.label(binary_mask)
            if num_features == 0:
                return binary_mask
            component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
            img_area = binary_mask.shape[0] * binary_mask.shape[1]
            adaptive_min_size = max(self.min_component_size, int(img_area * 0.0001))
            valid_components = np.where(component_sizes >= adaptive_min_size)[0] + 1
            filtered_mask = np.isin(labeled_mask, valid_components).astype(np.uint8)
            return filtered_mask
        except Exception as e:
            print(f"Error in multi-scale component filtering: {e}")
            return binary_mask
    
    def _edge_preserving_smooth(self, binary_mask):
        """Edge-preserving smoothing using bilateral filter approach"""
        try:
            if self.gaussian_sigma > 0:
                float_mask = binary_mask.astype(np.float32)
                smoothed = ndimage.gaussian_filter(float_mask, sigma=self.gaussian_sigma)
                return (smoothed > 0.6).astype(np.uint8)
            return binary_mask
        except Exception as e:
            print(f"Error in edge-preserving smoothing: {e}")
            return binary_mask
    
    def _final_size_filtering(self, binary_mask):
        """Final aggressive size filtering for small noise"""
        try:
            labeled_mask, num_features = ndimage.label(binary_mask)
            if num_features == 0:
                return binary_mask
            component_sizes = ndimage.sum(binary_mask, labeled_mask, range(1, num_features + 1))
            final_min_size = max(15, self.min_component_size // 2)
            valid_components = np.where(component_sizes >= final_min_size)[0] + 1
            filtered_mask = np.isin(labeled_mask, valid_components).astype(np.uint8)
            return filtered_mask
        except Exception as e:
            print(f"Error in final size filtering: {e}")
            return binary_mask

class FocalTverskyLoss(torch.nn.Module):
    """Focal Tversky Loss for extreme class imbalance"""
    
    def __init__(self, alpha=0.2, beta=0.8, gamma=2.0, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        try:
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
        except Exception as e:
            print(f"Error in FocalTverskyLoss: {e}")
            raise e

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
        try:
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
        except Exception as e:
            print(f"Error in EnhancedCombinedLoss: {e}")
            raise e

def calculate_detailed_metrics(pred, target):
    """Calculate detailed per-class metrics like in reference"""
    try:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        
        cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
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
            'acc': acc,
            'miou': miou,
            'mf1': mf1,
            'iou_0': iou_0,
            'iou_1': iou_1,
            'F1_0': f1_0,
            'F1_1': f1_1,
            'precision_0': precision_0,
            'precision_1': precision_1,
            'recall_0': recall_0,
            'recall_1': recall_1
        }
    except Exception as e:
        print(f"Error in calculate_detailed_metrics: {e}")
        return {}

class ImprovedTrainingVisualizer:
    """Enhanced visualizer with cross-validation tracking"""
    
    def __init__(self, vis_dir, exp_name):
        self.vis_dir = Path(vis_dir) / exp_name
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.vis_dir / "plots"
        self.predictions_dir = self.vis_dir / "predictions"
        self.postprocessed_dir = self.vis_dir / "postprocessed"
        self.augmented_samples_dir = self.vis_dir / "augmented_samples"
        self.cv_results_dir = self.vis_dir / "cv_results"
        self.plots_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        self.postprocessed_dir.mkdir(exist_ok=True)
        self.augmented_samples_dir.mkdir(exist_ok=True)
        self.cv_results_dir.mkdir(exist_ok=True)
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
        
        self.cv_fold_results = []
        
        plt.style.use('default')
        print(f"Enhanced Visualizer with CV tracking initialized. Saving to: {self.vis_dir}")
    
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
    
    def save_fold_results(self, fold_results):
        """Save single-fold cross-validation results"""
        try:
            self.cv_fold_results = fold_results
            
            with open(self.cv_results_dir / 'single_fold_results.json', 'w') as f:
                json.dump(fold_results, f, indent=2)
            
            self.plot_fold_summary()
            
            print(f"Single-fold CV results saved to: {self.cv_results_dir}")
        except Exception as e:
            print(f"Error saving fold results: {e}")
    
    def plot_fold_summary(self):
        """Create summary plot for single-fold cross-validation"""
        try:
            if not self.cv_fold_results:
                return
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            metrics = ['best_val_mf1', 'best_val_miou', 'best_change_f1', 'best_change_iou']
            values = [self.cv_fold_results.get(metric, 0.0) for metric in metrics]
            labels = ['mF1', 'mIoU', 'Change F1', 'Change IoU']
            
            ax1.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
            ax1.set_title('Best Validation Metrics')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            for i, v in enumerate(values):
                ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            final_metrics = ['final_val_mf1', 'final_val_miou', 'final_change_f1', 'final_change_iou']
            final_values = [self.cv_fold_results.get(metric, 0.0) for metric in final_metrics]
            
            ax2.bar(labels, final_values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'], alpha=0.7)
            ax2.set_title('Final Validation Metrics')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1)
            for i, v in enumerate(final_values):
                ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            if 'training_history' in self.cv_fold_results:
                history = self.cv_fold_results['training_history']
                if 'val_mf1_history' in history:
                    ax3.plot(history['val_mf1_history'], label='Validation mF1', color='#2E86AB')
                    ax3.plot(history.get('val_change_f1_history', []), label='Change F1', color='#A23B72')
                    ax3.set_title('Validation Metrics Progression')
                    ax3.set_xlabel('Epoch')
                    ax3.set_ylabel('Score')
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)
            
            ax4.axis('off')
            summary_text = []
            summary_text.append("SINGLE-FOLD CROSS-VALIDATION SUMMARY")
            summary_text.append("=" * 40)
            summary_text.append(f"Best Validation mF1: {self.cv_fold_results.get('best_val_mf1', 0.0):.4f}")
            summary_text.append(f"Best Change Detection F1: {self.cv_fold_results.get('best_change_f1', 0.0):.4f}")
            summary_text.append(f"Best Change Detection IoU: {self.cv_fold_results.get('best_change_iou', 0.0):.4f}")
            summary_text.append(f"Final Validation mF1: {self.cv_fold_results.get('final_val_mf1', 0.0):.4f}")
            summary_text.append(f"Training Duration: {self.cv_fold_results.get('training_duration', 'N/A')}")
            summary_text.append(f"Total Epochs: {self.cv_fold_results.get('total_epochs', 'N/A')}")
            
            final_change_f1 = self.cv_fold_results.get('final_change_f1', 0.0)
            if final_change_f1 > 0.4:
                summary_text.append("\nSTATUS: EXCELLENT Change Detection")
            elif final_change_f1 > 0.3:
                summary_text.append("\nSTATUS: GOOD Change Detection")
            elif final_change_f1 > 0.15:
                summary_text.append("\nSTATUS: MODERATE Change Detection")
            else:
                summary_text.append("\nSTATUS: POOR Change Detection")
            
            summary_str = "\n".join(summary_text)
            ax4.text(0.05, 0.95, summary_str, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Single-Fold Cross-Validation Results - {datetime.now().strftime("%Y-%m-%d")}', 
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.cv_results_dir / 'single_fold_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error plotting fold summary: {e}")
    
    def visualize_augmented_samples(self, img_a, img_b, mask, aug_img_a, aug_img_b, aug_mask, filename, aug_type, epoch):
        """Visualize original and augmented image pairs with masks"""
        try:
            fig = plt.figure(figsize=(15, 5))
            gs = GridSpec(2, 3, figure=fig)
            
            if img_a.shape[2] == 3:
                img_a_vis = img_a.astype(np.uint8)
                img_b_vis = img_b.astype(np.uint8)
                aug_img_a_vis = aug_img_a.astype(np.uint8)
                aug_img_b_vis = aug_img_b.astype(np.uint8)
            else:
                img_a_vis = img_a[:, :, [2, 1, 0]].astype(np.uint8)
                img_b_vis = img_b[:, :, [2, 1, 0]].astype(np.uint8)
                aug_img_a_vis = aug_img_a[:, :, [2, 1, 0]].astype(np.uint8)
                aug_img_b_vis = aug_img_b[:, :, [2, 1, 0]].astype(np.uint8)
            
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(img_a_vis)
            ax1.set_title('Original Image A')
            ax1.axis('off')
            
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(img_b_vis)
            ax2.set_title('Original Image B')
            ax2.axis('off')
            
            ax3 = fig.add_subplot(gs[0, 2])
            ax3.imshow(mask, cmap='gray')
            ax3.set_title('Original Mask')
            ax3.axis('off')
            
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(aug_img_a_vis)
            ax4.set_title(f'Augmented Image A ({aug_type})')
            ax4.axis('off')
            
            ax5 = fig.add_subplot(gs[1, 1])
            ax5.imshow(aug_img_b_vis)
            ax5.set_title(f'Augmented Image B ({aug_type})')
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 2])
            ax6.imshow(aug_mask, cmap='gray')
            ax6.set_title(f'Augmented Mask ({aug_type})')
            ax6.axis('off')
            
            plt.suptitle(f'Augmentation Visualization - Epoch {epoch+1} - {filename}', fontsize=12)
            plt.savefig(self.augmented_samples_dir / f'aug_{filename}_epoch_{epoch+1}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error visualizing augmented samples: {e}")
    
    def plot_enhanced_training_curves(self, save_plots=True):
        """Create comprehensive training curves"""
        try:
            fig = plt.figure(figsize=(20, 16))
            gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
            train_color = '#2E86AB'
            val_color = '#A23B72'
            
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
            
            ax3 = fig.add_subplot(gs[0, 2])
            if self.train_miou_scores:
                ax3.plot(self.epochs[:len(self.train_miou_scores)], self.train_miou_scores, 
                        label='Train mIoU', color=train_color, linewidth=2)
            if self.val_miou_scores:
                ax3.plot(self.epochs[:len(self.val_miou_scores)], self.val_miou_scores, 
                        label='Val mIoU', color=val_color, linewidth=2)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Mean IoU')
            ax3.set_title('Mean IoU Progress')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim(0, 1)
            
            ax4 = fig.add_subplot(gs[1, 0])
            if self.train_f1_1:
                ax4.plot(self.epochs[:len(self.train_f1_1)], self.train_f1_1, 
                        label='Train F1_1 (Change)', color=train_color, linewidth=2)
            if self.val_f1_1:
                ax4.plot(self.epochs[:len(self.val_f1_1)], self.val_f1_1, 
                        label='Val F1_1 (Change)', color=val_color, linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('F1 Score')
            ax4.set_title('Change Class F1 Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1)
            
            ax5 = fig.add_subplot(gs[1, 1])
            if self.train_iou_1:
                ax5.plot(self.epochs[:len(self.train_iou_1)], self.train_iou_1, 
                        label='Train IoU_1 (Change)', color=train_color, linewidth=2)
            if self.val_iou_1:
                ax5.plot(self.epochs[:len(self.val_iou_1)], self.val_iou_1, 
                        label='Val IoU_1 (Change)', color=val_color, linewidth=2)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('IoU')
            ax5.set_title('Change Class IoU')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0, 1)
            
            ax6 = fig.add_subplot(gs[1, 2])
            if self.train_acc_scores:
                ax6.plot(self.epochs[:len(self.train_acc_scores)], self.train_acc_scores, 
                        label='Train Acc', color=train_color, linewidth=2)
            if self.val_acc_scores:
                ax6.plot(self.epochs[:len(self.val_acc_scores)], self.val_acc_scores, 
                        label='Val Acc', color=val_color, linewidth=2)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Accuracy')
            ax6.set_title('Overall Accuracy')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(0, 1)
            
            ax7 = fig.add_subplot(gs[2, 0])
            if self.learning_rates:
                ax7.plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 
                        color='orange', linewidth=2)
                ax7.set_xlabel('Epoch')
                ax7.set_ylabel('Learning Rate')
                ax7.set_title('Learning Rate Schedule')
                ax7.set_yscale('log')
                ax7.grid(True, alpha=0.3)
            
            ax8 = fig.add_subplot(gs[2, 1:])
            ax8.axis('off')
            
            if self.val_mf1_scores and self.val_f1_1:
                summary_text = []
                summary_text.append("SINGLE-FOLD CV PERFORMANCE SUMMARY:")
                summary_text.append(f"Best Val mF1: {max(self.val_mf1_scores):.5f}")
                summary_text.append(f"Best Val mIoU: {max(self.val_miou_scores):.5f}")
                summary_text.append(f"Best Change F1: {max(self.val_f1_1):.5f}")
                summary_text.append(f"Best Change IoU: {max(self.val_iou_1):.5f}")
                summary_text.append(f"Final Val mF1: {self.val_mf1_scores[-1]:.5f}")
                summary_text.append(f"Final Change F1: {self.val_f1_1[-1]:.5f}")
                summary_text.append(f"Final Change IoU: {self.val_iou_1[-1]:.5f}")
                
                final_change_f1 = self.val_f1_1[-1]
                if final_change_f1 > 0.4:
                    summary_text.append("Status: EXCELLENT Change Detection")
                elif final_change_f1 > 0.3:
                    summary_text.append("Status: GOOD Change Detection")
                elif final_change_f1 > 0.15:
                    summary_text.append("Status: MODERATE Change Detection")
                else:
                    summary_text.append("Status: POOR Change Detection")
                
                summary_str = "\n".join(summary_text)
                ax8.text(0.1, 0.9, summary_str, transform=ax8.transAxes, fontsize=11,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
            
            plt.suptitle(f'Enhanced Single-Fold CV Training Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                         fontsize=16, fontweight='bold')
            
            if save_plots:
                plt.savefig(self.plots_dir / 'enhanced_training_curves.png', dpi=300, bbox_inches='tight')
                plt.savefig(self.plots_dir / 'enhanced_training_curves.pdf', bbox_inches='tight')
            plt.close()
            return fig
        except Exception as e:
            print(f"Error plotting training curves: {e}")
            return None
    
    def save_metrics_json(self):
        """Save enhanced metrics to JSON file"""
        try:
            metrics = {
                'epochs': self.epochs,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_mf1_scores': self.train_mf1_scores,
                'val_mf1_scores': self.val_mf1_scores,
                'train_miou_scores': self.train_miou_scores,
                'val_miou_scores': self.val_miou_scores,
                'train_f1_0': self.train_f1_0,
                'train_f1_1': self.train_f1_1,
                'val_f1_0': self.val_f1_0,
                'val_f1_1': self.val_f1_1,
                'timestamp': datetime.now().isoformat(),
                'best_val_mf1': max(self.val_mf1_scores) if self.val_mf1_scores else 0.0,
                'best_change_f1': max(self.val_f1_1) if self.val_f1_1 else 0.0,
                'cv_fold_results': self.cv_fold_results
            }
            with open(self.vis_dir / 'enhanced_training_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Enhanced metrics saved to: {self.vis_dir / 'enhanced_training_metrics.json'}")
        except Exception as e:
            print(f"Error saving metrics JSON: {e}")

class EnhancedCDTrainerWithCV(CDTrainer):
    """Enhanced CD Trainer with single-fold cross-validation support"""
    
    def __init__(self, args, dataloaders):
        print("Initializing Enhanced CD Trainer with single-fold cross-validation...")
        
        try:
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
            
            self.augmentation_pipeline = None
            if args.augment_factor > 1:
                self.augmentation_pipeline = AdvancedAugmentationPipeline(target_multiplier=args.augment_factor)
                print(f"Augmentation enabled with target multiplier: {args.augment_factor}")
            
            self.running_mf1 = 0.0
            self.batch_count = 0
            self.epoch_start_time = time.time()
            
            self.fold_metrics = {
                'best_val_mf1': 0.0,
                'best_val_miou': 0.0,
                'best_change_f1': 0.0,
                'best_change_iou': 0.0,
                'final_val_mf1': 0.0,
                'final_val_miou': 0.0,
                'final_change_f1': 0.0,
                'final_change_iou': 0.0,
                'training_history': {
                    'val_mf1_history': [],
                    'val_change_f1_history': [],
                    'val_miou_history': [],
                    'train_loss_history': [],
                    'val_loss_history': []
                }
            }
            
            print("Enhanced CD Trainer with CV initialized with detailed metrics tracking and augmentation")
        except Exception as e:
            print(f"Error initializing EnhancedCDTrainerWithCV: {e}")
            raise e
    
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
        """Set input tensors for the model, including augmentation"""
        try:
            img_A, img_B, mask, filenames = self._extract_tensors_from_batch(batch)
            self.real_A = img_A
            self.real_B = img_B
            self.L = mask
            self.filenames = filenames
            self.batch = batch
            
            if self.augmentation_pipeline and self.training:
                aug_types = ['geometric', 'intensity', 'combined', 'noise', 'mixed', 'weather']
                aug_type = random.choice(aug_types)
                
                img_A_np = img_A.cpu().numpy().transpose(0, 2, 3, 1)
                img_B_np = img_B.cpu().numpy().transpose(0, 2, 3, 1)
                mask_np = mask.cpu().numpy().squeeze(1)
                
                aug_img_A = []
                aug_img_B = []
                aug_mask = []
                
                for i in range(img_A_np.shape[0]):
                    aug_A, aug_B, aug_M = self.augmentation_pipeline.apply_augmentation_set(
                        img_A_np[i], img_B_np[i], mask_np[i], aug_type
                    )
                    aug_img_A.append(aug_A)
                    aug_img_B.append(aug_B)
                    aug_mask.append(aug_M)
                    
                    if self.batch_count % 50 == 0 and hasattr(self.args, 'visualizer') and i < 1:
                        self.args.visualizer.visualize_augmented_samples(
                            img_A_np[i], img_B_np[i], mask_np[i],
                            aug_A, aug_B, aug_M,
                            filenames[i] if i < len(filenames) else f"sample_{i}",
                            aug_type, self.current_epoch
                        )
                
                aug_img_A = torch.from_numpy(np.stack(aug_img_A).transpose(0, 3, 1, 2)).to(self.device)
                aug_img_B = torch.from_numpy(np.stack(aug_img_B).transpose(0, 3, 1, 2)).to(self.device)
                aug_mask = torch.from_numpy(np.stack(aug_mask)[:, np.newaxis, :, :]).to(self.device)
                
                self.real_A = aug_img_A
                self.real_B = aug_img_B
                self.L = aug_mask
            
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
        """Enhanced training loop with detailed metrics and single-fold cross-validation"""
        print("Starting enhanced single-fold cross-validation training...")
        
        try:
            training_start_time = time.time()
            best_val_mf1 = 0.0
            best_val_miou = 0.0
            best_change_f1 = 0.0
            best_change_iou = 0.0
            
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
                            
                            print(f"CV Training: [{epoch+1},{self.args.max_epochs}][{batch_idx+1},{total_batches}], "
                                  f"imps: {imps:.2f}, est: {eta/3600:.2f}h, G_loss: {metrics['loss']:.5f}, "
                                  f"running_mF1: {self.running_mf1:.5f}")
                        
                    except Exception as e:
                        print(f"Error in training batch: {e}")
                        raise e
                
                num_batches = self.batch_count
                for key in epoch_metrics:
                    epoch_metrics[key] = epoch_metrics[key] / num_batches if num_batches > 0 else 0.0
                
                self.fold_metrics['training_history']['train_loss_history'].append(epoch_metrics['loss'])
                
                print(f"CV Training Epoch {epoch+1}/{self.args.max_epochs}, mF1={epoch_metrics['mf1']:.5f}")
                print(f"acc: {epoch_metrics['acc']:.5f} miou: {epoch_metrics['miou']:.5f} mf1: {epoch_metrics['mf1']:.5f} "
                      f"change_F1: {epoch_metrics['F1_1']:.5f} change_IoU: {epoch_metrics['iou_1']:.5f}")
                
                val_metrics = None
                if epoch % self.args.val_freq == 0 and 'val' in self.dataloaders:
                    print("Begin cross-validation...")
                    val_metrics = self.validate(epoch)
                    
                    if val_metrics:
                        if val_metrics['mf1'] > best_val_mf1:
                            best_val_mf1 = val_metrics['mf1']
                            self.fold_metrics['best_val_mf1'] = best_val_mf1
                        
                        if val_metrics['miou'] > best_val_miou:
                            best_val_miou = val_metrics['miou']
                            self.fold_metrics['best_val_miou'] = best_val_miou
                        
                        if val_metrics['F1_1'] > best_change_f1:
                            best_change_f1 = val_metrics['F1_1']
                            self.fold_metrics['best_change_f1'] = best_change_f1
                        
                        if val_metrics['iou_1'] > best_change_iou:
                            best_change_iou = val_metrics['iou_1']
                            self.fold_metrics['best_change_iou'] = best_change_iou
                        
                        self.fold_metrics['training_history']['val_mf1_history'].append(val_metrics['mf1'])
                        self.fold_metrics['training_history']['val_change_f1_history'].append(val_metrics['F1_1'])
                        self.fold_metrics['training_history']['val_miou_history'].append(val_metrics['miou'])
                        self.fold_metrics['training_history']['val_loss_history'].append(val_metrics['loss'])
                
                if hasattr(self.args, 'visualizer'):
                    lr = self.optimizer_G.param_groups[0]['lr']
                    self.args.visualizer.update_metrics(epoch, epoch_metrics, val_metrics, lr)
                    if epoch % self.args.vis_freq == 0:
                        self.args.visualizer.plot_enhanced_training_curves()
                        self.args.visualizer.save_metrics_json()
                
                if epoch % self.args.save_epoch_freq == 0:
                    checkpoint_path = os.path.join(self.args.checkpoint_dir, f'cv_epoch_{epoch+1}.pt')
                    torch.save(self.net_G.state_dict(), checkpoint_path)
                    print(f"Saved CV checkpoint: cv_epoch_{epoch+1}.pt")
                
                if val_metrics and val_metrics['mf1'] > getattr(self, 'best_val_mf1', 0.0):
                    self.best_val_mf1 = val_metrics['mf1']
                    best_path = os.path.join(self.args.checkpoint_dir, 'best_cv_ckpt.pt')
                    torch.save(self.net_G.state_dict(), best_path)
                    print(f"New best CV validation mF1: {self.best_val_mf1:.5f} - saved best_cv_ckpt.pt")
            
            if 'val' in self.dataloaders:
                print("Final cross-validation evaluation...")
                final_val_metrics = self.validate(self.args.max_epochs - 1, final_eval=True)
                if final_val_metrics:
                    self.fold_metrics['final_val_mf1'] = final_val_metrics['mf1']
                    self.fold_metrics['final_val_miou'] = final_val_metrics['miou']
                    self.fold_metrics['final_change_f1'] = final_val_metrics['F1_1']
                    self.fold_metrics['final_change_iou'] = final_val_metrics['iou_1']
            
            training_duration = time.time() - training_start_time
            self.fold_metrics['training_duration'] = f"{training_duration/3600:.2f}h"
            self.fold_metrics['total_epochs'] = self.args.max_epochs
            
            print("Enhanced single-fold cross-validation training completed!")
            print(f"Best validation mF1: {self.fold_metrics['best_val_mf1']:.5f}")
            print(f"Best change detection F1: {self.fold_metrics['best_change_f1']:.5f}")
            print(f"Training duration: {self.fold_metrics['training_duration']}")
            
            return self.fold_metrics
        
        except Exception as e:
            print(f"Error in training models: {e}")
            raise e
    
    def validate(self, epoch, final_eval=False):
        """Enhanced validation with detailed metrics for cross-validation"""
        try:
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
                        
                        if final_eval or batch_idx % 5 == 0:
                            elapsed = time.time() - val_start_time
                            total_batches = len(self.dataloaders['val'])
                            eta = elapsed * (total_batches - batch_idx) / max(batch_idx + 1, 1)
                            imps = batch_count * self.args.batch_size / elapsed if elapsed > 0 else 0
                            
                            status = "Final CV Eval" if final_eval else "CV Validation"
                            print(f"{status}: [{epoch+1},{self.args.max_epochs}][{batch_idx+1},{total_batches}], "
                                  f"imps: {imps:.2f}, est: {eta/3600:.2f}h, G_loss: {metrics['loss']:.5f}, "
                                  f"running_mF1: {running_mf1:.5f}")
                    
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        raise e
                
                for key in val_metrics:
                    val_metrics[key] = val_metrics[key] / batch_count if batch_count > 0 else 0.0
                
                status = "Final CV Evaluation" if final_eval else f"CV Validation Epoch {epoch+1}/{self.args.max_epochs}"
                print(f"{status}, mF1={val_metrics['mf1']:.5f}")
                print(f"acc: {val_metrics['acc']:.5f} miou: {val_metrics['miou']:.5f} mf1: {val_metrics['mf1']:.5f} "
                      f"change_F1: {val_metrics['F1_1']:.5f} change_IoU: {val_metrics['iou_1']:.5f}")
                
                return val_metrics
        
        except Exception as e:
            print(f"Error in validation: {e}")
            raise e

def fix_dataset_nan_values(data_root):
    """Fix NaN values in the dataset"""
    print("Fixing NaN values in dataset...")
    data_path = Path(data_root)
    nan_files_fixed = 0
    
    try:
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
    
    except Exception as e:
        print(f"Error in fix_dataset_nan_values: {e}")
        return False

def calculate_enhanced_class_weights(data_root, split_file='train.txt'):
    """Calculate enhanced class weights for extreme class imbalance"""
    print("Calculating enhanced class weights...")
    list_path = Path(data_root) / 'list' / split_file
    try:
        if not list_path.exists():
            print(f"Training list not found: {list_path}")
            return [1.0, 30.0]
        
        with open(list_path, 'r') as f:
            file_names = [line.strip() for line in f.readlines()]
        
        print(f"Base training samples: {len(file_names)}")
        
        total_pixels = 0
        change_pixels = 0
        label_dir = Path(data_root) / 'label'
        
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
        
        beta = 0.9999
        effective_num_change = (1.0 - beta ** change_pixels) / (1.0 - beta) if change_pixels > 0 else 1.0
        effective_num_no_change = (1.0 - beta ** (total_pixels - change_pixels)) / (1.0 - beta)
        
        change_weight = 1.0 / effective_num_change
        no_change_weight = 1.0 / effective_num_no_change
        
        total = change_weight + no_change_weight
        change_weight = change_weight / total * 2
        no_change_weight = no_change_weight / total * 2
        
        print(f"Calculated class weights: no_change={no_change_weight:.4f}, change={change_weight:.4f}")
        return [no_change_weight, change_weight]
    
    except Exception as e:
        print(f"Error in calculate_enhanced_class_weights: {e}")
        return [1.0, 30.0]
        
def validate_args(args):
    """Validate and fix argument values"""
    data_path = Path(args.data_root)
    if not data_path.exists():
        return False
    
    required_dirs = ['A', 'B', 'label', 'list']
    missing_dirs = [req_dir for req_dir in required_dirs if not (data_path / req_dir).exists()]
    if missing_dirs:
        return False
    
    train_list = data_path / 'list' / 'train.txt'
    if not train_list.exists():
        return False
    
    if len(args.rgb_bands) != 3:
        args.rgb_bands = [2, 1, 0]
    
    if args.focal_tversky_weight + args.weighted_bce_weight != 1.0:
        total = args.focal_tversky_weight + args.weighted_bce_weight
        args.focal_tversky_weight = args.focal_tversky_weight / total
        args.weighted_bce_weight = args.weighted_bce_weight / total
    
    if args.tversky_alpha + args.tversky_beta != 1.0:
        total = args.tversky_alpha + args.tversky_beta
        args.tversky_alpha = args.tversky_alpha / total
        args.tversky_beta = args.tversky_beta / total
    
    # Validate cross-validation parameters
    if not 0.1 <= args.cv_validation_split <= 0.4:
        args.cv_validation_split = 0.2
    
    return True

def enhanced_train_with_cv(args):
    """Enhanced training function with single-fold cross-validation"""
    
    if not validate_args(args):
        exit(1)
    
    if args.fix_nan_values:
        fix_dataset_nan_values(args.data_root)
    
    # Initialize single-fold cross-validator
    cv_manager = SingleFoldCrossValidator(
        data_root=args.data_root,
        validation_split=args.cv_validation_split,
        random_seed=args.cv_random_seed
    )
    
    try:
        # Check if validation split already exists
        val_list_path = Path(args.data_root) / 'list' / 'val.txt'
        train_list_path = Path(args.data_root) / 'list' / 'train.txt'
        
        if val_list_path.exists() and train_list_path.exists():
            with open(train_list_path, 'r') as f:
                train_files = [line.strip() for line in f.readlines()]
            with open(val_list_path, 'r') as f:
                val_files = [line.strip() for line in f.readlines()]
            
            # Update CV manager with actual split ratio
            actual_split = len(val_files) / (len(train_files) + len(val_files))
            cv_manager.validation_split = actual_split
        else:
            # Create single-fold split
            train_files, val_files = cv_manager.create_single_fold_split(
                stratify_by_change_ratio=args.cv_stratify
            )
        
        # Create augmented dataset if needed
        if args.augment_factor > 1:
            create_augmented_dataset(
                args.data_root, 
                target_samples=int(args.augment_factor * len(train_files)), 
                preserve_validation=True
            )
        
        if args.auto_class_weights:
            class_weights = calculate_enhanced_class_weights(args.data_root)
            args.class_weights = [float(w) for w in class_weights]
        
        # Ensure parameters are correct type
        args.batch_size = int(args.batch_size)
        args.num_workers = int(args.num_workers)
        args.max_epochs = int(args.max_epochs)
        args.img_size = int(args.img_size)
        args.vis_freq = int(args.vis_freq)
        args.save_epoch_freq = int(args.save_epoch_freq)
        
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Initialize enhanced visualizer with CV support
        visualizer = ImprovedTrainingVisualizer(args.vis_dir, args.exp_name)
        args.visualizer = visualizer
        
        try:
            dataloaders = utils.get_loaders(args)
        except Exception as e:
            raise e
        
        # Convert class weights to tensor
        args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
        
        # Initialize enhanced trainer with cross-validation
        model = EnhancedCDTrainerWithCV(args=args, dataloaders=dataloaders)
        
        fold_results = model.train_models()
        
        # Save cross-validation results
        visualizer.save_fold_results(fold_results)
        
        # Create final visualizations
        visualizer.plot_enhanced_training_curves()
        visualizer.save_metrics_json()
        
        # Print final summary
        print("\n" + "=" * 80)
        print("SINGLE-FOLD CROSS-VALIDATION TRAINING COMPLETED!")
        print("=" * 80)
        print(f"Best Validation mF1: {fold_results['best_val_mf1']:.5f}")
        print(f"Best Change Detection F1: {fold_results['best_change_f1']:.5f}")
        print(f"Best Change Detection IoU: {fold_results['best_change_iou']:.5f}")
        print(f"Final Validation mF1: {fold_results['final_val_mf1']:.5f}")
        print(f"Final Change Detection F1: {fold_results['final_change_f1']:.5f}")
        print(f"Training Duration: {fold_results['training_duration']}")
        print(f"Results saved in: {args.vis_dir}/{args.exp_name}")
        print("=" * 80)
        
        return model, fold_results
        
    except Exception as e:
        # Restore original split if training failed
        cv_manager.restore_original_split()
        raise e
    
    finally:
        # Optionally restore original split after training
        if args.restore_original_split:
            cv_manager.restore_original_split()

if __name__ == '__main__':
    parser = ArgumentParser(description="Enhanced ChangeFormer Training with Single-Fold Cross-Validation")

    # Data settings
    parser.add_argument('--data_name', type=str, default='cartoCustom')
    parser.add_argument('--data_root', type=str, default='./data/cartoCustom')
    parser.add_argument('--data_format', type=str, default='tif')
    parser.add_argument('--split', type=str, default='list')
    parser.add_argument('--dataset', type=str, default='CDDataset')
    parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_Enhanced_SingleFold_CV_updatedCode')

    # Cartosat-3 specific parameters
    parser.add_argument('--satellite_type', type=str, default='cartosat')
    parser.add_argument('--rgb_bands', type=int, nargs=3, default=[2, 1, 0])
    parser.add_argument('--normalize_method', type=str, default='adaptive_percentile')

    # Enhanced Model config for class imbalance
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--net_G', type=str, default='ChangeFormerV6')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--n_class', type=int, default=2)

    # Enhanced Training params for class imbalance
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=5e-5)
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
    
    # Single-fold Cross-validation parameters
    parser.add_argument('--cv_validation_split', type=float, default=0.2)
    parser.add_argument('--cv_random_seed', type=int, default=42)
    parser.add_argument('--cv_stratify', type=bool, default=True)
    parser.add_argument('--restore_original_split', type=bool, default=False)
    
    # Enhanced checkpointing and visualization
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/ChangeFormer_myCustom/ChangeFormerV6_Enhanced_SingleFold_CV_updatedCode')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--vis_freq', type=int, default=5)
    parser.add_argument('--save_epoch_freq', type=int, default=10)

    # Enhanced pretraining and transfer learning
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', type=str, 
                       default=r'D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_LEVIR\ChangeFormerV6_LEVIR\LEVIR_WEIGHT\best_ckpt.pt')
    
    # Enhanced training strategy
    parser.add_argument('--early_stopping_patience', type=int, default=40)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # Validation and testing
    parser.add_argument('--val_freq', type=int, default=3)
    parser.add_argument('--save_best_recall', type=bool, default=True)
    
    # Dataset preprocessing
    parser.add_argument('--fix_nan_values', type=bool, default=True)
    parser.add_argument('--label_transform', type=str, default='binary')
    
    # Data augmentation settings
    parser.add_argument('--augment_factor', type=int, default=1)
    parser.add_argument('--multi_scale_train', type=bool, default=False)
    parser.add_argument('--multi_scale_infer', type=bool, default=False)
    parser.add_argument('--multi_pred_weights', type=float, nargs='+', default=[1.0])
    parser.add_argument('--shuffle_AB', type=bool, default=False)

    # GPU setup
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()

    # Parse GPU ids
    args.gpu_ids = [int(i) for i in args.gpu.split(',')] if torch.cuda.is_available() else []
    
    # Setup device
    if torch.cuda.is_available() and args.gpu_ids:
        args.device = f'cuda:{args.gpu_ids[0]}'
        gpu_memory = torch.cuda.get_device_properties(args.gpu_ids[0]).total_memory
    else:
        args.device = 'cpu'

    print("ENHANCED CHANGEFORMER WITH SINGLE-FOLD CROSS-VALIDATION")
    print("=" * 80)
    print("KEY IMPROVEMENTS FOR STABLE METRICS:")
    print(f"✓ Adaptive threshold calculation (prevents sudden metric drops)")
    print(f"✓ Safe metrics calculation with NaN/Inf handling")
    print(f"✓ Robust loss computation with fallback mechanisms")
    print(f"✓ Enhanced validation with error handling")
    print(f"✓ Gradient clipping and validation")
    print("=" * 80)
    print(f"Dataset: {args.data_name}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    try:
        model, fold_results = enhanced_train_with_cv(args)
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f" Best Validation mF1: {fold_results['best_val_mf1']:.5f}")
        print(f"🎯 Best Change Detection F1: {fold_results['best_change_f1']:.5f}")
        print(f"📈 Best Change Detection IoU: {fold_results['best_change_iou']:.5f}")
        
        # Performance assessment
        if fold_results['best_change_f1'] > 0.4:
            print("🟢 STATUS: EXCELLENT Change Detection Performance")
        elif fold_results['best_change_f1'] > 0.3:
            print("🟡 STATUS: GOOD Change Detection Performance")  
        elif fold_results['best_change_f1'] > 0.15:
            print("🟠 STATUS: MODERATE Change Detection Performance")
        else:
            print("🔴 STATUS: NEEDS IMPROVEMENT")
        
        print(f"\n📁 Results saved in: {args.vis_dir}/{args.exp_name}")
        print(f"🏆 Best model: {args.checkpoint_dir}/best_cv_ckpt.pt")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()