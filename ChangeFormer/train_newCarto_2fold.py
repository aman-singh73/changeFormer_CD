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

class CrossValidationManager:
    """Manage 3-fold cross-validation with heavy augmentation for multi-spectral data"""
    
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
        
        print(f"CrossValidation Manager initialized:")
        print(f"  Original samples: {len(self.original_files)}")
        print(f"  Folds: {n_folds}")
        print(f"  Augmentation factor: {augment_factor}x")
        print(f"  Spectral bands: {n_bands}")
    
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
    
    def create_heavy_augmented_fold(self, fold_info: Dict, augment_factor: int = None):
        """Create heavily augmented dataset for a specific fold"""
        if augment_factor is None:
            augment_factor = self.augment_factor
            
        fold_num = fold_info['fold']
        train_files = fold_info['train_files']
        val_files = fold_info['val_files']
        
        print(f"\nCreating heavy augmentation for Fold {fold_num}...")
        print(f"Original training samples: {len(train_files)}")
        
        # Target number of augmented samples
        target_samples = len(train_files) * augment_factor
        additional_needed = target_samples - len(train_files)
        augmentations_per_image = int(np.ceil(additional_needed / len(train_files)))
        
        print(f"Target samples: {target_samples}")
        print(f"Augmentations per image: {augmentations_per_image}")
        
        # Initialize multispectral augmentation pipeline
        aug_pipeline = MultispectralHeavyAugmentationPipeline(
            augment_factor=augment_factor, 
            n_bands=self.n_bands
        )
        aug_types = aug_pipeline.get_augmentation_types()
        
        # Create fold-specific directories
        fold_dir = self.data_root / f'cv_fold_{fold_num}'
        fold_dir.mkdir(exist_ok=True)
        
        for subdir in ['A', 'B', 'label', 'list']:
            (fold_dir / subdir).mkdir(exist_ok=True)
        
        # Copy original training and validation files
        augmented_files = []
        val_augmented_files = []
        
        # Copy training files
        for filename in train_files:
            base_name = filename.replace('.tif', '').replace('.png', '')
            
            # Copy original files
            for subdir in ['A', 'B']:
                src_path = self.data_root / subdir / filename
                dst_path = fold_dir / subdir / filename
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
            
            # Copy label
            for ext in ['.tif', '.png']:
                src_path = self.data_root / 'label' / f"{base_name}{ext}"
                if src_path.exists():
                    dst_path = fold_dir / 'label' / f"{base_name}{ext}"
                    shutil.copy2(src_path, dst_path)
                    break
            
            augmented_files.append(filename)
        
        # Copy validation files
        for filename in val_files:
            base_name = filename.replace('.tif', '').replace('.png', '')
            
            # Copy original files
            for subdir in ['A', 'B']:
                src_path = self.data_root / subdir / filename
                dst_path = fold_dir / subdir / filename
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
            
            # Copy label
            for ext in ['.tif', '.png']:
                src_path = self.data_root / 'label' / f"{base_name}{ext}"
                if src_path.exists():
                    dst_path = fold_dir / 'label' / f"{base_name}{ext}"
                    shutil.copy2(src_path, dst_path)
                    break
            
            val_augmented_files.append(filename)
        
        # Create heavy augmentations for training only
        augmented_count = 0
        successful_augmentations = 0
        
        for i, filename in enumerate(train_files):
            if augmented_count >= additional_needed:
                break
                
            base_name = filename.replace('.tif', '').replace('.png', '')
            
            try:
                # Load original images
                img_a_path = self.data_root / 'A' / filename
                img_b_path = self.data_root / 'B' / filename
                
                # Load images with proper multi-spectral handling
                with rasterio.open(img_a_path) as src:
                    img_a = src.read().transpose(1, 2, 0).astype(np.float32)
                    # Normalize to 0-1 range if needed
                    if img_a.max() > 1.0:
                        img_a = (img_a / img_a.max()).astype(np.float32)
                
                with rasterio.open(img_b_path) as src:
                    img_b = src.read().transpose(1, 2, 0).astype(np.float32)
                    if img_b.max() > 1.0:
                        img_b = (img_b / img_b.max()).astype(np.float32)
                
                # Load mask
                mask_paths = [self.data_root / 'label' / f"{base_name}.tif", 
                             self.data_root / 'label' / f"{base_name}.png"]
                mask_path = next((p for p in mask_paths if p.exists()), None)
                
                if mask_path is None:
                    print(f"  Warning: Mask not found for {filename}")
                    continue
                
                if mask_path.suffix == '.tif':
                    with rasterio.open(mask_path) as src:
                        mask = src.read(1).astype(np.float32)
                        mask = (mask > 0).astype(np.uint8)
                else:
                    mask = np.array(Image.open(mask_path), dtype=np.uint8)
                    if len(mask.shape) > 2:
                        mask = mask[:, :, 0]
                    mask = (mask > 0).astype(np.uint8)
                
                # Create multiple heavy augmentations
                for aug_idx in range(augmentations_per_image):
                    if augmented_count >= additional_needed:
                        break
                    
                    # Cycle through augmentation types with preference for safe ones
                    aug_type_idx = (aug_idx * 3 + i) % len(aug_types)  # Add randomness
                    aug_type = aug_types[aug_type_idx]
                    
                    # Prefer geometric and mixed for reliability
                    if random.random() < 0.4:
                        aug_type = 'geometric_heavy'
                    elif random.random() < 0.3:
                        aug_type = 'mixed_heavy'
                    elif random.random() < 0.2:
                        aug_type = 'spectral_heavy' if 'spectral_heavy' in aug_types else 'intensity_heavy'
                    
                    # Apply heavy augmentation
                    aug_img_a, aug_img_b, aug_mask = aug_pipeline.apply_heavy_augmentation(
                        img_a, img_b, mask, aug_type
                    )
                    
                    # Check if augmentation was successful (images changed)
                    if not np.array_equal(aug_img_a, img_a) or not np.array_equal(aug_img_b, img_b):
                        successful_augmentations += 1
                        
                        # Generate filename
                        aug_filename = f"{base_name}_heavy_{aug_type}_{aug_idx:03d}.tif"
                        
                        # Save augmented images as float32 for multi-spectral
                        self._save_multispectral_image(fold_dir / 'A' / aug_filename, aug_img_a)
                        self._save_multispectral_image(fold_dir / 'B' / aug_filename, aug_img_b)
                        self._save_augmented_mask(fold_dir / 'label' / aug_filename, aug_mask)
                        
                        augmented_files.append(aug_filename)
                        augmented_count += 1
                        
                        if augmented_count % 50 == 0:
                            print(f"  Created {augmented_count}/{additional_needed} heavy augmentations "
                                  f"(successful: {successful_augmentations})")
                    else:
                        print(f"  Skipped identical augmentation for {filename} (type: {aug_type})")
            
            except Exception as e:
                print(f"  Error augmenting {filename}: {e}")
                continue
        
        # Ensure we have enough samples by duplicating if needed
        current_count = len(augmented_files)
        if current_count < target_samples:
            print(f"  Only created {current_count} samples, need {target_samples}. Duplicating originals...")
            # Duplicate some originals to meet target
            originals_to_duplicate = target_samples - current_count
            for i in range(min(originals_to_duplicate, len(train_files))):
                if current_count >= target_samples:
                    break
                orig_filename = train_files[i % len(train_files)]
                aug_filename = f"{orig_filename.replace('.tif', '')}_dup_{i:03d}.tif"
                augmented_files.append(aug_filename)
                # Copy original files
                for subdir in ['A', 'B']:
                    src_path = fold_dir / subdir / orig_filename
                    dst_path = fold_dir / subdir / aug_filename
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)
                # Copy label
                for ext in ['.tif', '.png']:
                    src_path = fold_dir / 'label' / f"{orig_filename.replace('.tif', '')}{ext}"
                    if src_path.exists():
                        dst_path = fold_dir / 'label' / f"{aug_filename.replace('.tif', '')}{ext}"
                        shutil.copy2(src_path, dst_path)
                        break
                current_count += 1
        
        # Create training and validation lists for this fold
        with open(fold_dir / 'list' / 'train.txt', 'w') as f:
            for filename in augmented_files:
                f.write(f"{filename}\n")
        
        with open(fold_dir / 'list' / 'val.txt', 'w') as f:
            for filename in val_files:
                f.write(f"{filename}\n")
        
        print(f"Fold {fold_num} heavy augmentation completed:")
        print(f"  Total training samples: {len(augmented_files)}")
        print(f"  Successful heavy augmentations: {successful_augmentations}")
        print(f"  Validation samples: {len(fold_info['val_files'])}")
        
        return {
            'fold_dir': str(fold_dir),
            'train_samples': len(augmented_files),
            'val_samples': len(fold_info['val_files']),
            'augmentation_count': successful_augmentations
        }
    
    def _save_multispectral_image(self, path: Path, image: np.ndarray):
        """Save multi-spectral image preserving format"""
        dtype = 'uint8' if image.max() <= 255 else 'uint16'  # Change to strings
        image = np.clip(image, 0, 255) if dtype == 'uint8' else np.clip(image, 0, 65535)
        image = image.astype(dtype)  # Remove .name
        
        if len(image.shape) == 3:
            with rasterio.open(
                path, 'w',
                driver='GTiff',
                height=image.shape[0],
                width=image.shape[1],
                count=image.shape[2],
                dtype=dtype,  # This is already a string
                compress='lzw'
            ) as dst:
                for band_idx in range(image.shape[2]):
                    dst.write(image[:, :, band_idx], band_idx + 1)
        else:
            with rasterio.open(
                path, 'w',
                driver='GTiff',
                height=image.shape[0],
                width=image.shape[1],
                count=1,
                dtype=dtype,  # This is already a string
                compress='lzw'
            ) as dst:
                dst.write(image, 1)
    
    def _save_augmented_mask(self, path: Path, mask: np.ndarray):
        """Save augmented mask as uint8"""
        with rasterio.open(
            path, 'w',
            driver='GTiff',
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype='uint8',  # Change to string 'uint8'
            compress='lzw'
        ) as dst:
            dst.write(mask, 1)
    
    def cleanup_fold_data(self, fold_num: int):
        """Clean up fold-specific data to save disk space"""
        fold_dir = self.data_root / f'cv_fold_{fold_num}'
        if fold_dir.exists():
            shutil.rmtree(fold_dir)
            print(f"Cleaned up Fold {fold_num} data")

class CrossValidationTrainer:
    """Enhanced trainer for cross-validation with heavy augmentation"""
    
    def __init__(self, base_args, cv_manager: CrossValidationManager):
        self.base_args = base_args
        self.cv_manager = cv_manager
        self.cv_results = []
    
    def train_fold(self, fold_info: Dict, fold_data: Dict):
        """Train a single fold with heavy augmentation"""
        fold_num = fold_info['fold']
        print(f"\n{'='*60}")
        print(f"TRAINING FOLD {fold_num}/{self.cv_manager.n_folds}")
        print(f"{'='*60}")
        
        # Create fold-specific arguments
        fold_args = self._create_fold_args(fold_info, fold_data)
        
        # Set n_bands for this fold
        fold_args.n_bands = self.cv_manager.n_bands
        
        # Initialize fold-specific visualizer
        fold_visualizer = ImprovedTrainingVisualizer(
            fold_args.vis_dir, f"{fold_args.exp_name}_fold_{fold_num}"
        )
        fold_args.visualizer = fold_visualizer
        
        # Create dataloaders for this fold
        print(f"Creating dataloaders for fold {fold_num}...")
        dataloaders = utils.get_loaders(fold_args)
        
        # Initialize enhanced trainer for this fold
        print(f"Initializing enhanced trainer for fold {fold_num}...")
        model = EnhancedCDTrainer(args=fold_args, dataloaders=dataloaders)
        
        try:
            # Train the model
            print(f"Starting training for fold {fold_num}...")
            model.train_models()
            
            # Get final validation metrics
            final_val_metrics = model.validate(fold_args.max_epochs - 1)
            
            # Store fold results
            fold_result = {
                'fold': fold_num,
                'train_samples': fold_data['train_samples'],
                'val_samples': fold_data['val_samples'],
                'augmentation_count': fold_data['augmentation_count'],
                'final_metrics': final_val_metrics,
                'best_val_mf1': getattr(model, 'best_val_mf1', 0.0),
                'fold_dir': fold_data['fold_dir']
            }
            
            self.cv_results.append(fold_result)
            
            print(f"Fold {fold_num} completed successfully!")
            print(f"Final validation mF1: {final_val_metrics['mf1']:.5f}")
            print(f"Best validation mF1: {fold_result['best_val_mf1']:.5f}")
            
            return fold_result
            
        except Exception as e:
            print(f"Training failed for fold {fold_num}: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _create_fold_args(self, fold_info: Dict, fold_data: Dict):
        """Create fold-specific arguments"""
        import copy
        fold_args = copy.deepcopy(self.base_args)
        
        # Update paths for this fold
        fold_num = fold_info['fold']
        fold_args.data_root = fold_data['fold_dir']
        fold_args.exp_name = f"{self.base_args.exp_name}_Fold{fold_num}_Heavy5x"
        fold_args.checkpoint_dir = os.path.join(
            self.base_args.checkpoint_root,
            f"ChangeFormer_CV_Heavy_MultiSpectral",
            f"Fold_{fold_num}_{fold_args.exp_name}"
        )
        fold_args.vis_dir = os.path.join(self.base_args.vis_dir, "cv_heavy_augmentation_multispectral")
        
        # Create directories
        os.makedirs(fold_args.checkpoint_dir, exist_ok=True)
        os.makedirs(fold_args.vis_dir, exist_ok=True)
        
        # Disable online augmentation since we're doing offline heavy augmentation
        fold_args.augment_factor = 1.0
        
        return fold_args
    
    def run_cross_validation(self):
        """Run complete 3-fold cross-validation with heavy augmentation"""
        print("STARTING 3-FOLD CROSS-VALIDATION WITH HEAVY MULTISPECTRAL AUGMENTATION")
        print("=" * 80)
        
        # Create CV folds
        cv_folds = self.cv_manager.create_cv_folds()
        
        # Train each fold
        for fold_info in cv_folds:
            fold_num = fold_info['fold']
            
            # Create heavy augmented dataset for this fold
            fold_data = self.cv_manager.create_heavy_augmented_fold(fold_info)
            
            # Train this fold
            try:
                fold_result = self.train_fold(fold_info, fold_data)
                
                # Save fold results
                self._save_fold_results(fold_result)
                
            except Exception as e:
                print(f"Fold {fold_num} failed: {e}")
                continue
            
            # Optional: Clean up fold data to save disk space
            if hasattr(self.base_args, 'cleanup_folds') and self.base_args.cleanup_folds:
                self.cv_manager.cleanup_fold_data(fold_num)
        
        # Compute and display final CV results
        self._compute_cv_summary()
        
        return self.cv_results
    
    def _save_fold_results(self, fold_result: Dict):
        """Save individual fold results"""
        fold_num = fold_result['fold']
        results_dir = Path(self.base_args.vis_dir) / "cv_heavy_augmentation_multispectral" / "fold_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / f"fold_{fold_num}_results.json", 'w') as f:
            json.dump(fold_result, f, indent=2, default=str)
    
    def _compute_cv_summary(self):
        """Compute and display cross-validation summary"""
        if not self.cv_results:
            print("No fold results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION SUMMARY (HEAVY MULTISPECTRAL AUGMENTATION)")
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
        print(f"Heavy augmentation factor: {self.cv_manager.augment_factor}x")
        print(f"Spectral bands: {self.cv_manager.n_bands}")
        print("\nPER-FOLD RESULTS:")
        
        for i, fold_result in enumerate(self.cv_results):
            fold_num = fold_result['fold']
            metrics = fold_result['final_metrics']
            print(f"  Fold {fold_num}: mF1={metrics['mf1']:.5f}, mIoU={metrics['miou']:.5f}, "
                  f"Change_F1={metrics['F1_1']:.5f}, Change_IoU={metrics['iou_1']:.5f}, "
                  f"Train samples={fold_result['train_samples']}")
        
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
            'timestamp': datetime.now().isoformat()
        }
        
        summary_path = Path(self.base_args.vis_dir) / "cv_heavy_augmentation_multispectral" / "cv_summary.json"
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
            
            # Generate visualization of cross-validation results
            self._visualize_cv_results(summary_stats, best_fold)
    
    def _visualize_cv_results(self, summary_stats: Dict, best_fold: int):
        """Visualize cross-validation results with plots"""
        vis_dir = Path(self.base_args.vis_dir) / "cv_heavy_augmentation_multispectral" / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bar plot for key metrics
        metrics_to_plot = ['mf1', 'miou', 'F1_1', 'iou_1']
        fold_numbers = [fold_result['fold'] for fold_result in self.cv_results]
        
        plt.figure(figsize=(12, 8))
        x = np.arange(len(fold_numbers))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            if metric in summary_stats:
                values = summary_stats[metric]['values']
                plt.bar(x + i*width, values, width, label=f'{metric.upper()}', alpha=0.8)
        
        plt.xlabel('Fold Number')
        plt.ylabel('Metric Value')
        plt.title('Cross-Validation Performance Across Folds (Multispectral)')
        plt.legend()
        plt.xticks(x + width*1.5, fold_numbers)
        plt.grid(True, alpha=0.3)
        
        # Highlight best fold
        best_idx = fold_numbers.index(best_fold)
        plt.axvline(x=best_idx + width*1.5, color='red', linestyle='--', alpha=0.7, label=f'Best Fold {best_fold}')
        
        plt.tight_layout()
        plot_path = vis_dir / 'cv_performance_multispectral.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cross-validation performance plot saved to: {plot_path}")

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
                # First geometric
                aug1 = self.geometric_transforms(image=image_a, imageB=image_b, mask=mask)
                # Then intensity
                aug2 = self.intensity_transforms(image=aug1['image'], imageB=aug1['imageB'])
                # Possibly add noise
                if random.random() < 0.3:
                    aug3 = self.noise_transforms(image=aug2['image'], imageB=aug2['imageB'])
                    return aug3['image'], aug3['imageB'], aug1['mask']
                return aug2['image'], aug2['imageB'], aug1['mask']
            
        except Exception as e:
            print(f"Augmentation failed for type {aug_type}: {e}")
            return image_a, image_b, mask
        
        return image_a, image_b, mask

def create_augmented_dataset(data_root: str, target_samples: int = 525, 
                           preserve_validation: bool = True):
    """Create augmented dataset to reach target number of samples"""
    
    print(f"Creating augmented dataset with target: {target_samples} samples")
    data_path = Path(data_root)
    
    # Read original training list
    train_list_path = data_path / 'list' / 'train.txt'
    if not train_list_path.exists():
        raise FileNotFoundError(f"Training list not found: {train_list_path}")
    
    with open(train_list_path, 'r') as f:
        original_files = [line.strip() for line in f.readlines()]
    
    original_count = len(original_files)
    print(f"Original training samples: {original_count}")
    
    if original_count >= target_samples:
        print("Dataset already has enough samples")
        return original_files
    
    # Calculate how many augmentations per image we need
    additional_needed = target_samples - original_count
    augmentations_per_image = int(np.ceil(additional_needed / original_count))
    
    print(f"Need {additional_needed} additional samples")
    print(f"Creating {augmentations_per_image} augmentations per original image")
    
    # Initialize augmentation pipeline
    aug_pipeline = AdvancedAugmentationPipeline()
    
    # Define augmentation types to cycle through
    aug_types = ['geometric', 'intensity', 'combined', 'noise', 'mixed', 'weather']
    
    # Create backup of original directories
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
        
        # Load original images
        try:
            # Load image A
            img_a_path = data_path / 'A' / filename
            if img_a_path.suffix == '.tif':
                with rasterio.open(img_a_path) as src:
                    img_a = src.read().transpose(1, 2, 0)  # CHW to HWC
            else:
                img_a = np.array(Image.open(img_a_path))
            
            # Load image B
            img_b_path = data_path / 'B' / filename
            if img_b_path.suffix == '.tif':
                with rasterio.open(img_b_path) as src:
                    img_b = src.read().transpose(1, 2, 0)  # CHW to HWC
            else:
                img_b = np.array(Image.open(img_b_path))
            
            # Load mask
            mask_paths = [data_path / 'label' / f"{base_name}.tif", 
                         data_path / 'label' / f"{base_name}.png"]
            mask_path = next((p for p in mask_paths if p.exists()), None)
            
            if mask_path is None:
                print(f"Warning: Mask not found for {filename}")
                continue
            
            if mask_path.suffix == '.tif':
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)  # Single channel
            else:
                mask = np.array(Image.open(mask_path))
                if len(mask.shape) > 2:
                    mask = mask[:, :, 0]
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
        
        # Create augmentations for this image
        for aug_idx in range(augmentations_per_image):
            if augmented_count >= additional_needed:
                break
            
            aug_type = aug_types[aug_idx % len(aug_types)]
            
            try:
                # Apply augmentation
                aug_img_a, aug_img_b, aug_mask = aug_pipeline.apply_augmentation_set(
                    img_a, img_b, mask, aug_type
                )
                
                # Generate new filename
                new_filename = f"{base_name}_aug_{aug_type}_{aug_idx:02d}.tif"
                new_files.append(new_filename)
                
                # Save augmented images
                # Save image A
                aug_a_path = data_path / 'A' / new_filename
                if len(aug_img_a.shape) == 3:
                    # Multi-channel image
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
                    # Single channel
                    with rasterio.open(
                        aug_a_path, 'w',
                        driver='GTiff',
                        height=aug_img_a.shape[0],
                        width=aug_img_a.shape[1],
                        count=1,
                        dtype=aug_img_a.dtype
                    ) as dst:
                        dst.write(aug_img_a, 1)
                
                # Save image B
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
                
                # Save mask
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
                
            except Exception as e:
                print(f"Error creating augmentation {aug_idx} for {filename}: {e}")
                continue
        
        if augmented_count >= additional_needed:
            break
    
    # Update training list
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

def debug_batch_structure(dataloader, num_batches=2):
    """Debug function to understand batch structure with detailed inspection"""
    print("=== DEBUGGING BATCH STRUCTURE ===")
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
    print("=== END DEBUG ===\n")

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
        # Handle list input from multi-scale predictions
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[-1]  # Select the last prediction
        
        if not isinstance(inputs, torch.Tensor):
            raise ValueError(f"Expected inputs to be a tensor, got {type(inputs)}")
        
        # Squeeze targets to remove extra channel dimension if present
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # Convert [B, 1, H, W] to [B, H, W]
        
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
        # Handle list input from multi-scale predictions
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[-1]  # Select the last prediction
        
        # Ensure predictions is a tensor
        if not isinstance(predictions, torch.Tensor):
            raise ValueError(f"Expected predictions to be a tensor, got {type(predictions)}")
        
        # Squeeze targets to remove extra channel dimension if present
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)  # Convert [B, 1, H, W] to [B, H, W]
        
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

def calculate_detailed_metrics(pred, target):
    """Calculate detailed per-class metrics like in reference"""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Create confusion matrix
    cm = confusion_matrix(target_flat, pred_flat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Per-class IoU
    iou_0 = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0.0  # No-change IoU
    iou_1 = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0  # Change IoU
    
    # Mean IoU
    miou = (iou_0 + iou_1) / 2
    
    # Per-class F1
    f1_0 = 2 * tn / (2 * tn + fp + fn) if (2 * tn + fp + fn) > 0 else 0.0  # No-change F1
    f1_1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0  # Change F1
    
    # Mean F1
    mf1 = (f1_0 + f1_1) / 2
    
    # Per-class Precision
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # No-change precision
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Change precision
    
    # Per-class Recall
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # No-change recall
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Change recall
    
    # Overall accuracy
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

class ImprovedTrainingVisualizer:
    """Enhanced visualizer with better metrics tracking and augmented data visualization"""
    
    def __init__(self, vis_dir, exp_name):
        self.vis_dir = Path(vis_dir) / exp_name
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.vis_dir / "plots"
        self.predictions_dir = self.vis_dir / "predictions"
        self.postprocessed_dir = self.vis_dir / "postprocessed"
        self.augmented_samples_dir = self.vis_dir / "augmented_samples"  # New directory for augmented samples
        self.plots_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        self.postprocessed_dir.mkdir(exist_ok=True)
        self.augmented_samples_dir.mkdir(exist_ok=True)  # Create directory for augmented samples
        self.postprocessor = ImprovedPostProcessor()
        
        # Enhanced metrics tracking
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
        
        # Per-class metrics
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
    
    def visualize_augmented_samples(self, img_a, img_b, mask, aug_img_a, aug_img_b, aug_mask, filename, aug_type, epoch):
        """Visualize original and augmented image pairs with masks"""
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(2, 3, figure=fig)
        
        # Ensure images are in correct format for visualization
        if img_a.shape[2] == 3:  # RGB
            img_a_vis = img_a.astype(np.uint8)
            img_b_vis = img_b.astype(np.uint8)
            aug_img_a_vis = aug_img_a.astype(np.uint8)
            aug_img_b_vis = aug_img_b.astype(np.uint8)
        else:  # Multi-band, select RGB bands
            img_a_vis = img_a[:, :, [2, 1, 0]].astype(np.uint8)
            img_b_vis = img_b[:, :, [2, 1, 0]].astype(np.uint8)
            aug_img_a_vis = aug_img_a[:, :, [2, 1, 0]].astype(np.uint8)
            aug_img_b_vis = aug_img_b[:, :, [2, 1, 0]].astype(np.uint8)
        
        # Plot original images
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
        
        # Plot augmented images
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
        
        # mIoU curves
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
        
        # Per-class F1 (Change class)
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
        
        # Per-class IoU (Change class)
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
        
        # Accuracy
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
        
        # Learning rate
        ax7 = fig.add_subplot(gs[2, 0])
        if self.learning_rates:
            ax7.plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 
                    color='orange', linewidth=2)
            ax7.set_xlabel('Epoch')
            ax7.set_ylabel('Learning Rate')
            ax7.set_title('Learning Rate Schedule')
            ax7.set_yscale('log')
            ax7.grid(True, alpha=0.3)
        
        # Per-class comparison
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('off')
        
        if self.val_mf1_scores and self.val_f1_1:
            summary_text = []
            summary_text.append("DETAILED PERFORMANCE SUMMARY:")
            summary_text.append(f"Best Val mF1: {max(self.val_mf1_scores):.5f}")
            summary_text.append(f"Best Val mIoU: {max(self.val_miou_scores):.5f}")
            summary_text.append(f"Best Change F1: {max(self.val_f1_1):.5f}")
            summary_text.append(f"Best Change IoU: {max(self.val_iou_1):.5f}")
            summary_text.append(f"Final Val mF1: {self.val_mf1_scores[-1]:.5f}")
            summary_text.append(f"Final Change F1: {self.val_f1_1[-1]:.5f}")
            summary_text.append(f"Final Change IoU: {self.val_iou_1[-1]:.5f}")
            
            final_change_f1 = self.val_f1_1[-1]
            if final_change_f1 > 0.3:
                summary_text.append("Status: Good Change Detection")
            elif final_change_f1 > 0.15:
                summary_text.append("Status: Moderate Change Detection")
            else:
                summary_text.append("Status: Poor Change Detection")
                
            summary_str = "\n".join(summary_text)
            ax8.text(0.1, 0.9, summary_str, transform=ax8.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.suptitle(f'Enhanced Training Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                     fontsize=16, fontweight='bold')
        
        if save_plots:
            plt.savefig(self.plots_dir / 'enhanced_training_curves.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.plots_dir / 'enhanced_training_curves.pdf', bbox_inches='tight')
        plt.close()
        return fig
    
    def save_metrics_json(self):
        """Save enhanced metrics to JSON file"""
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
            'best_change_f1': max(self.val_f1_1) if self.val_f1_1 else 0.0
        }
        with open(self.vis_dir / 'enhanced_training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Enhanced metrics saved to: {self.vis_dir / 'enhanced_training_metrics.json'}")

class EnhancedCDTrainer(CDTrainer):
    """Enhanced CD Trainer with detailed per-class metrics and augmentation support"""
    
    def __init__(self, args, dataloaders):
        print("Initializing Enhanced CD Trainer with detailed metrics and augmentation...")
        
        # Initialize the enhanced loss function
        self.enhanced_loss_fn = EnhancedCombinedLoss(
            focal_tversky_weight=args.focal_tversky_weight,
            weighted_bce_weight=args.weighted_bce_weight,
            class_weights=args.class_weights,
            alpha=args.tversky_alpha,
            beta=args.tversky_beta,
            gamma=args.focal_gamma
        )
        
        # Initialize parent class
        super().__init__(args, dataloaders)
        
        # Replace the loss function with enhanced version
        self.loss_fn = self.enhanced_loss_fn
        
        # Initialize augmentation pipeline if enabled
        self.augmentation_pipeline = None
        if args.augment_factor > 1:
            self.augmentation_pipeline = AdvancedAugmentationPipeline(target_multiplier=args.augment_factor)
            print(f"Augmentation enabled with target multiplier: {args.augment_factor}")
        
        # Training state tracking
        self.running_mf1 = 0.0
        self.batch_count = 0
        self.epoch_start_time = time.time()
        
        print("Enhanced CD Trainer initialized with detailed metrics tracking and augmentation")
    
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
            
            # Apply augmentation if enabled
            if self.augmentation_pipeline and self.training:
                aug_types = ['geometric', 'intensity', 'combined', 'noise', 'mixed', 'weather']
                aug_type = random.choice(aug_types)
                
                # Convert tensors to numpy for augmentation
                img_A_np = img_A.cpu().numpy().transpose(0, 2, 3, 1)  # BCHW to BHWC
                img_B_np = img_B.cpu().numpy().transpose(0, 2, 3, 1)
                mask_np = mask.cpu().numpy().squeeze(1)  # Remove channel dim
                
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
                    
                    # Visualize augmented samples (every 50 batches)
                    if self.batch_count % 50 == 0 and hasattr(self.args, 'visualizer') and i < 1:
                        self.args.visualizer.visualize_augmented_samples(
                            img_A_np[i], img_B_np[i], mask_np[i],
                            aug_A, aug_B, aug_M,
                            filenames[i] if i < len(filenames) else f"sample_{i}",
                            aug_type, self.current_epoch
                        )
                
                # Convert back to tensors
                aug_img_A = torch.from_numpy(np.stack(aug_img_A).transpose(0, 3, 1, 2)).to(self.device)
                aug_img_B = torch.from_numpy(np.stack(aug_img_B).transpose(0, 3, 1, 2)).to(self.device)
                aug_mask = torch.from_numpy(np.stack(aug_mask)[:, np.newaxis, :, :]).to(self.device)
                
                # Use augmented data
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
            
            # Compute loss
            loss, ft_loss, bce_loss = self.enhanced_loss_fn(self.G_final_pred, self.L)
            
            # Compute predictions for metrics
            if self.G_final_pred.shape[1] == 1:
                pred_prob = torch.sigmoid(self.G_final_pred.squeeze(1))
            else:
                pred_prob = torch.softmax(self.G_final_pred, dim=1)[:, 1]
            
            pred = (pred_prob > 0.5).cpu().numpy()
            target = self.L.squeeze(1).cpu().numpy() if self.L.dim() == 4 else self.L.cpu().numpy()
            
            # Calculate detailed metrics
            detailed_metrics = calculate_detailed_metrics(pred, target)
            detailed_metrics['loss'] = loss.item()
            detailed_metrics['ft_loss'] = ft_loss.item()
            detailed_metrics['bce_loss'] = bce_loss.item()
            
            # Backward pass (only during training)
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
        """Enhanced training loop with detailed metrics and augmentation"""
        print("Starting enhanced training with detailed metrics and augmentation...")
        
        # Store current epoch for visualization
        self.current_epoch = 0
        
        for epoch in range(self.args.max_epochs):
            self.current_epoch = epoch
            self.net_G.train()
            self.epoch_start_time = time.time()
            self.batch_count = 0
            self.running_mf1 = 0.0
            
            # Epoch metrics accumulation
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
                    
                    # Accumulate metrics
                    for key in epoch_metrics:
                        if key in metrics:
                            epoch_metrics[key] += metrics[key]
                    
                    self.batch_count += 1
                    self.running_mf1 = epoch_metrics['mf1'] / self.batch_count
                    
                    # Progress logging
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
            
            # Compute average epoch metrics
            num_batches = self.batch_count
            for key in epoch_metrics:
                epoch_metrics[key] = epoch_metrics[key] / num_batches if num_batches > 0 else 0.0
            
            # Print epoch summary
            print(f"Is_training: True. Epoch {epoch+1} / {self.args.max_epochs}, epoch_mF1= {epoch_metrics['mf1']:.5f}")
            print(f"acc: {epoch_metrics['acc']:.5f} miou: {epoch_metrics['miou']:.5f} mf1: {epoch_metrics['mf1']:.5f} "
                  f"iou_0: {epoch_metrics['iou_0']:.5f} iou_1: {epoch_metrics['iou_1']:.5f} "
                  f"F1_0: {epoch_metrics['F1_0']:.5f} F1_1: {epoch_metrics['F1_1']:.5f} "
                  f"precision_0: {epoch_metrics['precision_0']:.5f} precision_1: {epoch_metrics['precision_1']:.5f} "
                  f"recall_0: {epoch_metrics['recall_0']:.5f} recall_1: {epoch_metrics['recall_1']:.5f}")
            
            # Validation phase
            val_metrics = None
            if epoch % self.args.val_freq == 0 and 'val' in self.dataloaders:
                print("\nBegin evaluation...")
                val_metrics = self.validate(epoch)
            
            # Update visualizer
            if hasattr(self.args, 'visualizer'):
                lr = self.optimizer_G.param_groups[0]['lr']
                self.args.visualizer.update_metrics(epoch, epoch_metrics, val_metrics, lr)
                if epoch % self.args.vis_freq == 0:
                    self.args.visualizer.plot_enhanced_training_curves()
                    self.args.visualizer.save_metrics_json()
            
            # Save checkpoint
            if epoch % self.args.save_epoch_freq == 0:
                checkpoint_path = os.path.join(self.args.checkpoint_dir, f'epoch_{epoch+1}.pt')
                torch.save(self.net_G.state_dict(), checkpoint_path)
                print(f"Saved checkpoint: epoch_{epoch+1}.pt")
            
            # Save best model based on validation mF1
            if val_metrics and hasattr(self, 'best_val_mf1'):
                if val_metrics['mf1'] > self.best_val_mf1:
                    self.best_val_mf1 = val_metrics['mf1']
                    best_path = os.path.join(self.args.checkpoint_dir, 'best_ckpt.pt')
                    torch.save(self.net_G.state_dict(), best_path)
                    print(f"New best validation mF1: {self.best_val_mf1:.5f} - saved best_ckpt.pt")
            elif not hasattr(self, 'best_val_mf1'):
                self.best_val_mf1 = val_metrics['mf1'] if val_metrics else 0.0
        
        print("Enhanced training completed!")
    
    def validate(self, epoch):
        """Enhanced validation with detailed metrics"""
        self.net_G.eval()
        val_start_time = time.time()
        batch_count = 0
        running_mf1 = 0.0
        
        # Validation metrics accumulation
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
                    
                    # Accumulate metrics
                    for key in val_metrics:
                        if key in metrics:
                            val_metrics[key] += metrics[key]
                    
                    batch_count += 1
                    running_mf1 = val_metrics['mf1'] / batch_count
                    
                    # Progress logging
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
        
        # Compute average validation metrics
        for key in val_metrics:
            val_metrics[key] = val_metrics[key] / batch_count if batch_count > 0 else 0.0
        
        # Print validation summary
        print(f"Is_training: False. Epoch {epoch+1} / {self.args.max_epochs}, epoch_mF1= {val_metrics['mf1']:.5f}")
        print(f"acc: {val_metrics['acc']:.5f} miou: {val_metrics['miou']:.5f} mf1: {val_metrics['mf1']:.5f} "
              f"iou_0: {val_metrics['iou_0']:.5f} iou_1: {val_metrics['iou_1']:.5f} "
              f"F1_0: {val_metrics['F1_0']:.5f} F1_1: {val_metrics['F1_1']:.5f} "
              f"precision_0: {val_metrics['precision_0']:.5f} precision_1: {val_metrics['precision_1']:.5f} "
              f"recall_0: {val_metrics['recall_0']:.5f} recall_1: {val_metrics['recall_1']:.5f}")
        
        return val_metrics

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
    
    # Count actual training samples with augmentation factor
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

def enhanced_train(args):
    """Enhanced training function with improvements for class imbalance and augmentation"""
    print("ENHANCED CHANGEFORMER TRAINING FOR EXTREME CLASS IMBALANCE")
    print("=" * 70)
    
    if not validate_args(args):
        print("Argument validation failed. Exiting.")
        exit(1)
    
    if args.fix_nan_values:
        fix_dataset_nan_values(args.data_root)
    
    # Create augmented dataset if augment_factor > 1
    if args.augment_factor > 1:
        create_augmented_dataset(args.data_root, target_samples=int(args.augment_factor * 194), preserve_validation=True)
    
    if args.auto_class_weights:
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
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize enhanced visualizer
    visualizer = ImprovedTrainingVisualizer(args.vis_dir, args.exp_name)
    args.visualizer = visualizer
    
    print(f"Using Enhanced Combined Loss:")
    print(f"   Focal Tversky weight: {args.focal_tversky_weight}")
    print(f"   Weighted BCE weight: {args.weighted_bce_weight}")
    print(f"   Class weights: {args.class_weights}")
    print("=" * 70)
    
    print(f"Creating loaders with data_root: {args.data_root}")
    try:
        dataloaders = utils.get_loaders(args)
        print(f"Training samples loaded: {len(dataloaders['train'].dataset) if hasattr(dataloaders['train'], 'dataset') else 'Unknown'}")
        if 'val' in dataloaders:
            print(f"Validation samples loaded: {len(dataloaders['val'].dataset) if hasattr(dataloaders['val'], 'dataset') else 'Unknown'}")
    except Exception as e:
        print(f"Failed to create dataloaders: {e}")
        raise e
    
    # Convert class weights to tensor
    args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
    
    # Initialize enhanced trainer
    model = EnhancedCDTrainer(args=args, dataloaders=dataloaders)
    
    try:
        model.train_models()
    except Exception as e:
        print(f"Training failed: {e}")
        raise e
    
    print("Creating final enhanced training visualizations...")
    visualizer.plot_enhanced_training_curves()
    visualizer.save_metrics_json()
    
    return model

def enhanced_cross_validation_train(args):
    """Enhanced cross-validation training with heavy multispectral augmentation"""
    print("ENHANCED CROSS-VALIDATION CHANGEFORMER TRAINING (MULTISPECTRAL)")
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
    args.cleanup_folds = getattr(args, 'cleanup_folds', False)
    
    # Set number of bands for Cartosat-3 (adjust if different)
    args.n_bands = 4  # Pan + RGB
    
    # Initialize cross-validation manager with multispectral support
    cv_manager = CrossValidationManager(
        data_root=args.data_root,
        n_folds=args.n_folds,
        augment_factor=args.cv_augment_factor,
        random_state=args.random_state,
        n_bands=args.n_bands
    )
    
    # Initialize cross-validation trainer
    cv_trainer = CrossValidationTrainer(base_args=args, cv_manager=cv_manager)
    
    # Run cross-validation
    try:
        cv_results = cv_trainer.run_cross_validation()
        print(f"\nCross-validation completed! Results: {len(cv_results)} successful folds")
        return cv_results
    except Exception as e:
        print(f"Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

if __name__ == '__main__':
    parser = ArgumentParser(description="Enhanced ChangeFormer Training for Extreme Class Imbalance with Cross-Validation")

    # Data settings
    parser.add_argument('--data_name', type=str, default='cartoCustom')
    parser.add_argument('--data_root', type=str, default='./data/cartoCustom')
    parser.add_argument('--data_format', type=str, default='tif')
    parser.add_argument('--split', type=str, default='list')
    parser.add_argument('--dataset', type=str, default='CDDataset')
    parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_Enhanced_CV_Heavy_MultiSpectral_248')

    # Cartosat-3 specific parameters
    parser.add_argument('--satellite_type', type=str, default='cartosat')
    parser.add_argument('--rgb_bands', type=int, nargs=3, default=[2, 1, 0])
    parser.add_argument('--normalize_method', type=str, default='adaptive_percentile')
    parser.add_argument('--n_bands', type=int, default=4, help='Number of spectral bands')

    # Enhanced Model config for class imbalance
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--net_G', type=str, default='ChangeFormerV6')
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--n_class', type=int, default=2)

    # Enhanced Training params for class imbalance
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-5)
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
    parser.add_argument('--tversky_alpha', type=float, default=0.2)  # Less penalty for FP
    parser.add_argument('--tversky_beta', type=float, default=0.8)   # More penalty for FN
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    
    # Enhanced checkpointing and visualization
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/ChangeFormer_myCustom/ChangeFormerV6_Enhanced_CV_Heavy_MultiSpectral_248')
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

    # Cross-validation specific parameters
    parser.add_argument('--n_folds', type=int, default=2)
    parser.add_argument('--cv_augment_factor', type=int, default=5)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--cleanup_folds', type=bool, default=True)
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

    print("ENHANCED CHANGEFORMER FOR EXTREME CLASS IMBALANCE WITH MULTISPECTRAL CROSS-VALIDATION")
    print("=" * 80)
    print("KEY IMPROVEMENTS FOR CARTOSAT-3 MULTISPECTRAL DATA:")
    print(f"Enhanced Loss: Focal Tversky ({args.focal_tversky_weight}) + Weighted BCE ({args.weighted_bce_weight})")
    print(f"Tversky params: alpha={args.tversky_alpha} (FP penalty), beta={args.tversky_beta} (FN penalty)")
    print(f"Stronger backbone: {args.backbone}")
    print(f"Input size: {args.img_size}x{args.img_size}")
    print(f"Spectral bands: {args.n_bands}")
    print(f"Detailed per-class metrics tracking")
    print(f"Cross-validation: {args.n_folds}-fold with {args.cv_augment_factor}x multispectral augmentation")
    print("=" * 80)
    print(f"Dataset: {args.data_name} (Cartosat-3)")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Device: {args.device}")
    print("=" * 80)
    
    try:
        if args.mode == 'cv':
            enhanced_cross_validation_train(args)
            print("\nENHANCED MULTISPECTRAL CROSS-VALIDATION TRAINING COMPLETED SUCCESSFULLY!")
        else:
            enhanced_train(args)
            print("\nENHANCED TRAINING COMPLETED SUCCESSFULLY!")
        
        print(f"Results saved in: {args.vis_dir}/{args.exp_name}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()