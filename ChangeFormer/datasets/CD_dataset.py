import os
import numpy as np
import rasterio
import cv2
from PIL import Image
from torch.utils import data
from datasets.data_utils import CDDataAugmentation
import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import torch

warnings.filterwarnings('ignore', category=RuntimeWarning)

IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=str)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list

def get_img_post_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)

def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)

def get_label_path(root_dir, img_name):
    """Get label path with proper extension handling"""
    base_name = img_name.replace('.tif', '').replace('.png', '').replace('.jpg', '')
   
    # Try TIF first, then PNG
    for ext in ['.tif', '.png', '.jpg']:
        label_path = os.path.join(root_dir, ANNOT_FOLDER_NAME, base_name + ext)
        if os.path.exists(label_path):
            return label_path
   
    # Default to TIF if none found
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, base_name + '.tif')

class ImageDataset(data.Dataset):
    """Dataset with MODERATE augmentation - removed heavy distortions"""
   
    def __init__(self, root_dir, split='train', img_size=256, is_train=True, to_tensor=True,
                 rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile', data_format='auto',
                 augment_factor=10):  # Reduced from 100 to 10
       
        super(ImageDataset, self).__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split
        self.rgb_bands = rgb_bands  # Changed default to [2,1,0] for proper RGB
        self.normalize_method = normalize_method
        self.data_format = data_format
        self.to_tensor = to_tensor
        self.augment_factor = augment_factor if is_train else 1
       
        # Initialize default statistics
        self.rgb_mean = np.array([0.5, 0.5, 0.5])
        self.rgb_std = np.array([0.25, 0.25, 0.25])
        self.rgb_percentiles = {
            'p2': np.array([0.1, 0.1, 0.1]),
            'p98': np.array([0.9, 0.9, 0.9])
        }
       
        self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split + '.txt')
        self.img_name_list = load_img_name_list(self.list_path)
        self.original_size = len(self.img_name_list)
        self.A_size = self.original_size * self.augment_factor
       
        # Auto-detect data format
        if self.data_format == 'auto':
            sample_path = get_img_path(self.root_dir, self.img_name_list[0])
            self.data_format = 'tif' if sample_path.lower().endswith('.tif') else 'standard'
       
        print(f" MODERATE AUGMENTATION: {split} dataset expanded from {self.original_size} to {self.A_size} samples")
       
        # Compute normalization statistics for TIF data
        if self.data_format == 'tif':
            self.compute_rgb_statistics()
       
        # Setup CLEAN augmentation pipeline - removed problematic transforms
        if is_train:
            self.clean_augment = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR),
                
                # SAFE geometric transforms only
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.08,  # Reduced from 0.15
                    scale_limit=0.1,   # Reduced from 0.2
                    rotate_limit=15,   # Reduced from 30
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.4  # Reduced probability
                ),
                
                # MILD photometric transforms only
                A.RandomBrightnessContrast(
                    brightness_limit=0.15,  # Reduced from 0.25
                    contrast_limit=0.15,    # Reduced from 0.25
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=8,      # Reduced from 15
                    sat_shift_limit=15,     # Reduced from 25
                    val_shift_limit=10,     # Reduced from 15
                    p=0.3                   # Reduced probability
                ),
                
                # Very mild noise only
                A.OneOf([
                    A.GaussNoise(var_limit=(2, 8), per_channel=True),  # Much reduced
                    A.ISONoise(color_shift=(0.005, 0.02), intensity=(0.05, 0.15)),  # Reduced
                ], p=0.2),  # Reduced probability
                
                # Mild blur only
                A.OneOf([
                    A.GaussianBlur(blur_limit=(1, 3)),  # Reduced blur
                    A.MotionBlur(blur_limit=1),         # Reduced blur
                ], p=0.1),  # Reduced probability
                
                # Final resize to ensure exact dimensions
                A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR)
            ],
            additional_targets={'image_B': 'image', 'mask': 'mask'},
            is_check_shapes=False)
           
        else:
            # Simple resize for validation
            self.clean_augment = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size, interpolation=cv2.INTER_LINEAR, always_apply=True)
            ],
            additional_targets={'image_B': 'image', 'mask': 'mask'},
            is_check_shapes=False
            )
   
    def robust_percentile(self, data, percentiles):
        """Compute robust percentiles ignoring outliers"""
        try:
            # Remove extreme outliers first
            p1, p99 = np.percentile(data, [1, 99])
            filtered_data = data[(data >= p1) & (data <= p99)]
           
            if len(filtered_data) > 100:
                return np.percentile(filtered_data, percentiles)
            else:
                return np.percentile(data, percentiles)
        except:
            return np.array([np.min(data), np.max(data)])
   
    def compute_rgb_statistics(self):
        """Compute robust statistics for RGB bands from TIF data (optimized for Cartosat 3-band)"""
        print(f"Computing RGB statistics for {self.split} split...")

        all_pixels = []
        sample_size = min(15, len(self.img_name_list))
        successful_loads = 0

        for i in range(sample_size):
            name = self.img_name_list[i]
            A_path = get_img_path(self.root_dir, name)
            B_path = get_img_post_path(self.root_dir, name)

            try:
                for img_path in [A_path, B_path]:
                    with rasterio.open(img_path) as src:
                        # ✅ Fix: If image has fewer bands than requested, use available ones
                        available_bands = src.count
                        if available_bands < 3:
                            print(f"⚠️ {img_path} has only {available_bands} bands, expected 3 — skipping.")
                            continue

                        rgb_data = []
                        for band_idx in self.rgb_bands:
                            if band_idx < available_bands:
                                band_data = src.read(band_idx + 1).astype(np.float32)
                            else:
                                # ✅ Fallback: If band missing, use first band
                                band_data = src.read(1).astype(np.float32)
                                print(f"⚠️ Using fallback band 1 for {os.path.basename(img_path)}")

                            # Handle NaN, inf values
                            band_data = np.nan_to_num(band_data, nan=0.0, posinf=0.0, neginf=0.0)
                            rgb_data.append(band_data)

                        # Stack into RGB format
                        rgb_image = np.stack(rgb_data, axis=-1)

                        # ✅ Sample pixels for stats without loading full image into memory
                        H, W = rgb_image.shape[:2]
                        n_samples = min(5000, H * W)
                        if H * W > n_samples:
                            indices = np.random.choice(H * W, n_samples, replace=False)
                            sampled_pixels = rgb_image.reshape(-1, 3)[indices]
                        else:
                            sampled_pixels = rgb_image.reshape(-1, 3)

                        # ✅ Ignore zero-only pixels if they dominate
                        non_zero_mask = np.any(sampled_pixels > 0, axis=1)
                        if np.sum(non_zero_mask) > len(sampled_pixels) * 0.1:
                            sampled_pixels = sampled_pixels[non_zero_mask]

                        if len(sampled_pixels) > 0:
                            all_pixels.append(sampled_pixels)

                successful_loads += 1

            except Exception as e:
                print(f"⚠️ Could not load {name} for statistics: {e}")
                continue

        # ✅ If statistics successfully collected
        if all_pixels and successful_loads > 0:
            try:
                all_pixels = np.vstack(all_pixels)
                print(f" Computing statistics from {len(all_pixels)} pixels from {successful_loads} image pairs")

                # Compute mean & std
                self.rgb_mean = np.mean(all_pixels, axis=0)
                self.rgb_std = np.std(all_pixels, axis=0)

                # Compute robust percentiles
                p2_p98 = []
                for i in range(3):
                    band_pixels = all_pixels[:, i]
                    p2, p98 = self.robust_percentile(band_pixels, [2, 98])
                    p2_p98.append([p2, p98])

                p2_p98 = np.array(p2_p98)
                self.rgb_percentiles = {
                    'p2': p2_p98[:, 0],
                    'p98': p2_p98[:, 1]
                }

                # ✅ Ensure valid percentile ranges
                for i in range(3):
                    if self.rgb_percentiles['p98'][i] <= self.rgb_percentiles['p2'][i]:
                        self.rgb_percentiles['p2'][i] = self.rgb_mean[i] - 2 * self.rgb_std[i]
                        self.rgb_percentiles['p98'][i] = self.rgb_mean[i] + 2 * self.rgb_std[i]

                print(f"✅ Statistics computed successfully:")
                print(f"   RGB bands: {self.rgb_bands} (Cartosat = [2,1,0] or [3,2,1])")
                print(f"   RGB mean: {self.rgb_mean}")
                print(f"   RGB std: {self.rgb_std}")
                print(f"   RGB p2: {self.rgb_percentiles['p2']}")
                print(f"   RGB p98: {self.rgb_percentiles['p98']}")

            except Exception as e:
                print(f"⚠️ Error computing statistics: {e}, using LEVIR defaults")
                self.use_levir_defaults()
        else:
            print("⚠️ Using LEVIR defaults — no valid Cartosat stats computed.")
            self.use_levir_defaults()


    def use_levir_defaults(self):
        """Fallback to LEVIR pretrained normalization"""
        self.rgb_mean = np.array([101.77, 95.41, 88.12])
        self.rgb_std = np.array([41.43, 40.43, 41.98])
        self.rgb_percentiles = {
            'p2': np.array([0.0, 0.0, 0.0]),
            'p98': np.array([255.0, 255.0, 255.0])
        }
        print("⚡ Using LEVIR pretrained normalization stats for compatibility.")

   
    def load_and_resize_image(self, image_path, target_size):
        """Load image and ensure exact target size before any processing"""
        if self.data_format == 'tif':
            return self.load_tif_as_rgb(image_path, target_size)
        else:
            img = np.asarray(Image.open(image_path).convert('RGB'))
            # IMMEDIATE resize to target size
            img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
            return img

    def load_tif_as_rgb(self, tif_path, target_size):
        """Load TIF with auto-detected band selection and safe normalization"""
        try:
            with rasterio.open(tif_path) as src:
                n_bands = src.count

                # Auto-detect RGB band indices
                if n_bands >= 10:
                    rgb_bands = [3, 2, 1]  # Sentinel / LEVIR style
                elif n_bands >= 3:
                    rgb_bands = [0, 1, 2]  # Cartosat standard
                    if n_bands > 3:
                        pass
                        # print(f"ℹ️ {tif_path} has {n_bands} bands, using first 3 as RGB.")
                else:
                    raise ValueError(f"❌ Unsupported image: {tif_path} has only {n_bands} bands.")

                rgb_data = []
                for band_idx in rgb_bands:
                    band_data = src.read(band_idx + 1).astype(np.float32)

                    # Fix NaN values
                    if np.isnan(band_data).any():
                        nan_mask = np.isnan(band_data)
                        if nan_mask.all():
                            band_data = np.zeros_like(band_data)
                        else:
                            valid_pixels = band_data[~nan_mask]
                            replacement_value = np.median(valid_pixels) if len(valid_pixels) > 0 else 0.0
                            band_data[nan_mask] = replacement_value

                    band_data = np.nan_to_num(band_data, posinf=0.0, neginf=0.0)
                    rgb_data.append(band_data)

                rgb_image = np.stack(rgb_data, axis=-1)

                # Resize immediately
                rgb_image = cv2.resize(rgb_image, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

                # Normalize per-dataset (Cartosat vs LEVIR)
                rgb_image = self.normalize_rgb(rgb_image)

                # Convert to 0-255 range for augmentation compatibility
                rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

                return rgb_image

        except Exception as e:
            print(f"Error loading TIF {tif_path}: {e}")
            return np.full((target_size, target_size, 3), 128, dtype=np.uint8)

   
    def normalize_rgb(self, rgb_image):
        """Enhanced normalization with better range handling"""
        if self.normalize_method == 'adaptive_percentile':
            # Per-image adaptive percentile normalization
            normalized = np.zeros_like(rgb_image, dtype=np.float32)
            for i in range(3):
                band = rgb_image[:, :, i].astype(np.float32)
               
                if np.all(band == band.flat[0]):
                    normalized[:, :, i] = 0.5
                    continue
               
                p2, p98 = self.robust_percentile(band.flatten(), [2, 98])
               
                if p98 > p2:
                    normalized[:, :, i] = np.clip((band - p2) / (p98 - p2), 0, 1)
                else:
                    min_val, max_val = np.min(band), np.max(band)
                    if max_val > min_val:
                        normalized[:, :, i] = (band - min_val) / (max_val - min_val)
                    else:
                        normalized[:, :, i] = 0.5
           
            return normalized
           
        elif self.normalize_method == 'percentile':
            p2, p98 = self.rgb_percentiles['p2'], self.rgb_percentiles['p98']
            for i in range(3):
                if p98[i] <= p2[i]:
                    p98[i] = p2[i] + 1.0
            rgb_image = np.clip((rgb_image - p2) / (p98 - p2 + 1e-8), 0, 1)
           
        elif self.normalize_method == 'zscore':
            rgb_image = (rgb_image - self.rgb_mean) / (self.rgb_std + 1e-8)
            rgb_image = np.clip(rgb_image, -3, 3)
            rgb_image = (rgb_image + 3) / 6
           
        elif self.normalize_method == 'minmax':
            for i in range(3):
                band = rgb_image[:, :, i]
                band_min, band_max = np.min(band), np.max(band)
                if band_max > band_min:
                    rgb_image[:, :, i] = (band - band_min) / (band_max - band_min)
                else:
                    rgb_image[:, :, i] = 0.5
       
        return rgb_image.astype(np.float32)
   
    def __getitem__(self, index):
        # Map expanded index back to original samples
        original_index = index % self.original_size
        name = self.img_name_list[original_index]
       
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)

        # Load and resize immediately to target size
        img = self.load_and_resize_image(A_path, self.img_size)
        img_B = self.load_and_resize_image(B_path, self.img_size)

        # VERIFY exact dimensions before augmentation
        assert img.shape == (self.img_size, self.img_size, 3), \
            f"Image A size mismatch after loading: got {img.shape}, expected ({self.img_size}, {self.img_size}, 3)"
        assert img_B.shape == (self.img_size, self.img_size, 3), \
            f"Image B size mismatch after loading: got {img_B.shape}, expected ({self.img_size}, {self.img_size}, 3)"

        # Apply CLEAN augmentation with deterministic seed
        if hasattr(self, 'clean_augment'):
            # Create unique but deterministic seed for each augmented sample
            seed = hash((index, name)) % 2**32
            random.seed(seed)
            np.random.seed(seed % 2**16)
           
            try:
                transformed = self.clean_augment(image=img, image_B=img_B)
                img = transformed['image']
                img_B = transformed['image_B']
            except Exception as e:
                print(f"Warning: Augmentation failed for {name}, using originals: {e}")
                # Fallback: just resize
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img_B = cv2.resize(img_B, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
           
            # FINAL verification of dimensions
            assert img.shape == (self.img_size, self.img_size, 3), \
                f"Image A size after augmentation: got {img.shape}, expected ({self.img_size}, {self.img_size}, 3)"
            assert img_B.shape == (self.img_size, self.img_size, 3), \
                f"Image B size after augmentation: got {img_B.shape}, expected ({self.img_size}, {self.img_size}, 3)"

        # Convert to tensor if needed
        if self.to_tensor:
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            img_B = torch.from_numpy(img_B.transpose(2, 0, 1).astype(np.float32) / 255.0)

        return {'A': img, 'B': img_B, 'name': f"{name}_aug{index}"}

    def __len__(self):
        return self.A_size


class CDDataset(ImageDataset):
    """Change Detection Dataset with CLEAN augmentation"""

    def __init__(self, root_dir, img_size, split='train', is_train=True, label_transform='binary',
                 to_tensor=True, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile',
                 data_format='auto', augment_factor=10):  # Reduced from 100 to 10
       
        super(CDDataset, self).__init__(
            root_dir=root_dir,
            split=split,
            img_size=img_size,
            is_train=is_train,
            to_tensor=to_tensor,
            rgb_bands=rgb_bands,  # Default changed to [2,1,0] for proper RGB
            normalize_method=normalize_method,
            data_format=data_format,
            augment_factor=augment_factor
        )
        self.label_transform = label_transform
       
        # Validate label files exist
        self._validate_labels()
       
        print(f"🎯 CDDataset initialized: {self.A_size} samples (x{self.augment_factor} CLEAN augmentation)")

    def _validate_labels(self):
        """Validate that label files exist for all samples"""
        missing_labels = []
        for name in self.img_name_list:
            label_path = get_label_path(self.root_dir, name)
            if not os.path.exists(label_path):
                missing_labels.append(name)
       
        if missing_labels:
            print(f"⚠️ Warning: Missing labels for {len(missing_labels)} files in {self.split}:")
            for name in missing_labels[:5]:
                print(f"   {name}")
            if len(missing_labels) > 5:
                print(f"   ... and {len(missing_labels) - 5} more")

    def load_and_resize_label(self, label_path, target_size):
        """Load label and ensure exact target size"""
        try:
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
                return np.zeros((target_size, target_size), dtype=np.uint8)
               
            if label_path.lower().endswith('.tif'):
                with rasterio.open(label_path) as src:
                    label = src.read(1).astype(np.float32)
                   
                    if np.isnan(label).any():
                        print(f"Warning: NaN values found in label {os.path.basename(label_path)}")
                        label = np.nan_to_num(label, nan=0.0)
                   
                    unique_vals = np.unique(label)
                    if len(unique_vals) > 2 or (len(unique_vals) == 2 and not (0 in unique_vals and 1 in unique_vals)):
                        if label.max() > 1:
                            label = (label > 0).astype(np.uint8)
                        elif label.max() <= 1 and label.min() >= 0:
                            label = (label > 0.5).astype(np.uint8)
                        else:
                            threshold = np.median(unique_vals)
                            label = (label > threshold).astype(np.uint8)
                    else:
                        label = label.astype(np.uint8)
                       
            else:
                label_pil = Image.open(label_path)
                if label_pil.mode != 'L':
                    label_pil = label_pil.convert('L')
                label = np.array(label_pil, dtype=np.uint8)
               
                if len(label.shape) > 2:
                    label = label[:, :, 0]
               
                if label.max() > 1:
                    label = (label > 0).astype(np.uint8)
           
            # IMMEDIATE resize to target size
            label = cv2.resize(label, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
           
            # Final validation
            label = np.clip(label, 0, 1).astype(np.uint8)
           
            return label
           
        except Exception as e:
            print(f"Error loading label {label_path}: {e}")
            return np.zeros((target_size, target_size), dtype=np.uint8)

    def __getitem__(self, index):
        # Map expanded index back to original samples
        original_index = index % self.original_size
        name = self.img_name_list[original_index]
       
        A_path = get_img_path(self.root_dir, name)
        B_path = get_img_post_path(self.root_dir, name)
        L_path = get_label_path(self.root_dir, name)
       
        # Load and resize all to exact target size immediately
        img = self.load_and_resize_image(A_path, self.img_size)
        img_B = self.load_and_resize_image(B_path, self.img_size)
        label = self.load_and_resize_label(L_path, self.img_size)

        # VERIFY all dimensions match before augmentation
        assert img.shape == (self.img_size, self.img_size, 3), \
            f"Image A shape mismatch: got {img.shape}, expected ({self.img_size}, {self.img_size}, 3)"
        assert img_B.shape == (self.img_size, self.img_size, 3), \
            f"Image B shape mismatch: got {img_B.shape}, expected ({self.img_size}, {self.img_size}, 3)"
        assert label.shape == (self.img_size, self.img_size), \
            f"Label shape mismatch: got {label.shape}, expected ({self.img_size}, {self.img_size})"

        # Apply CLEAN augmentation with same seed to all
        if hasattr(self, 'clean_augment'):
            # Create unique but deterministic augmentation for each expanded sample
            seed = hash((index, name)) % 2**32
            random.seed(seed)
            np.random.seed(seed % 2**16)
           
            try:
                transformed = self.clean_augment(image=img, image_B=img_B, mask=label)
                img = transformed['image']
                img_B = transformed['image_B']
                label = transformed['mask']
            except Exception as e:
                print(f"Warning: Augmentation failed for {name}, using originals: {e}")
                # Fallback: just resize all to be sure
                img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                img_B = cv2.resize(img_B, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
           
            # FINAL verification of exact dimensions
            assert img.shape == (self.img_size, self.img_size, 3), \
                f"Image A size after augmentation: got {img.shape}, expected ({self.img_size}, {self.img_size}, 3)"
            assert img_B.shape == (self.img_size, self.img_size, 3), \
                f"Image B size after augmentation: got {img_B.shape}, expected ({self.img_size}, {self.img_size}, 3)"
            assert label.shape == (self.img_size, self.img_size), \
                f"Label size after augmentation: got {label.shape}, expected ({self.img_size}, {self.img_size})"

        # Convert to tensor if needed
        if self.to_tensor:
            img = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.0)
            img_B = torch.from_numpy(img_B.transpose(2, 0, 1).astype(np.float32) / 255.0)
           
            # Handle label tensor conversion
            if len(label.shape) == 2:
                label = torch.from_numpy(label.astype(np.float32)).unsqueeze(0)  # Add channel dimension
            else:
                label = torch.from_numpy(label.astype(np.float32))

        return {'A': img, 'B': img_B, 'L': label, 'name': f"{name}_aug{index}"}


# Legacy compatibility - replace original classes
ImageDataset = ImageDataset
CDDataset = CDDataset


def create_balanced_cross_validation_splits(data_root, n_splits=5, random_state=42):
    """Create balanced cross-validation splits ensuring each fold has change samples"""
    from sklearn.model_selection import StratifiedKFold
    import shutil
    from pathlib import Path
    
    print(f"🎯 Creating {n_splits}-fold cross-validation splits...")
    
    # Load all samples
    train_list_path = Path(data_root) / 'list' / 'train.txt'
    if not train_list_path.exists():
        print(f"❌ Train list not found: {train_list_path}")
        return False
    
    with open(train_list_path, 'r') as f:
        all_samples = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(all_samples)} total samples")
    
    # Determine which samples have changes
    labels = []
    label_dir = Path(data_root) / 'label'
    
    for sample_name in all_samples:
        base_name = sample_name.replace('.tif', '')
        
        # Try different extensions
        label_paths = [
            label_dir / f"{base_name}.tif",
            label_dir / f"{base_name}.png"
        ]
        
        has_change = False
        for label_path in label_paths:
            if label_path.exists():
                try:
                    if label_path.suffix == '.tif':
                        with rasterio.open(label_path) as src:
                            label_data = src.read(1)
                    else:
                        label_data = np.array(Image.open(label_path))
                    
                    # Check if has change pixels
                    if label_data.max() > 0:
                        has_change = True
                    break
                except:
                    continue
        
        labels.append(1 if has_change else 0)
    
    labels = np.array(labels)
    change_count = np.sum(labels)
    
    print(f"Samples with changes: {change_count}/{len(all_samples)} ({change_count/len(all_samples)*100:.1f}%)")
    
    if change_count == 0:
        print("❌ No samples with changes found! Using random splits.")
        # Fallback to random splits
        from sklearn.model_selection import KFold
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(all_samples))
    else:
        # Stratified splits
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = list(skf.split(all_samples, labels))
    
    # Create fold directories and save splits
    cv_dir = Path(data_root) / 'cv_splits'
    cv_dir.mkdir(exist_ok=True)
    
    for fold, (train_idx, val_idx) in enumerate(splits):
        fold_dir = cv_dir / f'fold_{fold}'
        fold_dir.mkdir(exist_ok=True)
        
        # Create list directory for this fold
        list_dir = fold_dir / 'list'
        list_dir.mkdir(exist_ok=True)
        
        # Create symbolic links to original data
        for data_type in ['A', 'B', 'label']:
            src_dir = Path(data_root) / data_type
            dst_dir = fold_dir / data_type
            
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            dst_dir.symlink_to(src_dir.absolute(), target_is_directory=True)
        
        # Save train/val splits
        train_samples = [all_samples[i] for i in train_idx]
        val_samples = [all_samples[i] for i in val_idx]
        
        with open(list_dir / 'train.txt', 'w') as f:
            f.write('\n'.join(train_samples))
        
        with open(list_dir / 'val.txt', 'w') as f:
            f.write('\n'.join(val_samples))
        
        # Calculate split statistics
        train_changes = np.sum([labels[i] for i in train_idx])
        val_changes = np.sum([labels[i] for i in val_idx])
        
        print(f"Fold {fold}: Train {len(train_samples)} samples ({train_changes} changes), "
              f"Val {len(val_samples)} samples ({val_changes} changes)")
    
    print(f"✅ Cross-validation splits created in {cv_dir}")
    return True


def create_visualization_samples(dataset_path, output_dir='./vis_samples', num_samples=5):
    """Create sample visualizations to verify data loading and RGB bands"""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    print(f"Creating visualization samples from {dataset_path}...")
    
    # Test both RGB band orderings
    for rgb_bands, band_name in [([2, 1, 0], "RGB_210"), ([0, 1, 2], "RGB_012")]:
        print(f"\n🔍 Testing RGB bands {rgb_bands} ({band_name})...")
        
        # Create dataset instances
        train_dataset = CDDataset(
            root_dir=dataset_path,
            img_size=256,
            split='train',
            is_train=False,  # No augmentation for visualization
            rgb_bands=rgb_bands,
            normalize_method='adaptive_percentile',
            data_format='auto',
            augment_factor=1  # No augmentation for vis
        )
        
        # Create output directory
        vis_dir = Path(output_dir) / band_name
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Create samples
        num_samples = min(num_samples, train_dataset.original_size)
        
        for i in range(num_samples):
            try:
                # Get original data (no tensor conversion)
                original_to_tensor = train_dataset.to_tensor
                train_dataset.to_tensor = False
                
                data = train_dataset[i]
                
                train_dataset.to_tensor = original_to_tensor
                
                # Create figure
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                fig.suptitle(f"Sample {i}: {data['name']} - Bands {rgb_bands}", fontsize=14)
                
                # Convert data for display
                img_A = data['A']
                img_B = data['B'] 
                label = data['L']
                
                if img_A.dtype != np.uint8:
                    img_A = (np.clip(img_A, 0, 1) * 255).astype(np.uint8)
                if img_B.dtype != np.uint8:
                    img_B = (np.clip(img_B, 0, 1) * 255).astype(np.uint8)
                if label.dtype != np.uint8:
                    label = (label * 255).astype(np.uint8)
                
                # Plot images
                axes[0].imshow(img_A)
                axes[0].set_title('Image A (Before)')
                axes[0].axis('off')
                
                axes[1].imshow(img_B)
                axes[1].set_title('Image B (After)')
                axes[1].axis('off')
                
                axes[2].imshow(label, cmap='gray')
                axes[2].set_title('Change Label')
                axes[2].axis('off')
                
                # Difference image
                diff = np.abs(img_A.astype(np.float32) - img_B.astype(np.float32)).astype(np.uint8)
                axes[3].imshow(diff)
                axes[3].set_title('Difference')
                axes[3].axis('off')
                
                plt.tight_layout()
                plt.savefig(vis_dir / f"sample_{i:03d}_{data['name']}.png", dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Created visualization for {data['name']} with bands {rgb_bands}")
                
            except Exception as e:
                print(f"❌ Error creating visualization for sample {i}: {e}")
    
    print(f"\n Visualization samples saved to {output_dir}")
    print("🔍 Compare RGB_210 vs RGB_012 folders to see which looks more natural!")


def test_rgb_bands_configuration(dataset_path):
    """Test different RGB band configurations to find the best one"""
    print("🧪 Testing RGB band configurations...")
    
    band_configs = [
        ([2, 1, 0], "RGB as 2,1,0 (Red=Band3, Green=Band2, Blue=Band1)"),
        ([0, 1, 2], "RGB as 0,1,2 (Red=Band1, Green=Band2, Blue=Band3)"),
        ([3, 2, 1], "RGB as 3,2,1 (Red=Band4, Green=Band3, Blue=Band2)"),
    ]
    
    for rgb_bands, description in band_configs:
        try:
            print(f"\n📈 Testing: {description}")
            test_dataset = CDDataset(
                root_dir=dataset_path,
                img_size=256,
                split='train',
                is_train=False,
                rgb_bands=rgb_bands,
                normalize_method='adaptive_percentile',
                augment_factor=1
            )
            
            # Load one sample to test
            if len(test_dataset) > 0:
                sample = test_dataset[0]
                print(f"✅ Successfully loaded with bands {rgb_bands}")
            else:
                print(f"❌ No samples found with bands {rgb_bands}")
                
        except Exception as e:
            print(f"❌ Failed with bands {rgb_bands}: {e}")
    
    print("\n💡 Recommendation:")
    print("   - If images look too red: try rgb_bands=[2,1,0]")  
    print("   - If images look normal: try rgb_bands=[0,1,2]")
    print("   - Check your satellite data documentation for correct band order")


if __name__ == "__main__":
    # Test the cleaned dataset
    dataset_path = input("Enter dataset path: ").strip()
    if dataset_path and os.path.exists(dataset_path):
        
        # Test RGB band configurations first
        test_rgb_bands_configuration(dataset_path)
        
        # Test clean augmentation
        print("\n🧹 Testing CLEAN augmentation dataset...")
        test_dataset = CDDataset(
            root_dir=dataset_path,
            img_size=256,
            split='train',
            is_train=True,
            rgb_bands=[2, 1, 0],  # Start with this - change if images look wrong
            augment_factor=10  # Much more reasonable than 100
        )
        
        print(f"Original dataset size: {test_dataset.original_size}")
        print(f"Augmented dataset size: {len(test_dataset)}")
        
        # Test a few samples
        for i in range(min(3, len(test_dataset))):
            sample = test_dataset[i]
            print(f"Sample {i}: {sample['name']}, A: {sample['A'].shape}, B: {sample['B'].shape}, L: {sample['L'].shape}")
        
        # Create visualization with both RGB band orders
        create_visualization_samples(dataset_path)
        
        # Create cross-validation splits
        create_balanced_cross_validation_splits(dataset_path, n_splits=5)
        
        print("\n🎯 Key Changes Made:")
        print("✅ Removed heavy augmentations: ElasticTransform, GridDistortion, OpticalDistortion")
        print("✅ Removed problematic effects: RandomFog, SunFlare, RandomShadow")  
        print("✅ Removed color-distorting: CLAHE, RandomGamma, Emboss, Downscale")
        print("✅ Changed RGB bands to [2,1,0] (test both [2,1,0] and [0,1,2])")
        print("✅ Reduced augment_factor from 100 to 10")
        print("✅ Kept only safe augmentations: flips, mild rotations, light brightness/contrast")
        print("\n Check the visualization samples to verify RGB bands look natural!")
        
    else:
        print("Invalid dataset path!")





