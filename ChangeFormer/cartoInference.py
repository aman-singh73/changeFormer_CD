# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# import rasterio
# from PIL import Image
# import torch.nn.functional as F
# import warnings
# import cv2
# from scipy import ndimage
# from argparse import ArgumentParser
# from models.ChangeFormer import ChangeFormerV6
# # Suppress warnings
# warnings.filterwarnings('ignore')
# import matplotlib
# matplotlib.use('Agg')   # use non-interactive backend so plt.show() won't block
# import sys
# plt.ioff()  

# class PostProcessor:
#     """Postprocessing for change detection predictions"""
    
#     def __init__(self, 
#                  min_component_size=15,
#                  morphology_kernel_size=3,
#                  gaussian_sigma=0.5):
#         self.min_component_size = min_component_size
#         self.morphology_kernel_size = morphology_kernel_size
#         self.gaussian_sigma = gaussian_sigma
        
#     def postprocess_prediction(self, prediction):
#         """Apply postprocessing to prediction"""
#         # Ensure binary prediction
#         if prediction.max() <= 1.0:
#             binary_pred = (prediction > 0.5).astype(np.uint8)
#         else:
#             binary_pred = (prediction > 0).astype(np.uint8)
            
#         # Morphological operations
#         processed = self._apply_morphological_ops(binary_pred)
        
#         # Connected component filtering
#         processed = self._filter_small_components(processed)
        
#         # Final smoothing
#         processed = self._smooth_boundaries(processed)
        
#         return processed
    
#     def _apply_morphological_ops(self, binary_mask):
#         """Apply morphological operations"""
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
#                                          (self.morphology_kernel_size, self.morphology_kernel_size))
        
#         # Opening to remove noise
#         opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
#         # Closing to fill holes
#         closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
#         return closed
    
#     def _filter_small_components(self, binary_mask):
#         """Remove small connected components"""
#         labeled_mask, num_features = ndimage.label(binary_mask)
#         component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
#         mask_sizes = component_sizes >= self.min_component_size
#         remove_pixel = mask_sizes[labeled_mask]
        
#         filtered_mask = binary_mask.copy()
#         filtered_mask[~remove_pixel] = 0
        
#         return filtered_mask
    
#     def _smooth_boundaries(self, binary_mask):
#         """Smooth boundaries using Gaussian filter"""
#         if self.gaussian_sigma > 0:
#             smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=self.gaussian_sigma)
#             return (smoothed > 0.5).astype(np.uint8)
#         return binary_mask

# def load_tif_image(image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
#     """
#     Load and preprocess TIF image for Cartosat-3
    
#     For Cartosat-3 images, typical band order is:
#     - Band 0: Blue
#     - Band 1: Green  
#     - Band 2: Red
#     - Band 3: NIR (if available)
    
#     For proper RGB visualization: [2, 1, 0] (Red, Green, Blue)
#     """
#     with rasterio.open(image_path) as src:
#         # Read all bands
#         image_data = src.read()  # Shape: (bands, height, width)
        
#         print(f"Image bands available: {image_data.shape[0]}")
#         print(f"Image shape: {image_data.shape}")
        
#         # Handle NaN values
#         if np.isnan(image_data).any():
#             for band_idx in range(image_data.shape[0]):
#                 band_data = image_data[band_idx]
#                 if np.isnan(band_data).any():
#                     valid_pixels = band_data[~np.isnan(band_data)]
#                     if len(valid_pixels) > 0:
#                         replacement_value = np.median(valid_pixels)
#                     else:
#                         replacement_value = 0.0
#                     band_data[np.isnan(band_data)] = replacement_value
        
#         # For Cartosat-3, adjust band selection based on available bands
#         if image_data.shape[0] >= 3:
#             # Use specified RGB bands (default [2,1,0] for proper color)
#             if all(0 <= b < image_data.shape[0] for b in rgb_bands):
#                 rgb_data = image_data[rgb_bands]
#                 print(f"Using bands {rgb_bands} for RGB visualization")
#             else:
#                 # Fallback: take first 3 bands in reverse order for better color
#                 rgb_data = image_data[:3][::-1]  # Reverse for better color representation
#                 print("Using first 3 bands (reversed) for RGB")
#         else:
#             # If less than 3 bands, duplicate the available bands
#             if image_data.shape[0] == 1:
#                 rgb_data = np.repeat(image_data, 3, axis=0)
#             else:
#                 rgb_data = image_data
        
#         # Enhanced normalization for satellite imagery
#         if normalize_method == 'adaptive_percentile':
#             for i in range(rgb_data.shape[0]):
#                 band = rgb_data[i]
#                 # Use more robust percentile normalization
#                 valid_pixels = band[band > 0]
#                 if len(valid_pixels) > 100:  # Ensure enough valid pixels
#                     p1, p99 = np.percentile(valid_pixels, [1, 99])
#                     # Avoid division by zero
#                     if p99 > p1:
#                         rgb_data[i] = np.clip((band - p1) / (p99 - p1), 0, 1)
#                     else:
#                         rgb_data[i] = np.clip(band / (band.max() + 1e-8), 0, 1)
#                 else:
#                     # Fallback normalization
#                     band_max = band.max()
#                     if band_max > 0:
#                         rgb_data[i] = band / band_max
#                     else:
#                         rgb_data[i] = band
        
#         return rgb_data  # Shape: (3, height, width)

# def preprocess_for_model(image, target_size=256):
#     """Preprocess image for model input"""
#     # Resize if needed
#     if image.shape[1] != target_size or image.shape[2] != target_size:
#         image_tensor = torch.from_numpy(image).unsqueeze(0).float()
#         image_tensor = F.interpolate(image_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
#         image = image_tensor.squeeze(0).numpy()
    
#     # Convert to tensor and add batch dimension
#     image_tensor = torch.from_numpy(image).unsqueeze(0).float()
    
#     return image_tensor

# def _extract_logits_from_output(output):
#     """Return a single logits tensor [B,C,H,W] from model output which may be a tensor/list/tuple."""
#     # If it's already a tensor, return it
#     if torch.is_tensor(output):
#         return output

#     # If it's a list/tuple, try to find a good candidate
#     if isinstance(output, (list, tuple)):
#         # 1) Prefer last tensor that looks like logits with C==1 or C==2
#         for o in reversed(output):
#             if torch.is_tensor(o) and o.dim() == 4 and o.size(1) in (1, 2):
#                 return o
#         # 2) Otherwise prefer last 4D tensor found
#         for o in reversed(output):
#             if torch.is_tensor(o) and o.dim() == 4:
#                 return o
#         # 3) Otherwise pick last tensor in the list (if any)
#         for o in reversed(output):
#             if torch.is_tensor(o):
#                 return o

#     raise ValueError("Could not extract a logits tensor from model output (output type: {}).".format(type(output)))

# def create_comparison_visualization(img1, img2, raw_prediction, processed_prediction, save_path=None):
#     """Create comparison visualization with proper cleanup"""
#     # Create figure with explicit backend
#     plt.ioff()  # Turn off interactive mode
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
#     # Convert images for display (C, H, W) -> (H, W, C)
#     img1_display = np.transpose(img1, (1, 2, 0))
#     img2_display = np.transpose(img2, (1, 2, 0))
    
#     # Row 1: Input images and raw prediction
#     axes[0, 0].imshow(img1_display)
#     axes[0, 0].set_title('Image T1 (Cartosat-3)', fontsize=14, fontweight='bold')
#     axes[0, 0].axis('off')
    
#     axes[0, 1].imshow(img2_display)
#     axes[0, 1].set_title('Image T2 (Cartosat-3)', fontsize=14, fontweight='bold')
#     axes[0, 1].axis('off')
    
#     axes[0, 2].imshow(raw_prediction, cmap='Reds', alpha=0.8)
#     axes[0, 2].set_title('Raw Change Detection', fontsize=14, fontweight='bold')
#     axes[0, 2].axis('off')
    
#     # Row 2: Processed prediction, overlay, and statistics
#     axes[1, 0].imshow(processed_prediction, cmap='Greens', alpha=0.8)
#     axes[1, 0].set_title('Processed Change Detection', fontsize=14, fontweight='bold')
#     axes[1, 0].axis('off')
    
#     # Create overlay with better visualization
#     overlay = img2_display.copy()
#     change_mask = processed_prediction > 0
#     if np.any(change_mask):
#         # Make changes more visible with yellow/red overlay
#         overlay[change_mask, 0] = np.minimum(overlay[change_mask, 0] + 0.3, 1.0)  # Red
#         overlay[change_mask, 1] = np.minimum(overlay[change_mask, 1] + 0.3, 1.0)  # Green (for yellow)
    
#     axes[1, 1].imshow(overlay)
#     axes[1, 1].set_title('Change Overlay on T2', fontsize=14, fontweight='bold')
#     axes[1, 1].axis('off')
    
#     # Statistics
#     axes[1, 2].axis('off')
    
#     raw_change_pixels = np.sum(raw_prediction > 0)
#     processed_change_pixels = np.sum(processed_prediction > 0)
#     total_pixels = raw_prediction.size
    
#     raw_change_percent = (raw_change_pixels / total_pixels) * 100
#     processed_change_percent = (processed_change_pixels / total_pixels) * 100
    
#     stats_text = f"""CHANGE DETECTION STATISTICS

# Image Type: Cartosat-3

# Raw Prediction:
# • Change pixels: {raw_change_pixels:,}
# • Total pixels: {total_pixels:,}
# • Change area: {raw_change_percent:.2f}%

# Processed Prediction:
# • Change pixels: {processed_change_pixels:,}
# • Total pixels: {total_pixels:,}
# • Change area: {processed_change_percent:.2f}%

# Postprocessing Effect:
# • Pixel difference: {processed_change_pixels - raw_change_pixels:,}
# • Area difference: {processed_change_percent - raw_change_percent:.2f}%"""
    
#     axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
#                    fontsize=10, verticalalignment='top', fontfamily='monospace',
#                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
#         print(f"Results saved to: {save_path}")
    
#     # Important: Close the figure to free memory and prevent hanging
#     plt.close(fig)
    
#     return fig

# def load_model_smart(model_path, device='cpu'):
#     """Smart model loading that tries different approaches"""
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
#     print(f"Loading model from: {model_path}")
    
#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=device)
    
#     print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
#     # Try to extract model state dict
#     model_state = None
#     if 'G_state_dict' in checkpoint:
#         model_state = checkpoint['G_state_dict']
#         print("Found 'G_state_dict' in checkpoint")
#     elif 'model_state_dict' in checkpoint:
#         model_state = checkpoint['model_state_dict']
#         print("Found 'model_state_dict' in checkpoint")
#     elif 'model' in checkpoint:
#         model_state = checkpoint['model']
#         print("Found 'model' in checkpoint")
#     elif 'state_dict' in checkpoint:
#         model_state = checkpoint['state_dict']
#         print("Found 'state_dict' in checkpoint")
#     elif 'model_G_state_dict' in checkpoint:
#         model_state = checkpoint['model_G_state_dict']
#         print("Found 'model_G_state_dict' in checkpoint")
#     else:
#         print("Treating entire checkpoint as model state dict")
#         model_state = checkpoint
    
#     if model_state is None:
#         raise RuntimeError("Could not find model state dict in checkpoint")
    
#     # Print some layer names to understand the model structure
#     layer_names = list(model_state.keys())[:10]
#     print(f"First 10 layer names: {layer_names}")
    
#     return model_state

# def inference_with_actual_model(img1_path, img2_path, model_path, output_dir='./inference_results'):
#     """Run inference with the actual ChangeFormerV6 model"""
    
#     # Setup
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {device}")
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load the actual model
#     try:
#         print("Loading ChangeFormerV6 model...")
        
#         # Initialize your actual model with the correct parameters
#         model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(device)
        
#         # Load model state dict
#         model_state = load_model_smart(model_path, device)
#         model.load_state_dict(model_state, strict=False)  # Use strict=False in case of minor key mismatches
#         model.eval()
        
#         print("✓ Successfully loaded ChangeFormerV6 model")
        
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Available keys in checkpoint:")
#         try:
#             checkpoint = torch.load(model_path, map_location=device)
#             print(list(checkpoint.keys()))
#         except:
#             pass
#         return None, None
    
#     # Load and preprocess images with proper Cartosat-3 band mapping
#     print("Loading Cartosat-3 images...")
#     img1 = load_tif_image(img1_path, rgb_bands=[2, 1, 0])
#     img2 = load_tif_image(img2_path, rgb_bands=[2, 1, 0])
    
#     print(f"Image 1 shape: {img1.shape}")
#     print(f"Image 2 shape: {img2.shape}")
#     print(f"Image 1 range: {img1.min():.3f} - {img1.max():.3f}")
#     print(f"Image 2 range: {img2.min():.3f} - {img2.max():.3f}")
    
#     # Preprocess for model
#     img1_tensor = preprocess_for_model(img1).to(device)
#     img2_tensor = preprocess_for_model(img2).to(device)
    
#     # Run inference with ACTUAL model
#     print("Running inference with ChangeFormerV6...")
#     with torch.no_grad():
#         output = model(img1_tensor, img2_tensor)
        
#         # Debug output structure
#         if isinstance(output, (list, tuple)):
#             print("Model returned a list/tuple with lengths:", len(output))
#             for i, el in enumerate(output):
#                 if torch.is_tensor(el):
#                     print(f" - element {i}: tensor shape {tuple(el.shape)}")
#                 else:
#                     print(f" - element {i}: type {type(el)}")
#         else:
#             print(f"Model returned tensor shape: {tuple(output.shape)}")

#         # Extract logits tensor robustly
#         try:
#             logits = _extract_logits_from_output(output)
#         except Exception as e:
#             raise RuntimeError(f"Failed to extract logits from model output: {e}")

#         print(f"Using logits tensor with shape: {tuple(logits.shape)}")

#         # Process logits to probability map
#         if logits.dim() == 4:
#             B, C, H, W = logits.shape
#             if C == 2:
#                 prediction_prob = F.softmax(logits, dim=1)[:, 1, :, :]    # positive class probability
#             elif C == 1:
#                 prediction_prob = torch.sigmoid(logits[:, 0, :, :])       # single-channel logits
#             else:
#                 prediction_prob = F.softmax(logits, dim=1)[:, 1, :, :]    # fallback to class 1
#         elif logits.dim() == 3:
#             prediction_prob = logits.squeeze(1) if logits.shape[1] == 1 else logits
#             prediction_prob = torch.sigmoid(prediction_prob) if prediction_prob.max() > 1 else prediction_prob
#         else:
#             raise RuntimeError(f"Unexpected logits dimensionality: {logits.dim()}")

#         # Convert to numpy (remove batch dimension)
#         prediction_np = prediction_prob.cpu().numpy()[0]  # [H, W]

#     print(f"Prediction shape: {prediction_np.shape}")
#     print(f"Prediction range: {prediction_np.min():.4f} - {prediction_np.max():.4f}")
    
#     # Apply postprocessing
#     postprocessor = PostProcessor()
#     raw_binary = (prediction_np > 0.5).astype(np.uint8)
#     processed_prediction = postprocessor.postprocess_prediction(prediction_np)
    
#     # Create visualization
#     output_path = os.path.join(output_dir, 'change_detection_results.png')
#     fig = create_comparison_visualization(
#         img1, img2, raw_binary, processed_prediction, 
#         save_path=output_path
#     )
    
#     # Save individual results
#     raw_path = os.path.join(output_dir, 'raw_prediction.png')
#     processed_path = os.path.join(output_dir, 'processed_prediction.png')
    
#     # Raw prediction
#     plt.ioff()
#     plt.figure(figsize=(8, 8))
#     plt.imshow(raw_binary, cmap='Reds')
#     plt.title('Raw Change Detection (Cartosat-3)', fontsize=14, fontweight='bold')
#     plt.axis('off')
#     plt.savefig(raw_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     # Processed prediction
#     plt.figure(figsize=(8, 8))
#     plt.imshow(processed_prediction, cmap='Greens')
#     plt.title('Processed Change Detection (Cartosat-3)', fontsize=14, fontweight='bold')
#     plt.axis('off')
#     plt.savefig(processed_path, dpi=300, bbox_inches='tight', facecolor='white')
#     plt.close()
    
#     print(f"Raw prediction saved to: {raw_path}")
#     print(f"Processed prediction saved to: {processed_path}")
#     print("\n✓ REAL INFERENCE COMPLETED with ChangeFormerV6!")
    
#     return prediction_np, processed_prediction

# if __name__ == '__main__':
#     # Configuration
#     MODEL_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_myCustom\ChangeFormerV6_Enhanced_Strong\best_ckpt.pt"
#     IMG1_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\A\0001.tif"
#     IMG2_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\B\0001.tif"
#     OUTPUT_DIR = "./cartoinference_results"
    
#     # Verify paths exist
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: Model checkpoint not found at {MODEL_PATH}")
#         sys.exit(1)
    
#     if not os.path.exists(IMG1_PATH):
#         print(f"Error: Image 1 not found at {IMG1_PATH}")
#         sys.exit(1)
        
#     if not os.path.exists(IMG2_PATH):
#         print(f"Error: Image 2 not found at {IMG2_PATH}")
#         sys.exit(1)
    
#     print("ChangeFormer Inference for Cartosat-3 Images")
#     print("=" * 60)
#     print(f"Model: {MODEL_PATH}")
#     print(f"Image 1: {IMG1_PATH}")
#     print(f"Image 2: {IMG2_PATH}")
#     print(f"Output: {OUTPUT_DIR}")
#     print("=" * 60)
    
#     # Run inference with ACTUAL model (no more dummy model or helper function)
#     try:
#         raw_pred, processed_pred = inference_with_actual_model(IMG1_PATH, IMG2_PATH, MODEL_PATH, OUTPUT_DIR)
#         if raw_pred is not None:
#             print("\n" + "=" * 60)
#             print("REAL INFERENCE COMPLETED SUCCESSFULLY!")
#             print("=" * 60)
#             print("✓ Used actual ChangeFormerV6 model")
#             print("✓ Images loaded with proper Cartosat-3 band mapping")
#             print("✓ Real change detection results generated")
#             print("✓ All matplotlib figures closed properly")
#             print("✓ Process will now exit cleanly")
#     except Exception as e:
#         print(f"Inference failed: {e}")
#         print("\nTroubleshooting suggestions:")
#         print("1. Make sure your 'models' folder is in the same directory as this script")
#         print("2. Check if the import 'from models.ChangeFormer import ChangeFormerV6' works")
#         print("3. Verify that your checkpoint file contains the correct keys")
#         print("4. Make sure all required model architecture files are available")
#         import traceback
#         traceback.print_exc()
#     finally:
#         # Ensure all matplotlib resources are cleaned up
#         plt.close('all')
#         print("\nProcess ending...")
#         sys.exit(0)


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
from PIL import Image
import torch.nn.functional as F
import warnings
import cv2
from scipy import ndimage
from argparse import ArgumentParser
from models.ChangeFormer import ChangeFormerV6
import json
from datetime import datetime
import time
# Suppress warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')   # use non-interactive backend so plt.show() won't block
import sys
plt.ioff()  

class PostProcessor:
    """Postprocessing for change detection predictions"""
    
    def __init__(self, 
                 min_component_size=15,
                 morphology_kernel_size=3,
                 gaussian_sigma=0.5):
        self.min_component_size = min_component_size
        self.morphology_kernel_size = morphology_kernel_size
        self.gaussian_sigma = gaussian_sigma
        
    def postprocess_prediction(self, prediction):
        """Apply postprocessing to prediction"""
        # Ensure binary prediction
        if prediction.max() <= 1.0:
            binary_pred = (prediction > 0.5).astype(np.uint8)
        else:
            binary_pred = (prediction > 0).astype(np.uint8)
            
        # Morphological operations
        processed = self._apply_morphological_ops(binary_pred)
        
        # Connected component filtering
        processed = self._filter_small_components(processed)
        
        # Final smoothing
        processed = self._smooth_boundaries(processed)
        
        return processed
    
    def _apply_morphological_ops(self, binary_mask):
        """Apply morphological operations"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morphology_kernel_size, self.morphology_kernel_size))
        
        # Opening to remove noise
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _filter_small_components(self, binary_mask):
        """Remove small connected components"""
        labeled_mask, num_features = ndimage.label(binary_mask)
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
        mask_sizes = component_sizes >= self.min_component_size
        remove_pixel = mask_sizes[labeled_mask]
        
        filtered_mask = binary_mask.copy()
        filtered_mask[~remove_pixel] = 0
        
        return filtered_mask
    
    def _smooth_boundaries(self, binary_mask):
        """Smooth boundaries using Gaussian filter"""
        if self.gaussian_sigma > 0:
            smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=self.gaussian_sigma)
            return (smoothed > 0.5).astype(np.uint8)
        return binary_mask

class BatchProcessor:
    """Handles batch processing of change detection inference"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.postprocessor = PostProcessor()
        self.results = []
        
        print(f"Initializing BatchProcessor with device: {self.device}")
        
    def load_model(self):
        """Load the ChangeFormerV6 model once for batch processing"""
        if self.model is not None:
            return
            
        try:
            print("Loading ChangeFormerV6 model...")
            
            # Initialize model
            self.model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(self.device)
            
            # Load model state dict
            model_state = self._load_model_smart(self.model_path, self.device)
            self.model.load_state_dict(model_state, strict=False)
            self.model.eval()
            
            print("✓ Successfully loaded ChangeFormerV6 model")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def _load_model_smart(self, model_path, device='cpu'):
        """Smart model loading that tries different approaches"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Try to extract model state dict
        model_state = None
        if 'G_state_dict' in checkpoint:
            model_state = checkpoint['G_state_dict']
            print("Found 'G_state_dict' in checkpoint")
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            print("Found 'model_state_dict' in checkpoint")
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
            print("Found 'model' in checkpoint")
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            print("Found 'state_dict' in checkpoint")
        elif 'model_G_state_dict' in checkpoint:
            model_state = checkpoint['model_G_state_dict']
            print("Found 'model_G_state_dict' in checkpoint")
        else:
            print("Treating entire checkpoint as model state dict")
            model_state = checkpoint
        
        if model_state is None:
            raise RuntimeError("Could not find model state dict in checkpoint")
        
        return model_state
    
    def process_batch(self, validation_ids, data_dir_A, data_dir_B, output_dir):
        """Process a batch of validation image pairs"""
        
        # Load model once
        self.load_model()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories for organized results
        results_dirs = {
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'raw_predictions': os.path.join(output_dir, 'raw_predictions'),
            'processed_predictions': os.path.join(output_dir, 'processed_predictions'),
            'overlays': os.path.join(output_dir, 'overlays'),
            'reports': os.path.join(output_dir, 'reports')
        }
        
        for dir_path in results_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        print(f"\nStarting batch processing of {len(validation_ids)} image pairs...")
        print("=" * 70)
        
        batch_stats = {
            'total_pairs': len(validation_ids),
            'processed_pairs': 0,
            'failed_pairs': 0,
            'total_processing_time': 0,
            'results': []
        }
        
        start_time = time.time()
        
        for idx, image_id in enumerate(validation_ids):
            pair_start_time = time.time()
            
            try:
                print(f"\nProcessing pair {idx + 1}/{len(validation_ids)}: {image_id}")
                print("-" * 50)
                
                # Construct image paths
                img1_path = os.path.join(data_dir_A, f"{image_id}.tif")
                img2_path = os.path.join(data_dir_B, f"{image_id}.tif")
                
                # Verify files exist
                if not os.path.exists(img1_path):
                    print(f"⚠️  Warning: Image A not found: {img1_path}")
                    batch_stats['failed_pairs'] += 1
                    continue
                    
                if not os.path.exists(img2_path):
                    print(f"⚠️  Warning: Image B not found: {img2_path}")
                    batch_stats['failed_pairs'] += 1
                    continue
                
                # Process single pair
                result = self._process_single_pair(
                    image_id, img1_path, img2_path, results_dirs
                )
                
                if result:
                    pair_time = time.time() - pair_start_time
                    result['processing_time'] = pair_time
                    batch_stats['results'].append(result)
                    batch_stats['processed_pairs'] += 1
                    
                    print(f"✓ Completed {image_id} in {pair_time:.2f}s")
                    
                    # Print progress
                    progress = (idx + 1) / len(validation_ids) * 100
                    print(f" Overall Progress: {progress:.1f}% ({idx + 1}/{len(validation_ids)})")
                else:
                    batch_stats['failed_pairs'] += 1
                    
            except Exception as e:
                print(f"❌ Error processing {image_id}: {e}")
                batch_stats['failed_pairs'] += 1
                continue
        
        batch_stats['total_processing_time'] = time.time() - start_time
        
        # Generate batch summary report
        self._generate_batch_report(batch_stats, results_dirs['reports'])
        
        print("\n" + "=" * 70)
        print("🎯 BATCH PROCESSING COMPLETED!")
        print("=" * 70)
        print(f"✅ Successfully processed: {batch_stats['processed_pairs']}/{batch_stats['total_pairs']} pairs")
        print(f"❌ Failed pairs: {batch_stats['failed_pairs']}")
        print(f"⏱️  Total time: {batch_stats['total_processing_time']:.2f}s")
        print(f" Average time per pair: {batch_stats['total_processing_time']/max(batch_stats['processed_pairs'], 1):.2f}s")
        print(f"📁 Results saved to: {output_dir}")
        
        return batch_stats
    
    def _process_single_pair(self, image_id, img1_path, img2_path, results_dirs):
        """Process a single image pair"""
        
        try:
            # Load and preprocess images
            img1 = self._load_tif_image(img1_path, rgb_bands=[2, 1, 0])
            img2 = self._load_tif_image(img2_path, rgb_bands=[2, 1, 0])
            
            # Preprocess for model
            img1_tensor = self._preprocess_for_model(img1).to(self.device)
            img2_tensor = self._preprocess_for_model(img2).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img1_tensor, img2_tensor)
                
                # Extract logits tensor robustly
                logits = self._extract_logits_from_output(output)
                
                # Process logits to probability map
                if logits.dim() == 4:
                    B, C, H, W = logits.shape
                    if C == 2:
                        prediction_prob = F.softmax(logits, dim=1)[:, 1, :, :]
                    elif C == 1:
                        prediction_prob = torch.sigmoid(logits[:, 0, :, :])
                    else:
                        prediction_prob = F.softmax(logits, dim=1)[:, 1, :, :]
                elif logits.dim() == 3:
                    prediction_prob = logits.squeeze(1) if logits.shape[1] == 1 else logits
                    prediction_prob = torch.sigmoid(prediction_prob) if prediction_prob.max() > 1 else prediction_prob
                else:
                    raise RuntimeError(f"Unexpected logits dimensionality: {logits.dim()}")
                
                # Convert to numpy
                prediction_np = prediction_prob.cpu().numpy()[0]
            
            # Apply postprocessing
            raw_binary = (prediction_np > 0.5).astype(np.uint8)
            processed_prediction = self.postprocessor.postprocess_prediction(prediction_np)
            
            # Save results
            self._save_pair_results(
                image_id, img1, img2, raw_binary, processed_prediction, results_dirs
            )
            
            # Calculate statistics
            raw_change_pixels = np.sum(raw_binary > 0)
            processed_change_pixels = np.sum(processed_prediction > 0)
            total_pixels = raw_binary.size
            
            result = {
                'image_id': image_id,
                'raw_change_pixels': int(raw_change_pixels),
                'processed_change_pixels': int(processed_change_pixels),
                'total_pixels': int(total_pixels),
                'raw_change_percent': float(raw_change_pixels / total_pixels * 100),
                'processed_change_percent': float(processed_change_pixels / total_pixels * 100),
                'postprocess_effect': float((processed_change_pixels - raw_change_pixels) / total_pixels * 100)
            }
            
            return result
            
        except Exception as e:
            print(f"Error in _process_single_pair for {image_id}: {e}")
            return None
    
    def _save_pair_results(self, image_id, img1, img2, raw_pred, processed_pred, results_dirs):
        """Save all results for a single pair"""
        
        # 1. Comprehensive visualization
        plt.ioff()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Convert images for display (C, H, W) -> (H, W, C)
        img1_display = np.transpose(img1, (1, 2, 0))
        img2_display = np.transpose(img2, (1, 2, 0))
        
        # Row 1: Input images and raw prediction
        axes[0, 0].imshow(img1_display)
        axes[0, 0].set_title(f'Image T1 - {image_id}', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_display)
        axes[0, 1].set_title(f'Image T2 - {image_id}', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(raw_pred, cmap='Reds', alpha=0.8)
        axes[0, 2].set_title('Raw Change Detection', fontsize=14, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Processed prediction, overlay, and statistics
        axes[1, 0].imshow(processed_pred, cmap='Greens', alpha=0.8)
        axes[1, 0].set_title('Processed Change Detection', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Create overlay
        overlay = img2_display.copy()
        change_mask = processed_pred > 0
        if np.any(change_mask):
            overlay[change_mask, 0] = np.minimum(overlay[change_mask, 0] + 0.3, 1.0)
            overlay[change_mask, 1] = np.minimum(overlay[change_mask, 1] + 0.3, 1.0)
        
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Change Overlay on T2', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Statistics
        axes[1, 2].axis('off')
        
        raw_change_pixels = np.sum(raw_pred > 0)
        processed_change_pixels = np.sum(processed_pred > 0)
        total_pixels = raw_pred.size
        
        raw_change_percent = (raw_change_pixels / total_pixels) * 100
        processed_change_percent = (processed_change_pixels / total_pixels) * 100
        
        stats_text = f"""CHANGE DETECTION STATISTICS
Image ID: {image_id}
Image Type: Cartosat-3

Raw Prediction:
• Change pixels: {raw_change_pixels:,}
• Total pixels: {total_pixels:,}
• Change area: {raw_change_percent:.2f}%

Processed Prediction:
• Change pixels: {processed_change_pixels:,}
• Total pixels: {total_pixels:,}
• Change area: {processed_change_percent:.2f}%

Postprocessing Effect:
• Pixel difference: {processed_change_pixels - raw_change_pixels:,}
• Area difference: {processed_change_percent - raw_change_percent:.2f}%"""
        
        axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        viz_path = os.path.join(results_dirs['visualizations'], f'{image_id}_complete_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        # 2. Save individual prediction maps
        # Raw prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_pred, cmap='Reds')
        plt.title(f'Raw Change Detection - {image_id}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.colorbar(shrink=0.6)
        raw_path = os.path.join(results_dirs['raw_predictions'], f'{image_id}_raw.png')
        plt.savefig(raw_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Processed prediction
        plt.figure(figsize=(10, 10))
        plt.imshow(processed_pred, cmap='Greens')
        plt.title(f'Processed Change Detection - {image_id}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.colorbar(shrink=0.6)
        processed_path = os.path.join(results_dirs['processed_predictions'], f'{image_id}_processed.png')
        plt.savefig(processed_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 3. Save overlay
        plt.figure(figsize=(12, 12))
        plt.imshow(overlay)
        plt.title(f'Change Detection Overlay - {image_id}', fontsize=16, fontweight='bold')
        plt.axis('off')
        overlay_path = os.path.join(results_dirs['overlays'], f'{image_id}_overlay.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _generate_batch_report(self, batch_stats, reports_dir):
        """Generate comprehensive batch processing report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report_path = os.path.join(reports_dir, f'batch_report_{timestamp}.json')
        with open(json_report_path, 'w') as f:
            json.dump(batch_stats, f, indent=2)
        
        # Text summary report
        text_report_path = os.path.join(reports_dir, f'batch_summary_{timestamp}.txt')
        
        with open(text_report_path, 'w') as f:
            f.write("CHANGEFORMER BATCH PROCESSING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device Used: {self.device}\n")
            f.write(f"Model Path: {self.model_path}\n\n")
            
            f.write("BATCH STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Pairs: {batch_stats['total_pairs']}\n")
            f.write(f"Successfully Processed: {batch_stats['processed_pairs']}\n")
            f.write(f"Failed: {batch_stats['failed_pairs']}\n")
            f.write(f"Success Rate: {batch_stats['processed_pairs']/batch_stats['total_pairs']*100:.1f}%\n")
            f.write(f"Total Processing Time: {batch_stats['total_processing_time']:.2f} seconds\n")
            f.write(f"Average Time per Pair: {batch_stats['total_processing_time']/max(batch_stats['processed_pairs'], 1):.2f} seconds\n\n")
            
            if batch_stats['results']:
                f.write("INDIVIDUAL RESULTS\n")
                f.write("-" * 20 + "\n")
                for result in batch_stats['results']:
                    f.write(f"\nImage ID: {result['image_id']}\n")
                    f.write(f"  Processing Time: {result.get('processing_time', 'N/A'):.2f}s\n")
                    f.write(f"  Raw Change Area: {result['raw_change_percent']:.2f}%\n")
                    f.write(f"  Processed Change Area: {result['processed_change_percent']:.2f}%\n")
                    f.write(f"  Postprocess Effect: {result['postprocess_effect']:.2f}%\n")
                
                # Summary statistics
                change_areas = [r['processed_change_percent'] for r in batch_stats['results']]
                f.write(f"\nSUMMARY STATISTICS\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Change Area: {np.mean(change_areas):.2f}%\n")
                f.write(f"Median Change Area: {np.median(change_areas):.2f}%\n")
                f.write(f"Min Change Area: {np.min(change_areas):.2f}%\n")
                f.write(f"Max Change Area: {np.max(change_areas):.2f}%\n")
                f.write(f"Std Deviation: {np.std(change_areas):.2f}%\n")
        
        print(f"📄 Batch report saved to: {text_report_path}")
        print(f"📋 JSON report saved to: {json_report_path}")
    
    def _load_tif_image(self, image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
        """Load and preprocess TIF image for Cartosat-3"""
        with rasterio.open(image_path) as src:
            # Read all bands
            image_data = src.read()
            
            # Handle NaN values
            if np.isnan(image_data).any():
                for band_idx in range(image_data.shape[0]):
                    band_data = image_data[band_idx]
                    if np.isnan(band_data).any():
                        valid_pixels = band_data[~np.isnan(band_data)]
                        if len(valid_pixels) > 0:
                            replacement_value = np.median(valid_pixels)
                        else:
                            replacement_value = 0.0
                        band_data[np.isnan(band_data)] = replacement_value
            
            # For Cartosat-3, adjust band selection based on available bands
            if image_data.shape[0] >= 3:
                # Use specified RGB bands (default [2,1,0] for proper color)
                if all(0 <= b < image_data.shape[0] for b in rgb_bands):
                    rgb_data = image_data[rgb_bands]
                else:
                    # Fallback: take first 3 bands in reverse order for better color
                    rgb_data = image_data[:3][::-1]
            else:
                # If less than 3 bands, duplicate the available bands
                if image_data.shape[0] == 1:
                    rgb_data = np.repeat(image_data, 3, axis=0)
                else:
                    rgb_data = image_data
            
            # Enhanced normalization for satellite imagery
            if normalize_method == 'adaptive_percentile':
                for i in range(rgb_data.shape[0]):
                    band = rgb_data[i]
                    # Use more robust percentile normalization
                    valid_pixels = band[band > 0]
                    if len(valid_pixels) > 100:
                        p1, p99 = np.percentile(valid_pixels, [1, 99])
                        # Avoid division by zero
                        if p99 > p1:
                            rgb_data[i] = np.clip((band - p1) / (p99 - p1), 0, 1)
                        else:
                            rgb_data[i] = np.clip(band / (band.max() + 1e-8), 0, 1)
                    else:
                        # Fallback normalization
                        band_max = band.max()
                        if band_max > 0:
                            rgb_data[i] = band / band_max
                        else:
                            rgb_data[i] = band
            
            return rgb_data
    
    def _preprocess_for_model(self, image, target_size=256):
        """Preprocess image for model input"""
        # Resize if needed
        if image.shape[1] != target_size or image.shape[2] != target_size:
            image_tensor = torch.from_numpy(image).unsqueeze(0).float()
            image_tensor = F.interpolate(image_tensor, size=(target_size, target_size), mode='bilinear', align_corners=False)
            image = image_tensor.squeeze(0).numpy()
        
        # Convert to tensor and add batch dimension
        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        
        return image_tensor
    
    def _extract_logits_from_output(self, output):
        """Return a single logits tensor [B,C,H,W] from model output"""
        # If it's already a tensor, return it
        if torch.is_tensor(output):
            return output

        # If it's a list/tuple, try to find a good candidate
        if isinstance(output, (list, tuple)):
            # 1) Prefer last tensor that looks like logits with C==1 or C==2
            for o in reversed(output):
                if torch.is_tensor(o) and o.dim() == 4 and o.size(1) in (1, 2):
                    return o
            # 2) Otherwise prefer last 4D tensor found
            for o in reversed(output):
                if torch.is_tensor(o) and o.dim() == 4:
                    return o
            # 3) Otherwise pick last tensor in the list (if any)
            for o in reversed(output):
                if torch.is_tensor(o):
                    return o

        raise ValueError("Could not extract a logits tensor from model output (output type: {}).".format(type(output)))

def main():
    """Main function for batch processing"""
    
    # Configuration
    MODEL_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_myCustom\ChangeFormerV6_Simplified\best_ckpt.pt"
    DATA_DIR_A = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\A"
    DATA_DIR_B = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\cartoCustom\B"
    OUTPUT_DIR = "./batch_inference_results"
    
    # Your validation data image IDs
    VALIDATION_IDS = [
        "0129", "0059", "0055", "0023", "0007", "0008", "0108", "0151", 
        "0022", "0139", "0173", "0026", "0035", "0057", "0062", "0070", 
        "0189", "0006", "0028", "0163"
    ]
    
    # Verify model path exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Verify data directories exist
    if not os.path.exists(DATA_DIR_A):
        print(f"❌ Error: Data directory A not found at {DATA_DIR_A}")
        sys.exit(1)
        
    if not os.path.exists(DATA_DIR_B):
        print(f"❌ Error: Data directory B not found at {DATA_DIR_B}")
        sys.exit(1)
    
    print("🚀 CHANGEFORMER BATCH INFERENCE")
    print("=" * 70)
    print(f"📁 Model: {MODEL_PATH}")
    print(f"📁 Data A: {DATA_DIR_A}")
    print(f"📁 Data B: {DATA_DIR_B}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print(f" Validation Images: {len(VALIDATION_IDS)} pairs")
    print("=" * 70)
    
    # Initialize batch processor
    try:
        processor = BatchProcessor(MODEL_PATH)
        
        # Run batch processing
        batch_results = processor.process_batch(
            VALIDATION_IDS, 
            DATA_DIR_A, 
            DATA_DIR_B, 
            OUTPUT_DIR
        )
        
        print("\n🎉 BATCH PROCESSING COMPLETED SUCCESSFULLY!")
        print(f" Results: {batch_results['processed_pairs']}/{batch_results['total_pairs']} pairs processed")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up matplotlib resources
        plt.close('all')
        print("\n🧹 Cleanup completed. Process ending...")

if __name__ == '__main__':
    main()