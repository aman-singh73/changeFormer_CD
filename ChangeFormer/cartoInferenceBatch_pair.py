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
import glob
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
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

class UnseenDataProcessor:
    """Handles processing of completely unseen data for change detection inference"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.postprocessor = PostProcessor()
        self.results = []
        
        print(f"Initializing processor with device: {self.device}")
        
    def load_model(self):
        """Load the ChangeFormerV6 model once for processing"""
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
            
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    
    def _load_model_smart(self, model_path, device='cpu'):
        """Smart model loading that tries different approaches"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Try to extract model state dict
        model_state = None
        if 'G_state_dict' in checkpoint:
            model_state = checkpoint['G_state_dict']
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
        elif 'model_G_state_dict' in checkpoint:
            model_state = checkpoint['model_G_state_dict']
        else:
            model_state = checkpoint
        
        if model_state is None:
            raise RuntimeError("Could not find model state dict in checkpoint")
        
        return model_state
    
    def discover_image_pairs(self, data_dir_A, data_dir_B, supported_extensions=None):
        if supported_extensions is None:
            supported_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']

        print(f"Discovering image pairs...")
        print(f"Directory A: {data_dir_A}")
        print(f"Directory B: {data_dir_B}")

        # Collect normalized files from A
        files_A = {}
        for ext in supported_extensions:
            for f in glob.glob(os.path.join(data_dir_A, f"*{ext}")) + glob.glob(os.path.join(data_dir_A, f"*{ext.upper()}")):
                fname = os.path.basename(f)
                core = os.path.splitext(fname)[0].replace("t1_", "", 1)
                files_A[core] = fname

        # Collect normalized files from B
        files_B = {}
        for ext in supported_extensions:
            for f in glob.glob(os.path.join(data_dir_B, f"*{ext}")) + glob.glob(os.path.join(data_dir_B, f"*{ext.upper()}")):
                fname = os.path.basename(f)
                core = os.path.splitext(fname)[0].replace("t2_", "", 1)
                files_B[core] = fname

        # Find common core IDs
        common_ids = set(files_A.keys()).intersection(set(files_B.keys()))

        # Build final pairs
        image_pairs = []
        for cid in sorted(common_ids):
            image_pairs.append({
                "id": cid,
                "file_A": files_A[cid],
                "file_B": files_B[cid]
            })

        print(f"Found {len(image_pairs)} image pairs")
        return image_pairs
    
    def _process_single_pair(self, image_id, img1_path, img2_path, results_dirs, original_filename):
        """Process a single image pair"""
        try:
            # Load images
            img1 = self._load_tif_image(img1_path)
            img2 = self._load_tif_image(img2_path)
            
            # Get metadata
            img1_metadata = self._get_image_metadata(img1_path)
            img2_metadata = self._get_image_metadata(img2_path)
            
            # Preprocess for model
            img1_tensor = self._preprocess_for_model(img1).to(self.device)
            img2_tensor = self._preprocess_for_model(img2).to(self.device)
            
            # Model inference
            with torch.no_grad():
                output = self.model(img1_tensor, img2_tensor)
                logits = self._extract_logits_from_output(output)
                
                # Handle multi-class output
                if logits.size(1) > 1:
                    prediction = torch.softmax(logits, dim=1)[:, 1:].sum(dim=1)
                else:
                    prediction = torch.sigmoid(logits.squeeze(1))
                
                # Convert to numpy
                raw_pred = prediction.cpu().numpy().squeeze()
            
            # Postprocess
            processed_pred = self.postprocessor.postprocess_prediction(raw_pred)
            
            # Calculate statistics
            raw_change_pixels = np.sum(raw_pred > 0.5)
            processed_change_pixels = np.sum(processed_pred > 0)
            total_pixels = raw_pred.size
            
            raw_change_percent = (raw_change_pixels / total_pixels) * 100
            processed_change_percent = (processed_change_pixels / total_pixels) * 100
            
            # Save results
            self._save_pair_results(image_id, img1, img2, raw_pred, processed_pred, 
                                  results_dirs, original_filename, img1_metadata, img2_metadata)
            
            # Return result
            return {
                'image_id': image_id,
                'filename': original_filename,
                'raw_change_pixels': int(raw_change_pixels),
                'processed_change_pixels': int(processed_change_pixels),
                'total_pixels': int(total_pixels),
                'raw_change_percent': float(raw_change_percent),
                'processed_change_percent': float(processed_change_percent),
                'postprocess_effect': float(processed_change_percent - raw_change_percent)
            }
            
        except Exception as e:
            print(f"Error processing {image_id}: {e}")
            return None

    def process_unseen_data(self, data_dir_A, data_dir_B, output_dir, 
                           max_pairs=None, supported_extensions=None):

        # Verify directories exist
        if not os.path.exists(data_dir_A):
            raise FileNotFoundError(f"Directory A not found: {data_dir_A}")
        if not os.path.exists(data_dir_B):
            raise FileNotFoundError(f"Directory B not found: {data_dir_B}")

        # Load model once
        self.load_model()

        # Discover image pairs automatically
        image_pairs = self.discover_image_pairs(data_dir_A, data_dir_B, supported_extensions)

        if not image_pairs:
            print("No image pairs found! Check that:")
            print("1. Both directories contain images")
            print("2. Images have matching filenames (t1_ vs t2_)")
            print("3. Images have supported extensions")
            return None

        # Limit number of pairs if specified
        if max_pairs and max_pairs < len(image_pairs):
            image_pairs = image_pairs[:max_pairs]
            print(f"Processing limited to first {max_pairs} pairs")

        # Create output directory structure
        os.makedirs(output_dir, exist_ok=True)
        results_dirs = {
            'visualizations': os.path.join(output_dir, 'visualizations'),
            'raw_predictions': os.path.join(output_dir, 'raw_predictions'),
            'processed_predictions': os.path.join(output_dir, 'processed_predictions'),
            'overlays': os.path.join(output_dir, 'overlays'),
            'reports': os.path.join(output_dir, 'reports'),
            'metadata': os.path.join(output_dir, 'metadata')
        }
        for dir_path in results_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        print(f"Starting processing of {len(image_pairs)} image pairs...")

        # Initialize processing statistics
        processing_stats = {
            'total_pairs': len(image_pairs),
            'processed_pairs': 0,
            'failed_pairs': 0,
            'total_processing_time': 0,
            'results': [],
            'data_info': {
                'data_dir_A': data_dir_A,
                'data_dir_B': data_dir_B,
                'output_dir': output_dir,
                'model_path': self.model_path,
                'processing_date': datetime.now().isoformat()
            }
        }

        start_time = time.time()

        # Process each image pair
        for idx, pair_info in enumerate(image_pairs):
            pair_start_time = time.time()
            try:
                image_id = pair_info['id']
                file_A = pair_info['file_A']
                file_B = pair_info['file_B']

                print(f"Processing {idx + 1}/{len(image_pairs)}: {image_id}")

                # Construct image paths
                img1_path = os.path.join(data_dir_A, file_A)
                img2_path = os.path.join(data_dir_B, file_B)

                # Verify files exist
                if not os.path.exists(img1_path):
                    print(f"Warning: Image A not found: {img1_path}")
                    processing_stats['failed_pairs'] += 1
                    continue
                if not os.path.exists(img2_path):
                    print(f"Warning: Image B not found: {img2_path}")
                    processing_stats['failed_pairs'] += 1
                    continue

                # Process single pair
                result = self._process_single_pair(
                    image_id, img1_path, img2_path, results_dirs, f"{file_A} | {file_B}"
                )

                if result:
                    pair_time = time.time() - pair_start_time
                    result['processing_time'] = pair_time
                    result['file_A'] = file_A
                    result['file_B'] = file_B
                    processing_stats['results'].append(result)
                    processing_stats['processed_pairs'] += 1

                    print(f"Completed {image_id} in {pair_time:.2f}s")
                else:
                    processing_stats['failed_pairs'] += 1

            except Exception as e:
                print(f"Error processing {pair_info.get('id', 'unknown')}: {e}")
                processing_stats['failed_pairs'] += 1
                continue

        processing_stats['total_processing_time'] = time.time() - start_time

        # Generate report
        self._generate_unseen_data_report(processing_stats, results_dirs)

        print("Processing completed!")
        print(f"Successfully processed: {processing_stats['processed_pairs']}/{processing_stats['total_pairs']} pairs")
        print(f"Failed pairs: {processing_stats['failed_pairs']}")
        print(f"Total time: {processing_stats['total_processing_time']:.2f}s")
        if processing_stats['processed_pairs'] > 0:
            avg_time = processing_stats['total_processing_time'] / processing_stats['processed_pairs']
            print(f"Average time per pair: {avg_time:.2f}s")
        print(f"Results saved to: {output_dir}")

        return processing_stats
    
    def _get_image_metadata(self, image_path):
        """Extract metadata from image file"""
        try:
            with rasterio.open(image_path) as src:
                return {
                    'shape': (src.count, src.height, src.width),
                    'dtype': str(src.dtypes[0]),
                    'crs': str(src.crs) if src.crs else None,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'nodata': src.nodata
                }
        except Exception as e:
            # Fallback for non-geotiff images
            try:
                img = Image.open(image_path)
                return {
                    'shape': (len(img.getbands()), img.height, img.width),
                    'dtype': img.mode,
                    'format': img.format
                }
            except:
                return {'error': str(e)}
    
    def _save_pair_results(self, image_id, img1, img2, raw_pred, processed_pred, 
                          results_dirs, original_filename, img1_metadata, img2_metadata):
        """Save all results for a single pair including metadata"""
        
        # 1. Comprehensive visualization
        plt.ioff()
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Convert images for display (C, H, W) -> (H, W, C)
        img1_display = np.transpose(img1, (1, 2, 0))
        img2_display = np.transpose(img2, (1, 2, 0))
        
        # Row 1: Input images and raw prediction
        axes[0, 0].imshow(img1_display)
        axes[0, 0].set_title(f'Image T1 - {image_id}', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_display)
        axes[0, 1].set_title(f'Image T2 - {image_id}', fontsize=12, fontweight='bold')
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
        
        raw_change_pixels = np.sum(raw_pred > 0.5)
        processed_change_pixels = np.sum(processed_pred > 0)
        total_pixels = raw_pred.size
        
        raw_change_percent = (raw_change_pixels / total_pixels) * 100
        processed_change_percent = (processed_change_pixels / total_pixels) * 100
        
        stats_text = f"""CHANGE DETECTION RESULTS
Image ID: {image_id}
File: {original_filename}

Image Metadata:
• T1 Shape: {img1_metadata.get('shape', 'N/A')}
• T2 Shape: {img2_metadata.get('shape', 'N/A')}
• Data Type: {img1_metadata.get('dtype', 'N/A')}

Raw Prediction:
• Change pixels: {raw_change_pixels:,}
• Change area: {raw_change_percent:.2f}%

Processed Prediction:
• Change pixels: {processed_change_pixels:,}
• Change area: {processed_change_percent:.2f}%

Total pixels: {total_pixels:,}"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                       fontsize=9, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        viz_path = os.path.join(results_dirs['visualizations'], f'{image_id}_analysis.png')
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
        
        # 4. Save metadata
        metadata = {
            'image_id': image_id,
            'original_filename': original_filename,
            'processing_timestamp': datetime.now().isoformat(),
            'img1_metadata': img1_metadata,
            'img2_metadata': img2_metadata,
            'results': {
                'raw_change_pixels': int(raw_change_pixels),
                'processed_change_pixels': int(processed_change_pixels),
                'total_pixels': int(total_pixels),
                'raw_change_percent': float(raw_change_percent),
                'processed_change_percent': float(processed_change_percent)
            }
        }
        
        metadata_path = os.path.join(results_dirs['metadata'], f'{image_id}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _generate_unseen_data_report(self, processing_stats, results_dirs):
        """Generate comprehensive report for unseen data processing"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_report_path = os.path.join(results_dirs['reports'], f'report_{timestamp}.json')
        with open(json_report_path, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        
        # Text report
        text_report_path = os.path.join(results_dirs['reports'], f'summary_{timestamp}.txt')
        
        with open(text_report_path, 'w') as f:
            f.write("CHANGEFORMER PROCESSING REPORT\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Input A: {processing_stats['data_info']['data_dir_A']}\n")
            f.write(f"Input B: {processing_stats['data_info']['data_dir_B']}\n")
            f.write(f"Output: {processing_stats['data_info']['output_dir']}\n\n")
            
            f.write("STATISTICS\n")
            f.write("-" * 15 + "\n")
            f.write(f"Total pairs: {processing_stats['total_pairs']}\n")
            f.write(f"Processed: {processing_stats['processed_pairs']}\n")
            f.write(f"Failed: {processing_stats['failed_pairs']}\n")
            f.write(f"Success rate: {processing_stats['processed_pairs']/processing_stats['total_pairs']*100:.1f}%\n")
            f.write(f"Total time: {processing_stats['total_processing_time']:.2f}s\n")
            if processing_stats['processed_pairs'] > 0:
                avg_time = processing_stats['total_processing_time'] / processing_stats['processed_pairs']
                f.write(f"Average time: {avg_time:.2f}s\n")
            f.write("\n")
            
            if processing_stats['results']:
                f.write("RESULTS\n")
                f.write("-" * 10 + "\n")
                
                for i, result in enumerate(processing_stats['results']):
                    f.write(f"{i+1:3d}. {result['image_id']}: {result['processed_change_percent']:.2f}% change\n")
                
                # Summary statistics
                change_areas = [r['processed_change_percent'] for r in processing_stats['results']]
                
                f.write(f"\nSUMMARY\n")
                f.write("-" * 10 + "\n")
                f.write(f"Average change: {np.mean(change_areas):.2f}%\n")
                f.write(f"Median change: {np.median(change_areas):.2f}%\n")
                f.write(f"Min change: {np.min(change_areas):.2f}%\n")
                f.write(f"Max change: {np.max(change_areas):.2f}%\n")
        
        print(f"Report saved to: {text_report_path}")
    
    def _load_tif_image(self, image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
        """Load and preprocess various image formats"""
        
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext in ['.tif', '.tiff']:
            return self._load_with_rasterio(image_path, rgb_bands, normalize_method)
        else:
            return self._load_with_pil(image_path, normalize_method)
    
    def _load_with_rasterio(self, image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
        """Load TIFF/GeoTIFF with rasterio"""
        with rasterio.open(image_path) as src:
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
            
            # Band selection for RGB
            if image_data.shape[0] >= 3:
                if all(0 <= b < image_data.shape[0] for b in rgb_bands):
                    rgb_data = image_data[rgb_bands]
                else:
                    rgb_data = image_data[:3]
            else:
                if image_data.shape[0] == 1:
                    rgb_data = np.repeat(image_data, 3, axis=0)
                else:
                    rgb_data = image_data
            
            return self._normalize_image(rgb_data, normalize_method)
    
    def _load_with_pil(self, image_path, normalize_method='adaptive_percentile'):
        """Load standard image formats with PIL"""
        img = Image.open(image_path)
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32)
        img_array = img_array / 255.0
        
        return self._normalize_image(img_array, normalize_method)
    
    def _normalize_image(self, image_data, normalize_method='adaptive_percentile'):
        """Normalize image data"""
        if normalize_method == 'adaptive_percentile':
            for i in range(image_data.shape[0]):
                band = image_data[i].astype(np.float32)
                valid_pixels = band[band > 0]
                if len(valid_pixels) > 100:
                    p1, p99 = np.percentile(valid_pixels, [1, 99])
                    if p99 > p1:
                        band = np.clip((band - p1) / (p99 - p1), 0, 1)
                    else:
                        band = np.zeros_like(band)
                else:
                    band = (band - band.min()) / (band.max() - band.min() + 1e-6)
                image_data[i] = band

        elif normalize_method == 'minmax':
            for i in range(image_data.shape[0]):
                band = image_data[i].astype(np.float32)
                band = (band - band.min()) / (band.max() - band.min() + 1e-6)
                image_data[i] = band

        elif normalize_method == 'standardize':
            for i in range(image_data.shape[0]):
                band = image_data[i].astype(np.float32)
                mean, std = band.mean(), band.std()
                if std > 0:
                    band = (band - mean) / std
                else:
                    band = band - mean
                # rescale to 0–1 for visualization
                band = (band - band.min()) / (band.max() - band.min() + 1e-6)
                image_data[i] = band

        # Clip final output to [0,1]
        image_data = np.clip(image_data, 0, 1)
        return image_data.astype(np.float32)
    
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
        if torch.is_tensor(output):
            return output

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


def process_single_pair(model_path, img1_path, img2_path, output_dir, device='cuda'):
    """
    Convenience function to process a single image pair
    """
    
    print("Processing single image pair...")
    print(f"Model: {model_path}")
    print(f"Image A: {img1_path}")  
    print(f"Image B: {img2_path}")
    print(f"Output: {output_dir}")
    
    # Verify files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(img1_path):
        raise FileNotFoundError(f"Image A not found: {img1_path}")
    if not os.path.exists(img2_path):
        raise FileNotFoundError(f"Image B not found: {img2_path}")
    
    # Initialize processor
    processor = UnseenDataProcessor(model_path, device)
    processor.load_model()
    
    # Create temporary directories
    temp_dir_A = os.path.join(output_dir, 'temp_A')
    temp_dir_B = os.path.join(output_dir, 'temp_B')
    os.makedirs(temp_dir_A, exist_ok=True)
    os.makedirs(temp_dir_B, exist_ok=True)
    
    try:
        # Copy files to temporary structure
        img_name = os.path.basename(img1_path)
        temp_img1 = os.path.join(temp_dir_A, img_name)
        temp_img2 = os.path.join(temp_dir_B, img_name)
        
        shutil.copy2(img1_path, temp_img1)
        shutil.copy2(img2_path, temp_img2)
        
        # Process using the batch processor
        result = processor.process_unseen_data(temp_dir_A, temp_dir_B, output_dir, max_pairs=1)
        
        print("Single pair processing completed!")
        return result
        
    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_dir_A, ignore_errors=True)
        shutil.rmtree(temp_dir_B, ignore_errors=True)


def main():
    """Main function with flexible configuration options"""
    
    # Configuration
    MODEL_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\modifiedAUG_weight\Fold_2_ChangeFormerV6_Enhanced_CV_Heavy_MultiSpectral_Fold2_Heavy5x\best_ckpt.pt"
    
    # Batch processing directories
    UNSEEN_DATA_DIR_A = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\A"
    UNSEEN_DATA_DIR_B = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\B"
    OUTPUT_DIR = "./batch_unseen_data_results"
    
    # Options
    MAX_PAIRS = None
    SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
    DEVICE = 'cuda'
    
    # Processing mode
    PROCESSING_MODE = "batch"
    
    print("ChangeFormer Inference")
    print("=" * 40)
    
    # Verify model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model checkpoint not found at {MODEL_PATH}")
        sys.exit(1)
    
    try:
        if PROCESSING_MODE == "single":
            print("Mode: Single Pair Processing")
            # You would need to define these paths for single processing
            # result = process_single_pair(MODEL_PATH, SINGLE_IMG1_PATH, SINGLE_IMG2_PATH, SINGLE_OUTPUT_DIR, DEVICE)
            
        else:
            print("Mode: Batch Processing")
            print(f"T1 Images: {UNSEEN_DATA_DIR_A}")
            print(f"T2 Images: {UNSEEN_DATA_DIR_B}")
            print(f"Output: {OUTPUT_DIR}")
            
            # Verify directories exist
            if not os.path.exists(UNSEEN_DATA_DIR_A):
                print(f"Error: T1 directory not found: {UNSEEN_DATA_DIR_A}")
                sys.exit(1)
                
            if not os.path.exists(UNSEEN_DATA_DIR_B):
                print(f"Error: T2 directory not found: {UNSEEN_DATA_DIR_B}")
                sys.exit(1)
            
            # Initialize processor
            processor = UnseenDataProcessor(MODEL_PATH, DEVICE)
            
            # Process unseen data
            result = processor.process_unseen_data(
                UNSEEN_DATA_DIR_A,
                UNSEEN_DATA_DIR_B, 
                OUTPUT_DIR,
                max_pairs=MAX_PAIRS,
                supported_extensions=SUPPORTED_EXTENSIONS
            )
        
        if result and result.get('processed_pairs', 0) > 0:
            print("Processing completed successfully!")
            print(f"Successfully processed: {result['processed_pairs']} pairs")
            avg_change = np.mean([r['processed_change_percent'] for r in result['results']])
            print(f"Average change detected: {avg_change:.2f}%")
        
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close('all')


if __name__ == '__main__':
    main()