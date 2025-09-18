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

class UnseenDataProcessor:
    """Handles processing of completely unseen data for change detection inference"""
    
    def __init__(self, model_path, device='cuda'):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.postprocessor = PostProcessor()
        self.results = []
        
        print(f"Initializing UnseenDataProcessor with device: {self.device}")
        
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
    
    def discover_image_pairs(self, data_dir_A, data_dir_B, supported_extensions=None):
        """
        Automatically discover image pairs from two directories
        Returns list of image IDs that exist in both directories
        """
        if supported_extensions is None:
            supported_extensions = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
        
        print(f"Discovering image pairs in:")
        print(f"  Directory A: {data_dir_A}")
        print(f"  Directory B: {data_dir_B}")
        
        # Get all files from directory A
        files_A = set()
        for ext in supported_extensions:
            pattern_A = os.path.join(data_dir_A, f"*{ext}")
            files_A.update([os.path.basename(f) for f in glob.glob(pattern_A)])
            # Also check uppercase extensions
            pattern_A_upper = os.path.join(data_dir_A, f"*{ext.upper()}")
            files_A.update([os.path.basename(f) for f in glob.glob(pattern_A_upper)])
        
        # Get all files from directory B
        files_B = set()
        for ext in supported_extensions:
            pattern_B = os.path.join(data_dir_B, f"*{ext}")
            files_B.update([os.path.basename(f) for f in glob.glob(pattern_B)])
            # Also check uppercase extensions
            pattern_B_upper = os.path.join(data_dir_B, f"*{ext.upper()}")
            files_B.update([os.path.basename(f) for f in glob.glob(pattern_B_upper)])
        
        # Find common files (image pairs)
        common_files = files_A.intersection(files_B)
        
        # Extract image IDs (filename without extension)
        image_pairs = []
        for filename in common_files:
            # Get the base name without extension
            image_id = os.path.splitext(filename)[0]
            # Store both the ID and the original filename for later use
            image_pairs.append({
                'id': image_id,
                'filename': filename
            })
        
        # Sort by image ID for consistent processing order
        image_pairs = sorted(image_pairs, key=lambda x: x['id'])
        
        print(f"Found {len(image_pairs)} image pairs:")
        for i, pair in enumerate(image_pairs[:10]):  # Show first 10
            print(f"  {i+1:3d}. {pair['id']} ({pair['filename']})")
        if len(image_pairs) > 10:
            print(f"  ... and {len(image_pairs) - 10} more pairs")
        
        return image_pairs
    
    def process_unseen_data(self, data_dir_A, data_dir_B, output_dir, 
                           max_pairs=None, supported_extensions=None):
        """
        Process completely unseen data from two directories
        
        Args:
            data_dir_A: Directory containing T1 (before) images
            data_dir_B: Directory containing T2 (after) images
            output_dir: Directory to save results
            max_pairs: Maximum number of pairs to process (None for all)
            supported_extensions: List of supported file extensions
        """
        
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
            print("❌ No image pairs found! Check that:")
            print("  1. Both directories contain images")
            print("  2. Images have matching filenames in both directories")
            print("  3. Images have supported extensions (.tif, .jpg, .png, etc.)")
            return None
        
        # Limit number of pairs if specified
        if max_pairs and max_pairs < len(image_pairs):
            image_pairs = image_pairs[:max_pairs]
            print(f"🔢 Processing limited to first {max_pairs} pairs")
        
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
        
        print(f"\nStarting processing of {len(image_pairs)} unseen image pairs...")
        print("=" * 70)
        
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
                filename = pair_info['filename']
                
                print(f"\nProcessing pair {idx + 1}/{len(image_pairs)}: {image_id}")
                print("-" * 50)
                
                # Construct image paths
                img1_path = os.path.join(data_dir_A, filename)
                img2_path = os.path.join(data_dir_B, filename)
                
                # Verify files exist (they should, but double-check)
                if not os.path.exists(img1_path):
                    print(f"⚠️  Warning: Image A not found: {img1_path}")
                    processing_stats['failed_pairs'] += 1
                    continue
                    
                if not os.path.exists(img2_path):
                    print(f"⚠️  Warning: Image B not found: {img2_path}")
                    processing_stats['failed_pairs'] += 1
                    continue
                
                # Process single pair
                result = self._process_single_pair(
                    image_id, img1_path, img2_path, results_dirs, filename
                )
                
                if result:
                    pair_time = time.time() - pair_start_time
                    result['processing_time'] = pair_time
                    result['filename'] = filename
                    processing_stats['results'].append(result)
                    processing_stats['processed_pairs'] += 1
                    
                    print(f"✓ Completed {image_id} in {pair_time:.2f}s")
                    
                    # Print progress
                    progress = (idx + 1) / len(image_pairs) * 100
                    print(f" Overall Progress: {progress:.1f}% ({idx + 1}/{len(image_pairs)})")
                else:
                    processing_stats['failed_pairs'] += 1
                    
            except Exception as e:
                print(f"❌ Error processing {pair_info.get('id', 'unknown')}: {e}")
                processing_stats['failed_pairs'] += 1
                continue
        
        processing_stats['total_processing_time'] = time.time() - start_time
        
        # Generate comprehensive report
        self._generate_unseen_data_report(processing_stats, results_dirs)
        
        print("\n" + "=" * 70)
        print("🎯 UNSEEN DATA PROCESSING COMPLETED!")
        print("=" * 70)
        print(f"✅ Successfully processed: {processing_stats['processed_pairs']}/{processing_stats['total_pairs']} pairs")
        print(f"❌ Failed pairs: {processing_stats['failed_pairs']}")
        print(f"⏱️  Total time: {processing_stats['total_processing_time']:.2f}s")
        if processing_stats['processed_pairs'] > 0:
            avg_time = processing_stats['total_processing_time'] / processing_stats['processed_pairs']
            print(f" Average time per pair: {avg_time:.2f}s")
        print(f"📁 Results saved to: {output_dir}")
        
        return processing_stats
    
    def _process_single_pair(self, image_id, img1_path, img2_path, results_dirs, original_filename):
        """Process a single image pair"""
        
        try:
            # Get image metadata
            img1_metadata = self._get_image_metadata(img1_path)
            img2_metadata = self._get_image_metadata(img2_path)
            
            print(f"  Image A: {img1_metadata['shape']} | Type: {img1_metadata.get('dtype', 'unknown')}")
            print(f"  Image B: {img2_metadata['shape']} | Type: {img2_metadata.get('dtype', 'unknown')}")
            
            # Load and preprocess images
            img1 = self._load_tif_image(img1_path, rgb_bands=[2, 1, 0])
            img2 = self._load_tif_image(img2_path, rgb_bands=[2, 1, 0])
            
            print(f"  Processed A: {img1.shape} | Range: {img1.min():.3f}-{img1.max():.3f}")
            print(f"  Processed B: {img2.shape} | Range: {img2.min():.3f}-{img2.max():.3f}")
            
            # Preprocess for model
            img1_tensor = self._preprocess_for_model(img1).to(self.device)
            img2_tensor = self._preprocess_for_model(img2).to(self.device)
            
            # Run inference
            print("  Running inference...")
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
            
            print(f"  Prediction: {prediction_np.shape} | Range: {prediction_np.min():.4f}-{prediction_np.max():.4f}")
            
            # Apply postprocessing
            raw_binary = (prediction_np > 0.5).astype(np.uint8)
            processed_prediction = self.postprocessor.postprocess_prediction(prediction_np)
            
            # Save results
            self._save_pair_results(
                image_id, img1, img2, raw_binary, processed_prediction, 
                results_dirs, original_filename, img1_metadata, img2_metadata
            )
            
            # Calculate statistics
            raw_change_pixels = np.sum(raw_binary > 0)
            processed_change_pixels = np.sum(processed_prediction > 0)
            total_pixels = raw_binary.size
            
            result = {
                'image_id': image_id,
                'original_filename': original_filename,
                'raw_change_pixels': int(raw_change_pixels),
                'processed_change_pixels': int(processed_change_pixels),
                'total_pixels': int(total_pixels),
                'raw_change_percent': float(raw_change_pixels / total_pixels * 100),
                'processed_change_percent': float(processed_change_pixels / total_pixels * 100),
                'postprocess_effect': float((processed_change_pixels - raw_change_pixels) / total_pixels * 100),
                'img1_metadata': img1_metadata,
                'img2_metadata': img2_metadata
            }
            
            return result
            
        except Exception as e:
            print(f"Error in _process_single_pair for {image_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
        axes[0, 0].set_title(f'Image T1 - {image_id}\n({original_filename})', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img2_display)
        axes[0, 1].set_title(f'Image T2 - {image_id}\n({original_filename})', fontsize=12, fontweight='bold')
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
        
        stats_text = f"""CHANGE DETECTION RESULTS
Image ID: {image_id}
File: {original_filename}
Data: UNSEEN/TEST DATA

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

Postprocessing Effect:
• Pixel difference: {processed_change_pixels - raw_change_pixels:,}
• Area difference: {processed_change_percent - raw_change_percent:.2f}%

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
        json_report_path = os.path.join(results_dirs['reports'], f'unseen_data_report_{timestamp}.json')
        with open(json_report_path, 'w') as f:
            json.dump(processing_stats, f, indent=2, default=str)
        
        # Detailed text report
        text_report_path = os.path.join(results_dirs['reports'], f'unseen_data_summary_{timestamp}.txt')
        
        with open(text_report_path, 'w') as f:
            f.write("CHANGEFORMER UNSEEN DATA PROCESSING REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device Used: {self.device}\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Input Directory A: {processing_stats['data_info']['data_dir_A']}\n")
            f.write(f"Input Directory B: {processing_stats['data_info']['data_dir_B']}\n")
            f.write(f"Output Directory: {processing_stats['data_info']['output_dir']}\n\n")
            
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 25 + "\n")
            f.write(f"Total Image Pairs Found: {processing_stats['total_pairs']}\n")
            f.write(f"Successfully Processed: {processing_stats['processed_pairs']}\n")
            f.write(f"Failed: {processing_stats['failed_pairs']}\n")
            f.write(f"Success Rate: {processing_stats['processed_pairs']/processing_stats['total_pairs']*100:.1f}%\n")
            f.write(f"Total Processing Time: {processing_stats['total_processing_time']:.2f} seconds\n")
            if processing_stats['processed_pairs'] > 0:
                avg_time = processing_stats['total_processing_time'] / processing_stats['processed_pairs']
                f.write(f"Average Time per Pair: {avg_time:.2f} seconds\n")
            f.write("\n")
            
            if processing_stats['results']:
                f.write("INDIVIDUAL RESULTS\n")
                f.write("-" * 20 + "\n")
                
                # Sort results by change area for better analysis
                sorted_results = sorted(processing_stats['results'], 
                                      key=lambda x: x['processed_change_percent'], reverse=True)
                
                for i, result in enumerate(sorted_results):
                    f.write(f"\n{i+1:3d}. Image ID: {result['image_id']}\n")
                    f.write(f"     File: {result.get('filename', 'N/A')}\n")
                    f.write(f"     Processing Time: {result.get('processing_time', 0):.2f}s\n")
                    f.write(f"     Raw Change Area: {result['raw_change_percent']:.2f}%\n")
                    f.write(f"     Processed Change Area: {result['processed_change_percent']:.2f}%\n")
                    f.write(f"     Postprocess Effect: {result['postprocess_effect']:.2f}%\n")
                
                # Summary statistics
                change_areas = [r['processed_change_percent'] for r in processing_stats['results']]
                postprocess_effects = [r['postprocess_effect'] for r in processing_stats['results']]
                
                f.write(f"\nCHANGE DETECTION SUMMARY STATISTICS\n")
                f.write("-" * 35 + "\n")
                f.write(f"Change Area Statistics:\n")
                f.write(f"  Average: {np.mean(change_areas):.2f}%\n")
                f.write(f"  Median: {np.median(change_areas):.2f}%\n")
                f.write(f"  Minimum: {np.min(change_areas):.2f}%\n")
                f.write(f"  Maximum: {np.max(change_areas):.2f}%\n")
                f.write(f"  Std Deviation: {np.std(change_areas):.2f}%\n\n")
                
                f.write(f"Postprocessing Effect Statistics:\n")
                f.write(f"  Average Effect: {np.mean(postprocess_effects):.2f}%\n")
                f.write(f"  Median Effect: {np.median(postprocess_effects):.2f}%\n")
                f.write(f"  Min Effect: {np.min(postprocess_effects):.2f}%\n")
                f.write(f"  Max Effect: {np.max(postprocess_effects):.2f}%\n")
                
                # Categorize results
                high_change = [r for r in processing_stats['results'] if r['processed_change_percent'] > 5.0]
                medium_change = [r for r in processing_stats['results'] if 1.0 <= r['processed_change_percent'] <= 5.0]
                low_change = [r for r in processing_stats['results'] if r['processed_change_percent'] < 1.0]
                
                f.write(f"\nCHANGE CATEGORIZATION\n")
                f.write("-" * 20 + "\n")
                f.write(f"High Change (>5%): {len(high_change)} images\n")
                f.write(f"Medium Change (1-5%): {len(medium_change)} images\n")
                f.write(f"Low Change (<1%): {len(low_change)} images\n")
                
                if high_change:
                    f.write(f"\nHigh Change Images:\n")
                    for r in high_change:
                        f.write(f"  • {r['image_id']}: {r['processed_change_percent']:.2f}%\n")
        
        print(f"📄 Detailed report saved to: {text_report_path}")
        print(f"📋 JSON report saved to: {json_report_path}")
    
    def _load_tif_image(self, image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
        """Load and preprocess various image formats"""
        
        # Handle different file formats
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if file_ext in ['.tif', '.tiff']:
            # Use rasterio for TIFF files (including GeoTIFF)
            return self._load_with_rasterio(image_path, rgb_bands, normalize_method)
        else:
            # Use PIL for other formats (jpg, png, etc.)
            return self._load_with_pil(image_path, normalize_method)
    
    def _load_with_rasterio(self, image_path, rgb_bands=[2, 1, 0], normalize_method='adaptive_percentile'):
        """Load TIFF/GeoTIFF with rasterio"""
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
            
            # Band selection for RGB
            if image_data.shape[0] >= 3:
                # Use specified RGB bands
                if all(0 <= b < image_data.shape[0] for b in rgb_bands):
                    rgb_data = image_data[rgb_bands]
                else:
                    # Fallback: take first 3 bands
                    rgb_data = image_data[:3]
            else:
                # If less than 3 bands, duplicate the available bands
                if image_data.shape[0] == 1:
                    rgb_data = np.repeat(image_data, 3, axis=0)
                else:
                    rgb_data = image_data
            
            # Normalize
            return self._normalize_image(rgb_data, normalize_method)
    
    def _load_with_pil(self, image_path, normalize_method='adaptive_percentile'):
        """Load standard image formats with PIL"""
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array (H, W, C) -> (C, H, W)
        img_array = np.array(img).transpose(2, 0, 1).astype(np.float32)
        
        # Normalize from 0-255 to 0-1 range first
        img_array = img_array / 255.0
        
        return self._normalize_image(img_array, normalize_method)
    
    def _normalize_image(self, image_data, normalize_method='adaptive_percentile'):
        """Normalize image data"""
        if normalize_method == 'adaptive_percentile':
            for i in range(image_data.shape[0]):
                band = image_data[i]
                # Use robust percentile normalization
                valid_pixels = band[band > 0]
                if len(valid_pixels) > 100:
                    p1, p99 = np.percentile(valid_pixels, [1, 99])
                    # Avoid division by zero
                    if p99 > p1:
                        image_data[i] = np.clip((band - p1) / (p99 - p1), 0, 1)
                    else:
                        image_data[i] = np.clip(band / (band.max() + 1e-8), 0, 1)
                else:
                    # Fallback normalization
                    band_max = band.max()
                    if band_max > 0:
                        image_data[i] = band / band_max
                    else:
                        image_data[i] = band
        
        return image_data
    
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
    Useful for quick testing or single-image processing
    """
    
    print("🔍 SINGLE PAIR PROCESSING")
    print("=" * 50)
    print(f"Model: {model_path}")
    print(f"Image A: {img1_path}")  
    print(f"Image B: {img2_path}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    
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
        import shutil
        img_name = os.path.basename(img1_path)
        temp_img1 = os.path.join(temp_dir_A, img_name)
        temp_img2 = os.path.join(temp_dir_B, img_name)
        
        shutil.copy2(img1_path, temp_img1)
        shutil.copy2(img2_path, temp_img2)
        
        # Process using the batch processor
        result = processor.process_unseen_data(temp_dir_A, temp_dir_B, output_dir, max_pairs=1)
        
        print("✅ Single pair processing completed!")
        return result
        
    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_dir_A, ignore_errors=True)
        shutil.rmtree(temp_dir_B, ignore_errors=True)

def main():
    """Main function with flexible configuration options"""
    
    # ===== CONFIGURATION SECTION =====
    # Modify these paths according to your setup
    
    MODEL_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_CV_Heavy_MultiSpectral\Fold_2_ChangeFormerV6_Enhanced_CV_Heavy_MultiSpectral_248_Fold2_Heavy5x\best_ckpt.pt"
    
    # # ===== OPTION 1: Process directories of unseen data =====
    # UNSEEN_DATA_DIR_A = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\A"  # Directory with T1 (before) images
    # UNSEEN_DATA_DIR_B = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\B"  # Directory with T2 (after) images
    # OUTPUT_DIR = "./unseen_data_results"
    
    # ===== OPTION 2: Process a single pair =====
    SINGLE_IMG1_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\patch_00255_T1.tif"
    SINGLE_IMG2_PATH = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\TEST_DATA\patch_00255_T2.tif"
    SINGLE_OUTPUT_DIR = "./single_pair_results_1_2"
    
    # ===== PROCESSING OPTIONS =====
    MAX_PAIRS = None  # Set to a number to limit processing (None = process all)
    SUPPORTED_EXTENSIONS = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']
    DEVICE = 'cuda'  # or 'cpu'
    
    # ===== CHOOSE PROCESSING MODE =====
    PROCESSING_MODE = "single"  # "batch" or "single" - Changed to single for your case
    
    print("🚀 CHANGEFORMER UNSEEN DATA INFERENCE")
    print("=" * 70)
    
    # Verify model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model checkpoint not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model checkpoint.")
        sys.exit(1)
    
    try:
        if PROCESSING_MODE == "single":
            # Single pair processing
            print("Mode: Single Pair Processing")
            result = process_single_pair(
                MODEL_PATH, 
                SINGLE_IMG1_PATH, 
                SINGLE_IMG2_PATH, 
                SINGLE_OUTPUT_DIR,
                DEVICE
            )
            
        else:
            # Batch processing
            print("Mode: Batch Processing of Directory")
            print(f"📁 T1 Images: {UNSEEN_DATA_DIR_A}")
            print(f"📁 T2 Images: {UNSEEN_DATA_DIR_B}")
            print(f"📁 Output: {OUTPUT_DIR}")
            print(f"🔢 Max pairs: {MAX_PAIRS or 'All'}")
            print("=" * 70)
            
            # Verify directories exist
            if not os.path.exists(UNSEEN_DATA_DIR_A):
                print(f"❌ Error: T1 directory not found: {UNSEEN_DATA_DIR_A}")
                print("Please update UNSEEN_DATA_DIR_A to point to your T1 images directory.")
                sys.exit(1)
                
            if not os.path.exists(UNSEEN_DATA_DIR_B):
                print(f"❌ Error: T2 directory not found: {UNSEEN_DATA_DIR_B}")
                print("Please update UNSEEN_DATA_DIR_B to point to your T2 images directory.")
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
        
        if result:
            print("\n🎉 PROCESSING COMPLETED SUCCESSFULLY!")
            if 'processed_pairs' in result:
                print(f" Successfully processed: {result['processed_pairs']} pairs")
                if result['processed_pairs'] > 0:
                    avg_change = np.mean([r['processed_change_percent'] for r in result['results']])
                    print(f"📈 Average change detected: {avg_change:.2f}%")
        
    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting tips:")
        print("1. Ensure your model checkpoint path is correct")
        print("2. Verify that input directories contain matching image pairs")
        print("3. Check that image formats are supported")
        print("4. Make sure you have sufficient disk space for results")
    finally:
        plt.close('all')
        print("\n🧹 Cleanup completed. Process ending...")

if __name__ == '__main__':
    main()