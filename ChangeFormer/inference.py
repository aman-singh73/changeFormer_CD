# # inference_custom.py - Save this as a separate file
# import torch
# import torch.nn as nn
# import numpy as np
# import rasterio
# from PIL import Image
# from torchvision import transforms
# import cv2
# import matplotlib.pyplot as plt
# import glob
# import os
# from models.ChangeFormer import ChangeFormerV6
# import warnings
# warnings.filterwarnings('ignore')

# class CustomInferenceProcessor:
#     def __init__(self, model_path, device=None):
#         """
#         Initialize the inference processor
        
#         Args:
#             model_path: Path to your fine-tuned model (.pt file)
#             device: torch device ('cuda' or 'cpu'). Auto-detects if None.
#         """
#         self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print(f"Using device: {self.device}")
        
#         # Load the fine-tuned model
#         print(f"Loading model from: {model_path}")
#         self.model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(self.device)
        
#         checkpoint = torch.load(model_path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_G_state_dict'])
#         self.model.eval()
        
#         print("Model loaded successfully!")
        
#         # Same transforms as training (CRITICAL for consistency)
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
    
#     def load_and_preprocess_image(self, img_path, bands=[4, 3, 2], target_size=256):
#         """
#         Load and preprocess satellite image - EXACTLY same as training
        
#         Args:
#             img_path: Path to the .tif image
#             bands: Which bands to use (default: [4,3,2] for RGB from Sentinel-2)
#             target_size: Target size for inference
        
#         Returns:
#             Preprocessed image tensor
#         """
#         with rasterio.open(img_path) as src:
#             img_data = []
#             for band_idx in bands:
#                 if band_idx <= src.count:
#                     band = src.read(band_idx).astype(np.float32)
#                     band = np.nan_to_num(band, nan=0.0)
#                     img_data.append(band)
#             img = np.stack(img_data, axis=-1)
        
#         # SAME robust preprocessing as training
#         p1, p99 = np.percentile(img, [1, 99])
#         img_clipped = np.clip(img, p1, p99)
        
#         # Per-channel normalization
#         img_normalized = np.zeros_like(img_clipped)
#         for c in range(img.shape[2]):
#             channel = img_clipped[:, :, c]
#             channel_min, channel_max = channel.min(), channel.max()
#             if channel_max > channel_min:
#                 img_normalized[:, :, c] = (channel - channel_min) / (channel_max - channel_min)
#             else:
#                 img_normalized[:, :, c] = 0.5
        
#         # Convert to uint8 for PIL
#         img_uint8 = (img_normalized * 255).astype(np.uint8)
        
#         # Resize and transform
#         img_pil = Image.fromarray(img_uint8).resize((target_size, target_size), Image.LANCZOS)
#         img_tensor = self.transform(img_pil).unsqueeze(0)  # Add batch dimension
        
#         return img_tensor.to(self.device)
    
#     def predict_single_pair(self, img_A_path, img_B_path, output_size=None):
#         """
#         Predict change map for a single pair of images
        
#         Args:
#             img_A_path: Path to pre-change image
#             img_B_path: Path to post-change image
#             output_size: Tuple (H, W) for output size. If None, uses original image size.
        
#         Returns:
#             change_map: Binary change map (numpy array)
#             confidence_map: Confidence scores (numpy array)
#         """
#         with torch.no_grad():
#             # Load and preprocess images
#             img_A = self.load_and_preprocess_image(img_A_path)
#             img_B = self.load_and_preprocess_image(img_B_path)
            
#             # Get original size if needed
#             if output_size is None:
#                 with rasterio.open(img_A_path) as src:
#                     output_size = (src.height, src.width)
            
#             # Forward pass
#             outputs = self.model(img_A, img_B)
#             if isinstance(outputs, list):
#                 outputs = outputs[-1]  # Take final output if multi-scale
            
#             # Apply softmax to get probabilities
#             probs = torch.softmax(outputs, dim=1)
#             confidence_scores = probs[0, 1].cpu().numpy()  # Change class probability
            
#             # Get binary predictions
#             predictions = torch.argmax(outputs, dim=1)
#             change_map = predictions[0].cpu().numpy().astype(np.uint8)
            
#             # Resize to original size if needed
#             if change_map.shape != output_size:
#                 change_map = cv2.resize(change_map, (output_size[1], output_size[0]), 
#                                      interpolation=cv2.INTER_NEAREST)
#                 confidence_scores = cv2.resize(confidence_scores, (output_size[1], output_size[0]), 
#                                              interpolation=cv2.INTER_LINEAR)
        
#         return change_map, confidence_scores
    
#     def batch_inference(self, input_dir, output_dir, confidence_threshold=0.5):
#         """
#         Run inference on a batch of image pairs
        
#         Args:
#             input_dir: Directory containing A/, B/ subfolders with image pairs
#             output_dir: Directory to save results
#             confidence_threshold: Threshold for confident predictions
#         """
#         os.makedirs(output_dir, exist_ok=True)
#         os.makedirs(f"{output_dir}/change_maps", exist_ok=True)
#         os.makedirs(f"{output_dir}/confidence_maps", exist_ok=True)
#         os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
#         # Find image pairs
#         img_A_paths = sorted(glob.glob(f"{input_dir}/A/*.tif"))
#         img_B_paths = sorted(glob.glob(f"{input_dir}/B/*.tif"))
        
#         if len(img_A_paths) != len(img_B_paths):
#             print(f"Warning: Mismatch in number of images! A: {len(img_A_paths)}, B: {len(img_B_paths)}")
#             min_len = min(len(img_A_paths), len(img_B_paths))
#             img_A_paths = img_A_paths[:min_len]
#             img_B_paths = img_B_paths[:min_len]
        
#         print(f"Processing {len(img_A_paths)} image pairs...")
        
#         results_summary = []
        
#         for i, (img_A_path, img_B_path) in enumerate(zip(img_A_paths, img_B_paths)):
#             print(f"Processing pair {i+1}/{len(img_A_paths)}: {os.path.basename(img_A_path)}")
            
#             try:
#                 # Run inference
#                 change_map, confidence_scores = self.predict_single_pair(img_A_path, img_B_path)
                
#                 # Apply confidence threshold
#                 confident_changes = (confidence_scores > confidence_threshold) & (change_map > 0)
                
#                 # Calculate statistics
#                 total_pixels = change_map.size
#                 changed_pixels = np.sum(change_map > 0)
#                 confident_changed_pixels = np.sum(confident_changes)
#                 change_percentage = (changed_pixels / total_pixels) * 100
#                 confident_change_percentage = (confident_changed_pixels / total_pixels) * 100
#                 avg_confidence = np.mean(confidence_scores[change_map > 0]) if changed_pixels > 0 else 0
                
#                 # Save results
#                 base_name = os.path.splitext(os.path.basename(img_A_path))[0]
                
#                 # Save binary change map
#                 change_map_path = f"{output_dir}/change_maps/{base_name}_change.tif"
#                 self.save_geotiff(change_map, img_A_path, change_map_path)
                
#                 # Save confidence map
#                 conf_map_path = f"{output_dir}/confidence_maps/{base_name}_confidence.tif"
#                 self.save_geotiff(confidence_scores, img_A_path, conf_map_path)
                
#                 # Create visualization
#                 vis_path = f"{output_dir}/visualizations/{base_name}_visualization.png"
#                 self.create_visualization(img_A_path, img_B_path, change_map, 
#                                         confidence_scores, vis_path)
                
#                 # Store results
#                 results_summary.append({
#                     'image_pair': base_name,
#                     'total_pixels': total_pixels,
#                     'changed_pixels': changed_pixels,
#                     'confident_changed_pixels': confident_changed_pixels,
#                     'change_percentage': change_percentage,
#                     'confident_change_percentage': confident_change_percentage,
#                     'avg_confidence': avg_confidence
#                 })
                
#                 print(f"  - Changed pixels: {changed_pixels} ({change_percentage:.2f}%)")
#                 print(f"  - Confident changes: {confident_changed_pixels} ({confident_change_percentage:.2f}%)")
#                 print(f"  - Average confidence: {avg_confidence:.3f}")
                
#             except Exception as e:
#                 print(f"Error processing {base_name}: {str(e)}")
#                 continue
        
#         # Save summary report
#         self.save_summary_report(results_summary, f"{output_dir}/inference_summary.txt")
#         print(f"\nInference completed! Results saved in: {output_dir}")
    
#     def save_geotiff(self, data, reference_path, output_path):
#         """Save numpy array as GeoTIFF with same georeference as input"""
#         with rasterio.open(reference_path) as src:
#             profile = src.profile.copy()
#             profile.update({
#                 'count': 1,
#                 'dtype': data.dtype,
#                 'height': data.shape[0],
#                 'width': data.shape[1]
#             })
            
#             with rasterio.open(output_path, 'w', **profile) as dst:
#                 dst.write(data, 1)
    
#     def create_visualization(self, img_A_path, img_B_path, change_map, 
#                            confidence_scores, output_path):
#         """Create a comprehensive visualization"""
#         # Load original images for visualization
#         img_A_vis = self.load_image_for_display(img_A_path)
#         img_B_vis = self.load_image_for_display(img_B_path)
        
#         fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
#         # Pre-change image
#         axes[0, 0].imshow(img_A_vis)
#         axes[0, 0].set_title('Pre-change Image')
#         axes[0, 0].axis('off')
        
#         # Post-change image
#         axes[0, 1].imshow(img_B_vis)
#         axes[0, 1].set_title('Post-change Image')
#         axes[0, 1].axis('off')
        
#         # Change map
#         axes[0, 2].imshow(change_map, cmap='Reds', alpha=0.8)
#         axes[0, 2].set_title('Detected Changes')
#         axes[0, 2].axis('off')
        
#         # Confidence map
#         im1 = axes[1, 0].imshow(confidence_scores, cmap='viridis', vmin=0, vmax=1)
#         axes[1, 0].set_title('Confidence Scores')
#         axes[1, 0].axis('off')
#         plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
#         # Overlay on post-change image
#         axes[1, 1].imshow(img_B_vis)
#         axes[1, 1].imshow(change_map, cmap='Reds', alpha=0.5)
#         axes[1, 1].set_title('Changes Overlay')
#         axes[1, 1].axis('off')
        
#         # Confidence histogram
#         axes[1, 2].hist(confidence_scores.flatten(), bins=50, alpha=0.7, color='blue')
#         axes[1, 2].set_title('Confidence Distribution')
#         axes[1, 2].set_xlabel('Confidence Score')
#         axes[1, 2].set_ylabel('Frequency')
        
#         plt.tight_layout()
#         plt.savefig(output_path, dpi=200, bbox_inches='tight')
#         plt.close()
    
#     def load_image_for_display(self, img_path, bands=[4, 3, 2]):
#         """Load image for display purposes (without normalization)"""
#         with rasterio.open(img_path) as src:
#             img_data = []
#             for band_idx in bands:
#                 if band_idx <= src.count:
#                     band = src.read(band_idx).astype(np.float32)
#                     img_data.append(band)
#             img = np.stack(img_data, axis=-1)
        
#         # Simple percentile stretching for visualization
#         p2, p98 = np.percentile(img, [2, 98])
#         img_stretched = np.clip((img - p2) / (p98 - p2), 0, 1)
        
#         return img_stretched
    
#     def save_summary_report(self, results, output_path):
#         """Save a summary report of all results"""
#         with open(output_path, 'w') as f:
#             f.write("="*60 + "\n")
#             f.write("CHANGE DETECTION INFERENCE SUMMARY\n")
#             f.write("="*60 + "\n\n")
            
#             if results:
#                 total_images = len(results)
#                 avg_change_pct = np.mean([r['change_percentage'] for r in results])
#                 avg_confident_pct = np.mean([r['confident_change_percentage'] for r in results])
#                 avg_confidence = np.mean([r['avg_confidence'] for r in results if r['avg_confidence'] > 0])
                
#                 f.write(f"Total image pairs processed: {total_images}\n")
#                 f.write(f"Average change percentage: {avg_change_pct:.2f}%\n")
#                 f.write(f"Average confident change percentage: {avg_confident_pct:.2f}%\n")
#                 f.write(f"Average confidence score: {avg_confidence:.3f}\n\n")
                
#                 f.write("DETAILED RESULTS:\n")
#                 f.write("-" * 40 + "\n")
                
#                 for result in results:
#                     f.write(f"\nImage: {result['image_pair']}\n")
#                     f.write(f"  Total pixels: {result['total_pixels']:,}\n")
#                     f.write(f"  Changed pixels: {result['changed_pixels']:,} ({result['change_percentage']:.2f}%)\n")
#                     f.write(f"  Confident changes: {result['confident_changed_pixels']:,} ({result['confident_change_percentage']:.2f}%)\n")
#                     f.write(f"  Average confidence: {result['avg_confidence']:.3f}\n")
#             else:
#                 f.write("No results to summarize.\n")

# # ============================================================================
# # USAGE EXAMPLES
# # ============================================================================

# def single_pair_example():
#     """Example: Inference on a single pair of images"""
#     model_path = "./fine_tuned_models/best_custom_model.pt"
    
#     # Initialize processor
#     processor = CustomInferenceProcessor(model_path)
    
#     # Paths to your test images
#     img_A_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\A\0000.tif"
#     img_B_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\B\0000.tif"
    
#     # Run inference
#     change_map, confidence_scores = processor.predict_single_pair(img_A_path, img_B_path)
    
#     # Analyze results
#     changed_pixels = np.sum(change_map > 0)
#     total_pixels = change_map.size
#     change_percentage = (changed_pixels / total_pixels) * 100
    
#     print(f"Change detection results:")
#     print(f"- Total pixels: {total_pixels:,}")
#     print(f"- Changed pixels: {changed_pixels:,}")
#     print(f"- Change percentage: {change_percentage:.2f}%")
#     print(f"- Max confidence: {np.max(confidence_scores):.3f}")
#     print(f"- Min confidence: {np.min(confidence_scores):.3f}")
    
#     # Save results
#     cv2.imwrite("change_map.png", change_map * 255)
#     cv2.imwrite("confidence_map.png", (confidence_scores * 255).astype(np.uint8))
    
#     return change_map, confidence_scores

# def batch_inference_example():
#     """Example: Batch inference on multiple image pairs"""
#     model_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\fine_tuned_models\best_custom_model.pt"
    
#     # Initialize processor
#     processor = CustomInferenceProcessor(model_path)
    
#     # Directory structure should be:
#     # input_dir/
#     #   ├── A/  (pre-change images)
#     #   └── B/  (post-change images)
#     input_dir = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom"
#     output_dir = "./inference_results123"
    
#     # Run batch inference
#     processor.batch_inference(input_dir, output_dir, confidence_threshold=0.3)

# if __name__ == "__main__":
#     # Choose which example to run
    
#     # For single pair inference:
#     # single_pair_example()
    
#     # For batch inference:
#     batch_inference_example()
    
#     # Or customize your own:
#     model_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\fine_tuned_models\best_custom_model.pt"
#     input_dir = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom" 
#     output_dir = "./my_inference_results123"
    
#     processor = CustomInferenceProcessor(model_path)
#     processor.batch_inference(input_dir, output_dir, confidence_threshold=0.6)


import torch
import torch.nn as nn
import numpy as np
import rasterio
from PIL import Image
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import os
from models.ChangeFormer import ChangeFormerV6
import warnings
warnings.filterwarnings('ignore')

class SinglePairInferenceProcessor:
    def __init__(self, model_path, device=None):
        """
        Initialize the inference processor for single pair processing
        
        Args:
            model_path: Path to your fine-tuned model (.pt file)
            device: torch device ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load the fine-tuned model
        print(f"Loading model from: {model_path}")
        self.model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_G_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Same transforms as training
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_and_preprocess_image(self, img_path, bands=[4, 3, 2], target_size=256):
        """
        Load and preprocess satellite image - EXACTLY same as training
        
        Args:
            img_path: Path to the .tif image
            bands: Which bands to use (default: [4,3,2] for RGB from Sentinel-2)
            target_size: Target size for inference
        
        Returns:
            Preprocessed image tensor
        """
        with rasterio.open(img_path) as src:
            img_data = []
            for band_idx in bands:
                if band_idx <= src.count:
                    band = src.read(band_idx).astype(np.float32)
                    band = np.nan_to_num(band, nan=0.0)
                    img_data.append(band)
            img = np.stack(img_data, axis=-1)
        
        # Robust preprocessing as training
        p1, p99 = np.percentile(img, [1, 99])
        img_clipped = np.clip(img, p1, p99)
        
        # Per-channel normalization
        img_normalized = np.zeros_like(img_clipped)
        for c in range(img.shape[2]):
            channel = img_clipped[:, :, c]
            channel_min, channel_max = channel.min(), channel.max()  # Fixed typo here
            if channel_max > channel_min:
                img_normalized[:, :, c] = (channel - channel_min) / (channel_max - channel_min)
            else:
                img_normalized[:, :, c] = 0.5
        
        # Convert to uint8 for PIL
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        
        # Resize and transform
        img_pil = Image.fromarray(img_uint8).resize((target_size, target_size), Image.LANCZOS)
        img_tensor = self.transform(img_pil).unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def predict_single_pair(self, img_A_path, img_B_path, output_dir, confidence_threshold=0.5):
        """
        Predict change map for a single pair of images
        
        Args:
            img_A_path: Path to pre-change image
            img_B_path: Path to post-change image
            output_dir: Directory to save results
            confidence_threshold: Threshold for confident predictions
        
        Returns:
            change_map: Binary change map (numpy array)
            confidence_scores: Confidence scores (numpy array)
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/change_maps", exist_ok=True)
        os.makedirs(f"{output_dir}/confidence_maps", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        with torch.no_grad():
            # Load and preprocess images
            img_A = self.load_and_preprocess_image(img_A_path)
            img_B = self.load_and_preprocess_image(img_B_path)
            
            # Get original size
            with rasterio.open(img_A_path) as src:
                output_size = (src.height, src.width)
            
            # Forward pass
            outputs = self.model(img_A, img_B)
            if isinstance(outputs, list):
                outputs = outputs[-1]  # Take final output if multi-scale
            
            # Apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
            confidence_scores = probs[0, 1].cpu().numpy()  # Change class probability
            
            # Get binary predictions
            predictions = torch.argmax(outputs, dim=1)
            change_map = predictions[0].cpu().numpy().astype(np.uint8)
            
            # Resize to original size if needed
            if change_map.shape != output_size:
                change_map = cv2.resize(change_map, (output_size[1], output_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
                confidence_scores = cv2.resize(confidence_scores, (output_size[1], output_size[0]), 
                                             interpolation=cv2.INTER_LINEAR)
        
        # Apply confidence threshold
        confident_changes = (confidence_scores > confidence_threshold) & (change_map > 0)
        
        # Calculate statistics
        total_pixels = change_map.size
        changed_pixels = np.sum(change_map > 0)
        confident_changed_pixels = np.sum(confident_changes)
        change_percentage = (changed_pixels / total_pixels) * 100
        confident_change_percentage = (confident_changed_pixels / total_pixels) * 100
        avg_confidence = np.mean(confidence_scores[change_map > 0]) if changed_pixels > 0 else 0
        
        # Save results
        base_name = os.path.splitext(os.path.basename(img_A_path))[0]
        
        # Save binary change map
        change_map_path = f"{output_dir}/change_maps/{base_name}_change.tif"
        self.save_geotiff(change_map, img_A_path, change_map_path)
        
        # Save confidence map
        conf_map_path = f"{output_dir}/confidence_maps/{base_name}_confidence.tif"
        self.save_geotiff(confidence_scores, img_A_path, conf_map_path)
        
        # Create visualization
        vis_path = f"{output_dir}/visualizations/{base_name}_visualization.png"
        self.create_visualization(img_A_path, img_B_path, change_map, 
                                confidence_scores, vis_path)
        
        # Print results
        print(f"Change detection results for {base_name}:")
        print(f"- Total pixels: {total_pixels:,}")
        print(f"- Changed pixels: {changed_pixels:,} ({change_percentage:.2f}%)")
        print(f"- Confident changes: {confident_changed_pixels:,} ({confident_change_percentage:.2f}%)")
        print(f"- Average confidence: {avg_confidence:.3f}")
        print(f"Results saved in: {output_dir}")
        
        return change_map, confidence_scores
    
    def save_geotiff(self, data, reference_path, output_path):
        """Save numpy array as GeoTIFF with same georeference as input"""
        with rasterio.open(reference_path) as src:
            profile = src.profile.copy()
            profile.update({
                'count': 1,
                'dtype': data.dtype,
                'height': data.shape[0],
                'width': data.shape[1]
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
    
    def create_visualization(self, img_A_path, img_B_path, change_map, 
                           confidence_scores, output_path):
        """Create a comprehensive visualization"""
        img_A_vis = self.load_image_for_display(img_A_path)
        img_B_vis = self.load_image_for_display(img_B_path)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(img_A_vis)
        axes[0, 0].set_title('Pre-change Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(img_B_vis)
        axes[0, 1].set_title('Post-change Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(change_map, cmap='Reds', alpha=0.8)
        axes[0, 2].set_title('Detected Changes')
        axes[0, 2].axis('off')
        
        im1 = axes[1, 0].imshow(confidence_scores, cmap='viridis', vmin=0, vmax=1)
        axes[1, 0].set_title('Confidence Scores')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
        
        axes[1, 1].imshow(img_B_vis)
        axes[1, 1].imshow(change_map, cmap='Reds', alpha=0.5)
        axes[1, 1].set_title('Changes Overlay')
        axes[1, 1].axis('off')
        
        axes[1, 2].hist(confidence_scores.flatten(), bins=50, alpha=0.7, color='blue')
        axes[1, 2].set_title('Confidence Distribution')
        axes[1, 2].set_xlabel('Confidence Score')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
    
    def load_image_for_display(self, img_path, bands=[4, 3, 2]):
        """Load image for display purposes (without normalization)"""
        with rasterio.open(img_path) as src:
            img_data = []
            for band_idx in bands:
                if band_idx <= src.count:
                    band = src.read(band_idx).astype(np.float32)
                    img_data.append(band)
            img = np.stack(img_data, axis=-1)
        
        p2, p98 = np.percentile(img, [2, 98])
        img_stretched = np.clip((img - p2) / (p98 - p2), 0, 1)
        
        return img_stretched

def single_pair_example():
    """Example: Inference on a single pair of images"""
    model_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\fine_tuned_models\best_custom_model.pt"
    img_A_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\A\0030.tif"
    img_B_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\B\0030.tif"
    output_dir = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\single_pair_inference_results"
    
    processor = SinglePairInferenceProcessor(model_path)
    change_map, confidence_scores = processor.predict_single_pair(img_A_path, img_B_path, output_dir, confidence_threshold=0.5)

if __name__ == "__main__":
    single_pair_example()