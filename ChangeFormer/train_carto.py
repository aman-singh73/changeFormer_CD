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
warnings.filterwarnings('ignore')


# Suppress specific warnings
warnings.filterwarnings(
    'ignore',
    message=r'\*has 4 bands, using first 3 as RGB\*'
)

# Or if it's coming from a logging system
logging.getLogger('your_library_name').setLevel(logging.ERROR)

class PostProcessor:
    """Advanced postprocessing for change detection predictions"""
    
    def __init__(self, 
                 min_component_size=10,
                 morphology_kernel_size=3,
                 adaptive_threshold=True,
                 gaussian_sigma=0.5):
        self.min_component_size = min_component_size
        self.morphology_kernel_size = morphology_kernel_size
        self.adaptive_threshold = adaptive_threshold
        self.gaussian_sigma = gaussian_sigma
        
    def postprocess_prediction(self, prediction, confidence=None):
        """Apply comprehensive postprocessing to prediction"""
        # Ensure binary prediction
        if prediction.max() <= 1.0:
            binary_pred = (prediction > 0.5).astype(np.uint8)
        else:
            binary_pred = (prediction > 0).astype(np.uint8)
            
        # Step 1: Morphological operations
        processed = self._apply_morphological_ops(binary_pred)
        
        # Step 2: Connected component filtering
        processed = self._filter_small_components(processed)
        
        # Step 3: Adaptive thresholding if confidence map is available
        if confidence is not None and self.adaptive_threshold:
            processed = self._apply_adaptive_threshold(processed, confidence)
            
        # Step 4: Final smoothing
        processed = self._smooth_boundaries(processed)
        
        return processed
    
    def _apply_morphological_ops(self, binary_mask):
        """Apply morphological operations to clean up prediction"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (self.morphology_kernel_size, self.morphology_kernel_size))
        
        # Opening to remove noise
        opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        return closed
    
    def _filter_small_components(self, binary_mask):
        """Remove small connected components"""
        # Label connected components
        labeled_mask,     features = ndimage.label(binary_mask)
        
        # Calculate component sizes
        component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
        
        # Create mask for components to keep
        mask_sizes = component_sizes >= self.min_component_size
        remove_pixel = mask_sizes[labeled_mask]
        
        # Apply filtering
        filtered_mask = binary_mask.copy()
        filtered_mask[~remove_pixel] = 0
        
        return filtered_mask
    
    def _apply_adaptive_threshold(self, binary_mask, confidence_map):
        """Apply adaptive thresholding based on local confidence"""
        # Calculate adaptive threshold per region
        h, w = binary_mask.shape
        tile_size = 64
        
        refined_mask = binary_mask.copy()
        
        for i in range(0, h, tile_size):
            for j in range(0, w, tile_size):
                tile_mask = binary_mask[i:i+tile_size, j:j+tile_size]
                tile_conf = confidence_map[i:i+tile_size, j:j+tile_size]
                
                if tile_mask.sum() > 0:
                    # Calculate local threshold based on confidence distribution
                    local_threshold = np.percentile(tile_conf[tile_mask > 0], 25)
                    refined_tile = (tile_conf > local_threshold) & (tile_mask > 0)
                    refined_mask[i:i+tile_size, j:j+tile_size] = refined_tile.astype(np.uint8)
        
        return refined_mask
    
    def _smooth_boundaries(self, binary_mask):
        """Smooth boundaries using Gaussian filter"""
        if self.gaussian_sigma > 0:
            smoothed = ndimage.gaussian_filter(binary_mask.astype(float), sigma=self.gaussian_sigma)
            return (smoothed > 0.5).astype(np.uint8)
        return binary_mask

class HardExampleMiner:
    """Hard example mining for focusing on difficult pixels"""
    
    def __init__(self, ratio=0.3, min_kept=10000):
        self.ratio = ratio  # Ratio of hard examples to keep
        self.min_kept = min_kept
        
    def mine_hard_examples(self, predictions, targets, loss_map):
        """Mine hard examples based on loss values"""
        batch_size = predictions.shape[0]
        hard_masks = []
        
        for i in range(batch_size):
            pred_i = predictions[i]
            target_i = targets[i]
            loss_i = loss_map[i]
            
            # Flatten for easier processing
            loss_flat = loss_i.flatten()
            target_flat = target_i.flatten()
            
            # Calculate number of pixels to keep
            total_pixels = len(loss_flat)
            num_hard = max(int(total_pixels * self.ratio), self.min_kept)
            num_hard = min(num_hard, total_pixels)
            
            # Get indices of hardest examples
            hard_indices = np.argsort(loss_flat)[-num_hard:]
            
            # Create mask
            hard_mask = np.zeros_like(loss_flat, dtype=bool)
            hard_mask[hard_indices] = True
            hard_mask = hard_mask.reshape(loss_i.shape)
            
            hard_masks.append(hard_mask)
        
        return np.array(hard_masks)

class CombinedLoss(torch.nn.Module):
    """Combined Dice + Focal Loss for better change detection"""
    
    def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.eps = 1e-8
        
    def forward(self, predictions, targets):
        """Calculate combined loss"""
        # Dice Loss
        dice_loss = self._dice_loss(predictions, targets)
        
        # Focal Loss
        focal_loss = self._focal_loss(predictions, targets)
        
        # Combine losses
        total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        
        return total_loss, dice_loss, focal_loss
    
    def _dice_loss(self, predictions, targets):
        """Calculate Dice loss"""
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=1)
        
        # Get probability for positive class
        if probs.shape[1] == 2:
            probs_pos = probs[:, 1, :, :]
        else:
            probs_pos = torch.sigmoid(probs.squeeze(1))
        
        # Flatten tensors
        probs_flat = probs_pos.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice_coeff = (2.0 * intersection + self.eps) / (probs_flat.sum() + targets_flat.sum() + self.eps)
        
        return 1 - dice_coeff
    
    def _focal_loss(self, predictions, targets):
        """Calculate Focal loss"""
        ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')
        
        # Calculate p_t
        pt = torch.exp(-ce_loss)
        
        # Calculate alpha_t
        alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
        
        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()

class TrainingVisualizer:
    """Enhanced visualizer with postprocessing comparison"""
    
    def __init__(self, vis_dir, exp_name):
        self.vis_dir = Path(vis_dir) / exp_name
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking variables
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.learning_rates = []
        self.epochs = []
        
        # Loss component tracking
        self.dice_losses = []
        self.focal_losses = []
        
        # Create subdirectories
        self.plots_dir = self.vis_dir / "plots"
        self.predictions_dir = self.vis_dir / "predictions"
        self.postprocessed_dir = self.vis_dir / "postprocessed"
        self.plots_dir.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True)
        self.postprocessed_dir.mkdir(exist_ok=True)
        
        # Initialize postprocessor
        self.postprocessor = PostProcessor()
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
        print(f" Enhanced Visualizer initialized. Saving to: {self.vis_dir}")
    
    def update_metrics(self, epoch, train_loss=None, val_loss=None, 
                      train_f1=None, val_f1=None, lr=None,
                      dice_loss=None, focal_loss=None):
        """Update training metrics including loss components"""
        self.epochs.append(epoch)
        
        if train_loss is not None:
            self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_f1 is not None:
            self.train_f1_scores.append(train_f1)
        if val_f1 is not None:
            self.val_f1_scores.append(val_f1)
        if lr is not None:
            self.learning_rates.append(lr)
        if dice_loss is not None:
            self.dice_losses.append(dice_loss)
        if focal_loss is not None:
            self.focal_losses.append(focal_loss)
    
    def plot_training_curves(self, save_plots=True):
        """Create comprehensive training curves with loss components"""
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Define colors
        train_color = '#2E86AB'
        val_color = '#A23B72'
        lr_color = '#F18F01'
        dice_color = '#E63946'
        focal_color = '#457B9D'
        
        # Plot 1: Loss Curves
        ax1 = fig.add_subplot(gs[0, 0])
        if self.train_losses:
            ax1.plot(self.epochs[:len(self.train_losses)], self.train_losses, 
                    label='Training Loss', color=train_color, linewidth=2.5, marker='o', markersize=4)
        if self.val_losses:
            ax1.plot(self.epochs[:len(self.val_losses)], self.val_losses, 
                    label='Validation Loss', color=val_color, linewidth=2.5, marker='s', markersize=4)
        
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        # Plot 2: F1 Score Curves
        ax2 = fig.add_subplot(gs[0, 1])
        if self.train_f1_scores:
            ax2.plot(self.epochs[:len(self.train_f1_scores)], self.train_f1_scores, 
                    label='Training F1', color=train_color, linewidth=2.5, marker='o', markersize=4)
        if self.val_f1_scores:
            ax2.plot(self.epochs[:len(self.val_f1_scores)], self.val_f1_scores, 
                    label='Validation F1', color=val_color, linewidth=2.5, marker='s', markersize=4)
        
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('Training & Validation F1 Score', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        ax2.set_facecolor('#f8f9fa')
        
        # Plot 3: Learning Rate
        ax3 = fig.add_subplot(gs[0, 2])
        if self.learning_rates:
            ax3.plot(self.epochs[:len(self.learning_rates)], self.learning_rates, 
                    color=lr_color, linewidth=2.5, marker='d', markersize=4)
            ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
            ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
            ax3.set_facecolor('#f8f9fa')
        
        # Plot 4: Loss Components
        ax4 = fig.add_subplot(gs[0, 3])
        if self.dice_losses and self.focal_losses:
            ax4.plot(self.epochs[:len(self.dice_losses)], self.dice_losses, 
                    label='Dice Loss', color=dice_color, linewidth=2.5, marker='o', markersize=4)
            ax4.plot(self.epochs[:len(self.focal_losses)], self.focal_losses, 
                    label='Focal Loss', color=focal_color, linewidth=2.5, marker='s', markersize=4)
            ax4.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Component Loss', fontsize=12, fontweight='bold')
            ax4.set_title('Loss Components', fontsize=14, fontweight='bold')
            ax4.legend(fontsize=11)
            ax4.grid(True, alpha=0.3)
            ax4.set_facecolor('#f8f9fa')
        
        # Plot 5: Loss vs F1 Correlation (Training)
        ax5 = fig.add_subplot(gs[1, 0])
        if self.train_losses and self.train_f1_scores:
            min_len = min(len(self.train_losses), len(self.train_f1_scores))
            scatter = ax5.scatter(self.train_losses[:min_len], self.train_f1_scores[:min_len], 
                                c=range(min_len), cmap='viridis', alpha=0.7, s=50)
            ax5.set_xlabel('Training Loss', fontsize=12, fontweight='bold')
            ax5.set_ylabel('Training F1 Score', fontsize=12, fontweight='bold')
            ax5.set_title('Training: Loss vs F1 Score', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax5, label='Epoch')
            ax5.grid(True, alpha=0.3)
            ax5.set_facecolor('#f8f9fa')
        
        # Plot 6: Loss vs F1 Correlation (Validation)
        ax6 = fig.add_subplot(gs[1, 1])
        if self.val_losses and self.val_f1_scores:
            min_len = min(len(self.val_losses), len(self.val_f1_scores))
            scatter = ax6.scatter(self.val_losses[:min_len], self.val_f1_scores[:min_len], 
                                c=range(min_len), cmap='plasma', alpha=0.7, s=50)
            ax6.set_xlabel('Validation Loss', fontsize=12, fontweight='bold')
            ax6.set_ylabel('Validation F1 Score', fontsize=12, fontweight='bold')
            ax6.set_title('Validation: Loss vs F1 Score', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax6, label='Epoch')
            ax6.grid(True, alpha=0.3)
            ax6.set_facecolor('#f8f9fa')
        
        # Plot 7 & 8: Training Summary Stats
        ax7 = fig.add_subplot(gs[1, 2:])
        ax7.axis('off')
        
        # Create enhanced summary statistics
        summary_text = []
        if self.train_losses:
            summary_text.append(f"📈 TRAINING METRICS:")
            summary_text.append(f"   Loss - Initial: {self.train_losses[0]:.4f}, Final: {self.train_losses[-1]:.4f}, Best: {min(self.train_losses):.4f}")
            if self.train_f1_scores:
                summary_text.append(f"   F1   - Initial: {self.train_f1_scores[0]:.4f}, Final: {self.train_f1_scores[-1]:.4f}, Best: {max(self.train_f1_scores):.4f}")
            summary_text.append("")
            
        if self.val_losses:
            summary_text.append(f" VALIDATION METRICS:")
            summary_text.append(f"   Loss - Initial: {self.val_losses[0]:.4f}, Final: {self.val_losses[-1]:.4f}, Best: {min(self.val_losses):.4f}")
            if self.val_f1_scores:
                summary_text.append(f"   F1   - Initial: {self.val_f1_scores[0]:.4f}, Final: {self.val_f1_scores[-1]:.4f}, Best: {max(self.val_f1_scores):.4f}")
            summary_text.append("")
            
        if self.dice_losses and self.focal_losses:
            summary_text.append(f"🎯 LOSS COMPONENTS (Final):")
            summary_text.append(f"   Dice Loss:  {self.dice_losses[-1]:.4f}")
            summary_text.append(f"   Focal Loss: {self.focal_losses[-1]:.4f}")
            summary_text.append("")
            
        if self.val_f1_scores:
            # Performance assessment
            final_f1 = self.val_f1_scores[-1]
            best_f1 = max(self.val_f1_scores)
            improvement = best_f1 - self.val_f1_scores[0] if len(self.val_f1_scores) > 0 else 0
            
            summary_text.append(f"🏆 PERFORMANCE ASSESSMENT:")
            summary_text.append(f"   Current F1: {final_f1:.4f}")
            summary_text.append(f"   Best F1:    {best_f1:.4f}")
            summary_text.append(f"   Improvement: +{improvement:.4f}")
            
            if best_f1 > 0.8:
                summary_text.append(f"   Status: 🟢 Excellent")
            elif best_f1 > 0.6:
                summary_text.append(f"   Status: 🟡 Good")
            else:
                summary_text.append(f"   Status: 🔴 Needs Improvement")
                
        summary_str = "\n".join(summary_text)
        ax7.text(0.1, 0.9, summary_str, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        ax7.set_title('Enhanced Training Summary', fontsize=14, fontweight='bold')
        
        # Plot 9: F1 Score improvement over time
        ax9 = fig.add_subplot(gs[2, :2])
        if self.val_f1_scores:
            # Calculate moving average
            window_size = max(1, len(self.val_f1_scores) // 10)
            if len(self.val_f1_scores) >= window_size:
                moving_avg = np.convolve(self.val_f1_scores, np.ones(window_size)/window_size, mode='valid')
                epochs_ma = self.epochs[window_size-1:len(self.val_f1_scores)]
                ax9.plot(epochs_ma, moving_avg, label='Moving Average', color='red', linewidth=3, alpha=0.7)
            
            ax9.plot(self.epochs[:len(self.val_f1_scores)], self.val_f1_scores, 
                    label='Validation F1', color=val_color, linewidth=2, marker='o', markersize=3)
            ax9.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax9.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
            ax9.set_title('F1 Score Progression with Trend', fontsize=14, fontweight='bold')
            ax9.legend(fontsize=11)
            ax9.grid(True, alpha=0.3)
            ax9.set_facecolor('#f8f9fa')
        
        # Plot 10: Training efficiency
        ax10 = fig.add_subplot(gs[2, 2:])
        if len(self.val_f1_scores) > 1:
            # Calculate improvement rate
            improvements = np.diff(self.val_f1_scores)
            epochs_diff = self.epochs[1:len(self.val_f1_scores)]
            
            ax10.bar(epochs_diff, improvements, alpha=0.7, color='lightblue', edgecolor='navy')
            ax10.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax10.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax10.set_ylabel('F1 Score Improvement', fontsize=12, fontweight='bold')
            ax10.set_title('Per-Epoch F1 Score Improvement', fontsize=14, fontweight='bold')
            ax10.grid(True, alpha=0.3)
            ax10.set_facecolor('#f8f9fa')
        
        plt.suptitle(f'Enhanced Training Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                     fontsize=18, fontweight='bold')
        
        if save_plots:
            plt.savefig(self.plots_dir / 'enhanced_training_curves.png', dpi=300, bbox_inches='tight')
            plt.savefig(self.plots_dir / 'enhanced_training_curves.pdf', bbox_inches='tight')
        
        return fig
    
    def plot_predictions_with_postprocessing(self, images_A, images_B, predictions, ground_truths, 
                                           epoch, batch_idx=0, num_samples=4):
        """Visualize predictions with postprocessing comparison"""
        num_samples = min(num_samples, len(images_A))
        
        fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Convert tensors to numpy and handle different formats
            img_A = self._tensor_to_rgb(images_A[i])
            img_B = self._tensor_to_rgb(images_B[i])
            pred_raw = self._tensor_to_binary_mask(predictions[i])
            gt = self._tensor_to_binary_mask(ground_truths[i])
            
            # Apply postprocessing
            pred_processed = self.postprocessor.postprocess_prediction(pred_raw)
            
            # Plot Image A
            axes[i, 0].imshow(img_A)
            axes[i, 0].set_title(f'Image A (t1)', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Plot Image B  
            axes[i, 1].imshow(img_B)
            axes[i, 1].set_title(f'Image B (t2)', fontweight='bold')
            axes[i, 1].axis('off')
            
            # Plot Ground Truth
            axes[i, 2].imshow(gt, cmap='Reds', alpha=0.8)
            axes[i, 2].set_title(f'Ground Truth', fontweight='bold')
            axes[i, 2].axis('off')
            
            # Plot Raw Prediction
            axes[i, 3].imshow(pred_raw, cmap='Blues', alpha=0.8)
            axes[i, 3].set_title(f'Raw Prediction', fontweight='bold')
            axes[i, 3].axis('off')
            
            # Plot Processed Prediction
            axes[i, 4].imshow(pred_processed, cmap='Greens', alpha=0.8)
            axes[i, 4].set_title(f'Postprocessed', fontweight='bold')
            axes[i, 4].axis('off')
            
            # Plot Overlay comparison
            overlay = self._create_comparison_overlay(img_B, gt, pred_raw, pred_processed)
            axes[i, 5].imshow(overlay)
            axes[i, 5].set_title(f'Comparison Overlay', fontweight='bold')
            axes[i, 5].axis('off')
            
            # Add metrics
            if pred_raw.shape == gt.shape:
                # Raw metrics
                iou_raw = self._calculate_iou(pred_raw, gt)
                f1_raw = self._calculate_f1(pred_raw, gt)
                
                # Processed metrics
                iou_proc = self._calculate_iou(pred_processed, gt)
                f1_proc = self._calculate_f1(pred_processed, gt)
                
                # Display metrics
                axes[i, 3].text(0.02, 0.98, f'IoU: {iou_raw:.3f}\nF1: {f1_raw:.3f}', 
                               transform=axes[i, 3].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8),
                               verticalalignment='top', fontsize=10)
                
                axes[i, 4].text(0.02, 0.98, f'IoU: {iou_proc:.3f}\nF1: {f1_proc:.3f}', 
                               transform=axes[i, 4].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8),
                               verticalalignment='top', fontsize=10)
                
                # Show improvement
                improvement = f1_proc - f1_raw
                color = 'green' if improvement > 0 else 'red'
                axes[i, 5].text(0.02, 0.98, f'F1 Δ: {improvement:+.3f}', 
                               transform=axes[i, 5].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.6),
                               verticalalignment='top', fontsize=10, color='white')
        
        plt.suptitle(f'Predictions with Postprocessing - Epoch {epoch} | Batch {batch_idx}', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save predictions
        pred_filename = f'postprocessed_predictions_epoch_{epoch:03d}_batch_{batch_idx:03d}.png'
        plt.savefig(self.postprocessed_dir / pred_filename, dpi=200, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def _create_comparison_overlay(self, base_image, gt, pred_raw, pred_processed):
        """Create overlay showing GT, raw prediction, and processed prediction"""
        overlay = base_image.copy()
        
        # Convert to RGB if grayscale
        if len(overlay.shape) == 2:
            overlay = np.stack([overlay, overlay, overlay], axis=2)
        
        # Create colored masks
        gt_mask = gt > 0
        pred_raw_mask = pred_raw > 0
        pred_proc_mask = pred_processed > 0
        
        # GT in red
        overlay[gt_mask, 0] = np.maximum(overlay[gt_mask, 0], 0.8)
        
        # Raw prediction in blue (false positives)
        fp_raw = pred_raw_mask & ~gt_mask
        overlay[fp_raw, 2] = np.maximum(overlay[fp_raw, 2], 0.6)
        
        # Processed prediction in green (should have fewer false positives)
        fp_proc = pred_proc_mask & ~gt_mask
        overlay[fp_proc, 1] = np.maximum(overlay[fp_proc, 1], 0.6)
        
        # True positives in yellow (overlap of GT and processed)
        tp_proc = pred_proc_mask & gt_mask
        overlay[tp_proc, 0] = np.maximum(overlay[tp_proc, 0], 0.7)
        overlay[tp_proc, 1] = np.maximum(overlay[tp_proc, 1], 0.7)
        
        return overlay
    
    def _tensor_to_rgb(self, tensor):
        """Convert tensor to RGB image for visualization"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        # Handle different tensor shapes
        if tensor.ndim == 3:  # (C, H, W)
            if tensor.shape[0] == 3:  # RGB
                img = np.transpose(tensor, (1, 2, 0))
            else:  # Take first 3 channels
                img = np.transpose(tensor[:3], (1, 2, 0))
        elif tensor.ndim == 2:  # (H, W)
            img = np.stack([tensor, tensor, tensor], axis=2)
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # Normalize to [0, 1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return np.clip(img, 0, 1)
    
    def _tensor_to_binary_mask(self, tensor):
        """Convert tensor to binary mask"""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:  # (1, H, W)
                mask = tensor[0]
            else:  # (C, H, W) - take first channel or argmax
                mask = tensor[0] if tensor.shape[0] == 2 else np.argmax(tensor, axis=0)
        elif tensor.ndim == 2:  # (H, W)
            mask = tensor
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        
        # Convert to binary
        if mask.max() > 1:
            mask = (mask > 0.5).astype(np.uint8)
        else:
            mask = (mask > 0.5).astype(np.uint8)
            
        return mask
    
    def _calculate_iou(self, pred, gt):
        """Calculate IoU between prediction and ground truth"""
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        return intersection / (union + 1e-8)
    
    def _calculate_f1(self, pred, gt):
        """Calculate F1 score between prediction and ground truth"""
        tp = np.logical_and(pred, gt).sum()
        fp = np.logical_and(pred, np.logical_not(gt)).sum()
        fn = np.logical_and(np.logical_not(pred), gt).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        return f1
    
    def save_metrics_json(self):
        """Save enhanced metrics to JSON file"""
        metrics = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_f1_scores': self.train_f1_scores,
            'val_f1_scores': self.val_f1_scores,
            'learning_rates': self.learning_rates,
            'dice_losses': self.dice_losses,
            'focal_losses': self.focal_losses,
            'timestamp': datetime.now().isoformat(),
            'best_val_f1': max(self.val_f1_scores) if self.val_f1_scores else 0.0,
            'final_val_f1': self.val_f1_scores[-1] if self.val_f1_scores else 0.0,
        }
        
        with open(self.vis_dir / 'enhanced_training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Enhanced metrics saved to: {self.vis_dir / 'enhanced_training_metrics.json'}")

def fix_dataset_nan_values(data_root):
    """Fix NaN values in the dataset before training"""
    import rasterio
    from pathlib import Path
    
    print("🔧 Fixing NaN values in dataset...")
    
    data_path = Path(data_root)
    nan_files_fixed = 0
    
    for split in ['A', 'B']:
        img_dir = data_path / split
        if not img_dir.exists():
            print(f"❌ Directory {img_dir} not found!")
            continue
            
        for img_path in img_dir.glob('*.tif'):
            try:
                with rasterio.open(img_path, 'r+') as src:
                    data = src.read()
                    
                    if np.isnan(data).any():
                        print(f"   Fixing NaN in {img_path.name}")
                        nan_files_fixed += 1
                        
                        # Fix NaN values band by band
                        for band_idx in range(data.shape[0]):
                            band_data = data[band_idx]
                            if np.isnan(band_data).any():
                                # Calculate statistics from valid pixels
                                valid_pixels = band_data[~np.isnan(band_data)]
                                if len(valid_pixels) > 100:
                                    # Use median for robust replacement
                                    replacement_value = np.median(valid_pixels)
                                elif len(valid_pixels) > 0:
                                    replacement_value = np.mean(valid_pixels)
                                else:
                                    replacement_value = 0.0
                                
                                band_data[np.isnan(band_data)] = replacement_value
                        
                        # Write back the fixed data
                        src.write(data)
                        
            except Exception as e:
                print(f"   ❌ Error fixing {img_path.name}: {e}")
    
    print(f"✅ Fixed NaN values in {nan_files_fixed} files")
    return nan_files_fixed > 0

def calculate_class_weights(data_root, split_file='train.txt'):
    """Calculate class weights for imbalanced dataset"""
    import rasterio
    from PIL import Image
    
    print(" Calculating class weights...")
    
    # Load training file names
    list_path = Path(data_root) / 'list' / split_file
    with open(list_path, 'r') as f:
        file_names = [line.strip() for line in f.readlines()]
    
    total_pixels = 0
    change_pixels = 0
    
    label_dir = Path(data_root) / 'label'
    
    for file_name in file_names:
        base_name = file_name.replace('.tif', '')
        
        label_paths = [
            label_dir / f"{base_name}.tif",
            label_dir / f"{base_name}.png"
        ]
        
        label_path = None
        for path in label_paths:
            if path.exists():
                label_path = path
                break
        
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
            
            if label_data.max() > 1:
                label_data = (label_data > 0).astype(np.uint8)
            
            total_pixels += label_data.size
            change_pixels += np.sum(label_data > 0)
                
        except Exception as e:
            print(f"   Error reading {label_path}: {e}")
    
    if total_pixels > 0:
        change_ratio = change_pixels / total_pixels
        no_change_ratio = 1 - change_ratio
        
        no_change_weight = 1.0
        change_weight = no_change_ratio / change_ratio if change_ratio > 0 else 10.0
        
        print(f"   Change ratio: {change_ratio:.4f} ({change_ratio*100:.2f}%)")
        print(f"   Weights - No change: {no_change_weight:.2f}, Change: {change_weight:.2f}")
        
        # Convert to Python float to avoid np.float64
        return [float(no_change_weight), float(change_weight)]
    else:
        print("   ❌ Could not calculate class weights")
        return [1.0, 10.0]

def train(args):
    """Enhanced training function with stronger backbone and postprocessing"""
    
    # Fix NaN values before training
    if args.fix_nan_values:
        fix_dataset_nan_values(args.data_root)
    
    # Calculate class weights for imbalanced data
    if args.auto_class_weights:
        class_weights = calculate_class_weights(args.data_root)
        args.class_weights = [float(w) for w in class_weights]  # Ensure float
        print(f"🎯 Using calculated class weights: {args.class_weights}")
    
    # Ensure integer parameters
    args.batch_size = int(args.batch_size)
    args.num_workers = int(args.num_workers)
    args.max_epochs = int(args.max_epochs)
    args.img_size = int(args.img_size)
    args.vis_freq = int(args.vis_freq)
    args.save_epoch_freq = int(args.save_epoch_freq)
    args.early_stopping_patience = int(args.early_stopping_patience)
    args.warmup_epochs = int(args.warmup_epochs)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Initialize enhanced visualizer
    visualizer = TrainingVisualizer(args.vis_dir, args.exp_name)
    args.visualizer = visualizer  # Pass to trainer
    
    # Initialize hard example miner
    args.hard_example_miner = HardExampleMiner(ratio=0.3, min_kept=10000)
    
    print(f"🔥 Using ENHANCED TRAINING with stronger backbone and postprocessing")
    print(f" Backbone: {args.backbone}")
    print(f"🖼️ Image size: {args.img_size}")
    print(f"🎯 Loss function: {args.loss}")
    
    dataloaders = utils.get_loaders(args)
    
    # Convert class weights to tensor
    args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
    
    model = CDTrainer(args=args, dataloaders=dataloaders)
    
    # Train the model
    model.train_models()
    
    # Final visualization
    print(" Creating final enhanced training visualizations...")
    visualizer.plot_training_curves()
    visualizer.save_metrics_json()
    
    return model

def setup_device(args):
    """Setup device and GPU configuration"""
    if torch.cuda.is_available() and args.gpu_ids:
        args.device = f'cuda:{args.gpu_ids[0]}'
        print(f"🚀 Using GPU: {args.device}")
        print(f"   Available GPUs: {args.gpu_ids}")
        
        # Print GPU memory info
        gpu_memory = torch.cuda.get_device_properties(args.gpu_ids[0]).total_memory
        print(f"   GPU Memory: {gpu_memory / 1024**3:.1f} GB")
    else:
        args.device = 'cpu'
        print("⚠️ Using CPU (GPU not available)")
    
    return args

def validate_args(args):
    """Validate and fix argument values"""
    
    # Validate data paths
    data_path = Path(args.data_root)
    if not data_path.exists():
        print(f"❌ Data root directory does not exist: {args.data_root}")
        return False
    
    # Check required directories
    required_dirs = ['A', 'B', 'label', 'list']
    missing_dirs = []
    for req_dir in required_dirs:
        if not (data_path / req_dir).exists():
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        print(f"❌ Missing required directories: {missing_dirs}")
        print(f"   Expected structure: {args.data_root}/{{A,B,label,list}}/")
        return False
    
    # Check train.txt exists
    train_list = data_path / 'list' / 'train.txt'
    if not train_list.exists():
        print(f"❌ Training list not found: {train_list}")
        return False
    
    # Validate RGB bands
    if len(args.rgb_bands) != 3:
        print("⚠️ RGB bands should be exactly 3 values. Using default for Sentinel-2")
        args.rgb_bands = [3, 2, 1]
    
    # Validate focal loss parameters
    if args.focal_alpha < 0 or args.focal_alpha > 1:
        print("⚠️ focal_alpha should be between 0 and 1. Setting to 0.75")
        args.focal_alpha = 0.75
    
    # Validate class weights
    if len(args.class_weights) != 2:
        print("⚠️ Class weights should have exactly 2 values. Using default")
        args.class_weights = [1.0, 10.0]
    
    # Validate image size for stronger backbone
    if args.img_size < 384 and args.backbone in ['resnet50', 'resnet101']:
        print(f"⚠️ For {args.backbone}, consider img_size >= 384 for better performance")
    
    print("✅ Arguments validated successfully")
    return True

if __name__ == '__main__':
    parser = ArgumentParser(description="Enhanced ChangeFormer Training with Stronger Backbone and Postprocessing")

    # Data and experiment settings
    parser.add_argument('--data_name', type=str, default='cartoCustom')
    parser.add_argument('--data_root', type=str, default='./data/cartoCustom', 
                       help='Root directory containing A/, B/, label/, list/ folders')
    parser.add_argument('--data_format', type=str, default='tif', 
                       help='Data format: tif or standard')
    parser.add_argument('--split', type=str, default='list', help='Folder containing train.txt/val.txt')
    parser.add_argument('--dataset', type=str, default='CDDataset', help='Standard dataset class with simple augmentation')
    parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_Enhanced_Strong', help='Experiment name')

    # TIF-specific parameters
    parser.add_argument('--satellite_type', type=str, default='cartosat', 
                   choices=['cartosat', 'sentinel2', 'landsat8', 'planetscope', 'false_color', 'vegetation', 'default'],
                   help='Satellite type for RGB band selection')

    parser.add_argument('--rgb_bands', type=int, nargs=3, default=[3, 2, 1],
                       help='RGB band indices (0-based) to extract from 10-band data')
    parser.add_argument('--normalize_method', type=str, default='adaptive_percentile',
                       choices=['percentile', 'zscore', 'minmax', 'adaptive_percentile'],
                       help='Normalization method for TIF data')

    # Enhanced Model config - STRONGER BACKBONE
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--net_G', type=str, default='ChangeFormerV6', help='Network architecture')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'swin_tiny', 'swin_small'],
                       help='STRONGER backbone for better feature extraction')
    parser.add_argument('--n_class', type=int, default=2)

    # Enhanced Training params - LARGER IMAGE SIZE
    parser.add_argument('--max_epochs', type=int, default=100, help='Maximum training epochs for stronger model')
    parser.add_argument('--batch_size', type=int, default=4, help='Smaller batch size for larger images') 
    parser.add_argument('--lr', type=float, default=1e-5, help='Lower learning rate for stronger backbone')
    parser.add_argument('--img_size', type=int, default=256, help='LARGER image size for better context')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr_policy', type=str, default='cosine', 
                       choices=['linear', 'step', 'plateau', 'cosine'])
    
    parser.add_argument('--loss', type=str, default='combined', 
                       choices=['ce', 'bce', 'fl', 'focal', 'weighted_bce', 'dice', 'combined'])
    parser.add_argument('--auto_class_weights', type=bool, default=True)
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1.0, 15.0], 
                       help='Higher weight for change class')
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--dice_weight', type=float, default=0.6, help='Weight for Dice loss in combination')
    parser.add_argument('--focal_weight', type=float, default=0.4, help='Weight for Focal loss in combination')
    
    # Enhanced Data augmentation
    parser.add_argument('--use_simple_augmentation', type=bool, default=True, help='Enable simple augmentations')
    parser.add_argument('--horizontal_flip_prob', type=float, default=0.8, help='Higher flip probability')
    parser.add_argument('--vertical_flip_prob', type=float, default=0.8, help='Higher flip probability')
    parser.add_argument('--rotation_prob', type=float, default=0.6, help='Higher rotation probability')
    parser.add_argument('--color_jitter_prob', type=float, default=0.5, help='Higher color jitter probability')
    parser.add_argument('--gaussian_blur_prob', type=float, default=0.4, help='Higher blur probability')
    
    # Multi-scale and prediction settings
    parser.add_argument('--multi_scale_infer', type=bool, default=True)
    parser.add_argument('--multi_pred_weights', type=float, nargs='+', default=[1.0, 0.8, 0.6])
    parser.add_argument('--shuffle_AB', type=bool, default=True)
    parser.add_argument('--multi_scale_train', type=bool, default=True)

    # Hard Example Mining
    parser.add_argument('--use_hard_example_mining', type=bool, default=True, help='Enable hard example mining')
    parser.add_argument('--hard_example_ratio', type=float, default=0.3, help='Ratio of hard examples to keep')
    parser.add_argument('--min_hard_examples', type=int, default=10000, help='Minimum hard examples per batch')

    # Postprocessing settings
    parser.add_argument('--use_postprocessing', type=bool, default=True, help='Apply postprocessing to predictions')
    parser.add_argument('--min_component_size', type=int, default=15, help='Minimum connected component size')
    parser.add_argument('--morphology_kernel_size', type=int, default=3, help='Morphological operations kernel size')
    parser.add_argument('--adaptive_threshold', type=bool, default=True, help='Use adaptive thresholding')
    parser.add_argument('--gaussian_sigma', type=float, default=0.5, help='Gaussian smoothing sigma')

    # Checkpointing and enhanced visualization
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/ChangeFormer_myCustom/ChangeFormerV6_Enhanced_Strong')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--vis_freq', type=int, default=8, help='More frequent visualization')
    parser.add_argument('--save_vis_during_train', type=bool, default=True)
    parser.add_argument('--plot_freq', type=int, default=3, help='More frequent plotting')

    # Enhanced Pretraining
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', type=str, 
                       default=r'D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_LEVIR\ChangeFormerV6_LEVIR\LEVIR_WEIGHT\best_ckpt.pt',
                       help='Path to pretrained weights')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=30, 
                       help='More epochs to freeze stronger backbone')

    # Enhanced Training strategy
    parser.add_argument('--early_stopping_patience', type=int, default=35, help='More patience for stronger model')
    parser.add_argument('--save_epoch_freq', type=int, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Stronger gradient clipping')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='More warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Higher weight decay for regularization')
    
    # Dataset preprocessing
    parser.add_argument('--fix_nan_values', type=bool, default=True)
    parser.add_argument('--label_transform', type=str, default='binary', 
                       choices=['none', 'binary', 'norm'])

    # Validation and testing
    parser.add_argument('--val_freq', type=int, default=3, help='More frequent validation')
    parser.add_argument('--test_during_train', type=bool, default=False)
    parser.add_argument('--augment_factor', type=int, default=1,
                   help='Number of augmented copies per image (default=1 means no extra copies)')

    # Flags
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true') 
    parser.add_argument('--test', action='store_true')

    # GPU setup
    parser.add_argument('--gpu', type=str, default='0')
    
    args = parser.parse_args()

    # Parse GPU ids
    args.gpu_ids = [int(i) for i in args.gpu.split(',')] if torch.cuda.is_available() else []

    # Set RGB bands based on satellite type
    if args.satellite_type != 'default':
        rgb_configs = {
            'sentinel2': [3, 2, 1],     # Red, Green, Blue for Sentinel-2
            'landsat8': [3, 2, 1],      # Red, Green, Blue for Landsat-8  
            'planetscope': [2, 1, 0],   # RGB for PlanetScope
            'false_color': [7, 3, 2],   # NIR, Red, Green
            'vegetation': [7, 5, 3], 
            'cartosat': [3, 2, 1]      # Vegetation-focused bands
        }
        args.rgb_bands = rgb_configs.get(args.satellite_type, [3, 2, 1])

    # Setup device
    args = setup_device(args)

    # Validate arguments
    if not validate_args(args):
        print("❌ Argument validation failed. Exiting.")
        exit(1)

    # Print enhanced configuration summary
    print("🚀 ENHANCED CHANGEFORMER TRAINING")
    print("=" * 70)
    print(f"🎯 IMPROVEMENTS APPLIED:")
    print(f"   ✅ Stronger backbone: {args.backbone}")
    print(f"   ✅ Larger image size: {args.img_size}x{args.img_size}")
    print(f"   ✅ Combined loss: Dice ({args.dice_weight}) + Focal ({args.focal_weight})")
    print(f"   ✅ Hard example mining: {args.use_hard_example_mining}")
    print(f"   ✅ Postprocessing: {args.use_postprocessing}")
    print("=" * 70)
    print(f"Dataset: {args.data_name}")
    print(f"Data root: {args.data_root}")
    print(f"Data format: {args.data_format}")
    print(f"Satellite type: {args.satellite_type}")
    print(f"RGB bands: {args.rgb_bands}")
    print(f"Model: {args.net_G} with {args.backbone} backbone")
    print(f"Loss function: {args.loss}")
    print(f"Class weights: {args.class_weights}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Device: {args.device}")
    print(f"Hard example mining ratio: {args.hard_example_ratio}")
    print(f"Min component size: {args.min_component_size}")
    print("=" * 70)

    # Memory usage warning for larger settings
    if args.img_size >= 512 and args.backbone in ['resnet50', 'resnet101']:
        print("⚠️ HIGH MEMORY USAGE WARNING:")
        print(f"   Using {args.backbone} with {args.img_size}x{args.img_size} images")
        print(f"   Recommended: 8GB+ GPU memory")
        print(f"   Consider reducing batch_size if you get OOM errors")
        print("=" * 70)

    # Start enhanced training
    try:
        train(args)
        print("\n✅ Enhanced training completed successfully!")
        print(f" Check enhanced visualization results in: {args.vis_dir}/{args.exp_name}")
        print(f"🎯 Postprocessed predictions saved in: {args.vis_dir}/{args.exp_name}/postprocessed/")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Enhanced training script completed")
# import os
# import torch
# import numpy as np
# from scipy import ndimage
# from argparse import ArgumentParser
# from models.trainer import CDTrainer
# import utils
# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# import json
# from datetime import datetime
# import cv2
# from PIL import Image
# import torch.nn.functional as F
# import warnings
# warnings.filterwarnings('ignore')
# import warnings
# warnings.filterwarnings('ignore', category=FutureWarning, module='timm.models.layers')

# class PostProcessor:
#     """Simplified postprocessing for change detection predictions"""
    
#     def __init__(self, min_component_size=10, morphology_kernel_size=3):
#         self.min_component_size = min_component_size
#         self.morphology_kernel_size = morphology_kernel_size
        
#     def postprocess_prediction(self, prediction):
#         """Apply basic postprocessing to prediction"""
#         # Ensure binary prediction
#         binary_pred = (prediction > 0.5).astype(np.uint8)
            
#         # Apply morphological operations
#         processed = self._apply_morphological_ops(binary_pred)
        
#         # Filter small components
#         processed = self._filter_small_components(processed)
        
#         return processed
    
#     def _apply_morphological_ops(self, binary_mask):
#         """Apply basic morphological operations"""
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
#                                          (self.morphology_kernel_size, self.morphology_kernel_size))
#         # Only apply opening to remove noise
#         opened = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
#         return opened
    
#     def _filter_small_components(self, binary_mask):
#         """Remove small connected components"""
#         labeled_mask, num_features = ndimage.label(binary_mask)
#         component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))
#         mask_sizes = component_sizes >= self.min_component_size
#         remove_pixel = mask_sizes[labeled_mask]
#         filtered_mask = binary_mask.copy()
#         filtered_mask[~remove_pixel] = 0
#         return filtered_mask

# class CombinedLoss(torch.nn.Module):
#     """Combined Dice + Focal Loss for change detection"""
    
#     def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.5, focal_weight=0.5):
#         super(CombinedLoss, self).__init__()
#         self.focal_alpha = focal_alpha
#         self.focal_gamma = focal_gamma
#         self.dice_weight = dice_weight
#         self.focal_weight = focal_weight
#         self.eps = 1e-8
        
#     def forward(self, predictions, targets):
#         """Calculate combined loss"""
#         dice_loss = self._dice_loss(predictions, targets)
#         focal_loss = self._focal_loss(predictions, targets)
#         total_loss = self.dice_weight * dice_loss + self.focal_weight * focal_loss
#         return total_loss, dice_loss, focal_loss
    
#     def _dice_loss(self, predictions, targets):
#         """Calculate Dice loss"""
#         probs = F.softmax(predictions, dim=1)
#         if probs.shape[1] == 2:
#             probs_pos = probs[:, 1, :, :]
#         else:
#             probs_pos = torch.sigmoid(probs.squeeze(1))
        
#         probs_flat = probs_pos.contiguous().view(-1)
#         targets_flat = targets.contiguous().view(-1).float()
        
#         intersection = (probs_flat * targets_flat).sum()
#         dice_coeff = (2.0 * intersection + self.eps) / (probs_flat.sum() + targets_flat.sum() + self.eps)
#         return 1 - dice_coeff
    
#     def _focal_loss(self, predictions, targets):
#         """Calculate Focal loss"""
#         ce_loss = F.cross_entropy(predictions, targets.long(), reduction='none')
#         pt = torch.exp(-ce_loss)
#         alpha_t = self.focal_alpha * targets + (1 - self.focal_alpha) * (1 - targets)
#         focal_loss = alpha_t * (1 - pt) ** self.focal_gamma * ce_loss
#         return focal_loss.mean()

# class TrainingVisualizer:
#     """Simplified visualizer for training progress"""
    
#     def __init__(self, vis_dir, exp_name):
#         self.vis_dir = Path(vis_dir) / exp_name
#         self.vis_dir.mkdir(parents=True, exist_ok=True)
#         self.plots_dir = self.vis_dir / "plots"
#         self.predictions_dir = self.vis_dir / "predictions"
#         self.plots_dir.mkdir(exist_ok=True)
#         self.predictions_dir.mkdir(exist_ok=True)
        
#         self.postprocessor = PostProcessor()
#         plt.style.use('default')
#         sns.set_palette("husl")
        
#         self.train_losses = []
#         self.val_losses = []
#         self.train_f1_scores = []
#         self.val_f1_scores = []
#         self.epochs = []
#         self.learning_rates = []
        
#         print(f" Visualizer initialized. Saving to: {self.vis_dir}")
    
#     def update_metrics(self, epoch, train_loss=None, val_loss=None, 
#                     train_f1=None, val_f1=None, lr=None):
#         """Update training metrics"""
#         self.epochs.append(epoch)
#         if train_loss is not None:
#             self.train_losses.append(train_loss)
#         if val_loss is not None:
#             self.val_losses.append(val_loss)
#         if train_f1 is not None:
#             self.train_f1_scores.append(train_f1)
#         if val_f1 is not None:
#             self.val_f1_scores.append(val_f1)
#         if lr is not None:
#             self.learning_rates.append(lr)
    
#     def plot_training_curves(self, save_plots=True):
#         """Create simplified training curves"""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
#         # Plot Loss Curves
#         if self.train_losses:
#             ax1.plot(self.epochs[:len(self.train_losses)], self.train_losses, 
#                     label='Training Loss', color='blue', linewidth=2)
#         if self.val_losses:
#             ax1.plot(self.epochs[:len(self.val_losses)], self.val_losses, 
#                     label='Validation Loss', color='red', linewidth=2)
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Loss')
#         ax1.set_title('Training & Validation Loss')
#         ax1.legend()
#         ax1.grid(True)
        
#         # Plot F1 Score Curves
#         if self.train_f1_scores:
#             ax2.plot(self.epochs[:len(self.train_f1_scores)], self.train_f1_scores, 
#                     label='Training F1', color='blue', linewidth=2)
#         if self.val_f1_scores:
#             ax2.plot(self.epochs[:len(self.val_f1_scores)], self.val_f1_scores, 
#                     label='Validation F1', color='red', linewidth=2)
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('F1 Score')
#         ax2.set_title('Training & Validation F1 Score')
#         ax2.legend()
#         ax2.grid(True)
#         ax2.set_ylim(0, 1)
        
#         plt.suptitle(f'Training Dashboard - {datetime.now().strftime("%Y-%m-%d %H:%M")}')
#         if save_plots:
#             plt.savefig(self.plots_dir / 'training_curves.png', dpi=200, bbox_inches='tight')
#         plt.close()
#         return fig
    
#     def plot_predictions(self, images_A, images_B, predictions, ground_truths, 
#                         epoch, batch_idx=0, num_samples=4):
#         """Visualize predictions with simplified postprocessing"""
#         num_samples = min(num_samples, len(images_A))
#         fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
#         if num_samples == 1:
#             axes = axes.reshape(1, -1)
        
#         for i in range(num_samples):
#             img_A = self._tensor_to_rgb(images_A[i])
#             img_B = self._tensor_to_rgb(images_B[i])
#             pred_raw = self._tensor_to_binary_mask(predictions[i])
#             gt = self._tensor_to_binary_mask(ground_truths[i])
#             pred_processed = self.postprocessor.postprocess_prediction(pred_raw)
            
#             axes[i, 0].imshow(img_A)
#             axes[i, 0].set_title('Image A (t1)')
#             axes[i, 0].axis('off')
            
#             axes[i, 1].imshow(img_B)
#             axes[i, 1].set_title('Image B (t2)')
#             axes[i, 1].axis('off')
            
#             axes[i, 2].imshow(gt, cmap='Reds', alpha=0.8)
#             axes[i, 2].set_title('Ground Truth')
#             axes[i, 2].axis('off')
            
#             axes[i, 3].imshow(pred_processed, cmap='Greens', alpha=0.8)
#             axes[i, 3].set_title('Prediction')
#             axes[i, 3].axis('off')
            
#             iou = self._calculate_iou(pred_processed, gt)
#             f1 = self._calculate_f1(pred_processed, gt)
#             axes[i, 3].text(0.02, 0.98, f'IoU: {iou:.3f}\nF1: {f1:.3f}', 
#                            transform=axes[i, 3].transAxes, 
#                            bbox=dict(facecolor='white', alpha=0.8),
#                            verticalalignment='top')
        
#         plt.suptitle(f'Predictions - Epoch {epoch} | Batch {batch_idx}')
#         plt.tight_layout()
#         plt.savefig(self.predictions_dir / f'predictions_epoch_{epoch:03d}_batch_{batch_idx:03d}.png', 
#                    dpi=200, bbox_inches='tight')
#         plt.close()
#         return fig
    
#     def _tensor_to_rgb(self, tensor):
#         """Convert tensor to RGB image"""
#         if isinstance(tensor, torch.Tensor):
#             tensor = tensor.detach().cpu().numpy()
#         if tensor.ndim == 3:
#             if tensor.shape[0] == 3:
#                 img = np.transpose(tensor, (1, 2, 0))
#             else:
#                 img = np.transpose(tensor[:3], (1, 2, 0))
#         elif tensor.ndim == 2:
#             img = np.stack([tensor, tensor, tensor], axis=2)
#         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
#         return np.clip(img, 0, 1)
    
#     def _tensor_to_binary_mask(self, tensor):
#         """Convert tensor to binary mask"""
#         if isinstance(tensor, torch.Tensor):
#             tensor = tensor.detach().cpu().numpy()
#         if tensor.ndim == 3:
#             if tensor.shape[0] == 1:
#                 mask = tensor[0]
#             else:
#                 mask = tensor[0] if tensor.shape[0] == 2 else np.argmax(tensor, axis=0)
#         elif tensor.ndim == 2:
#             mask = tensor
#         mask = (mask > 0.5).astype(np.uint8)
#         return mask
    
#     def _calculate_iou(self, pred, gt):
#         """Calculate IoU"""
#         intersection = np.logical_and(pred, gt).sum()
#         union = np.logical_or(pred, gt).sum()
#         return intersection / (union + 1e-8)
    
#     def _calculate_f1(self, pred, gt):
#         """Calculate F1 score"""
#         tp = np.logical_and(pred, gt).sum()
#         fp = np.logical_and(pred, np.logical_not(gt)).sum()
#         fn = np.logical_and(np.logical_not(pred), gt).sum()
#         precision = tp / (tp + fp + 1e-8)
#         recall = tp / (tp + fn + 1e-8)
#         f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
#         return f1
    
#     def save_metrics_json(self):
#         """Save metrics to JSON"""
#         metrics = {
#             'epochs': self.epochs,
#             'train_losses': self.train_losses,
#             'val_losses': self.val_losses,
#             'train_f1_scores': self.train_f1_scores,
#             'val_f1_scores': self.val_f1_scores,
#             'timestamp': datetime.now().isoformat(),
#             'best_val_f1': max(self.val_f1_scores) if self.val_f1_scores else 0.0,
#         }
#         with open(self.vis_dir / 'training_metrics.json', 'w') as f:
#             json.dump(metrics, f, indent=2)
#         print(f"💾 Metrics saved to: {self.vis_dir / 'training_metrics.json'}")

# def fix_dataset_nan_values(data_root):
#     """Fix NaN values in dataset"""
#     import rasterio
#     from pathlib import Path
    
#     print("🔧 Fixing NaN values in dataset...")
#     data_path = Path(data_root)
#     nan_files_fixed = 0
    
#     for split in ['A', 'B']:
#         img_dir = data_path / split
#         if not img_dir.exists():
#             print(f"❌ Directory {img_dir} not found!")
#             continue
            
#         for img_path in img_dir.glob('*.tif'):
#             try:
#                 with rasterio.open(img_path, 'r+') as src:
#                     data = src.read()
#                     if np.isnan(data).any():
#                         print(f"   Fixing NaN in {img_path.name}")
#                         nan_files_fixed += 1
#                         for band_idx in range(data.shape[0]):
#                             band_data = data[band_idx]
#                             if np.isnan(band_data).any():
#                                 valid_pixels = band_data[~np.isnan(band_data)]
#                                 replacement_value = np.median(valid_pixels) if len(valid_pixels) > 0 else 0.0
#                                 band_data[np.isnan(band_data)] = replacement_value
#                         src.write(data)
#             except Exception as e:
#                 print(f"   ❌ Error fixing {img_path.name}: {e}")
    
#     print(f"✅ Fixed NaN values in {nan_files_fixed} files")
#     return nan_files_fixed > 0

# def calculate_class_weights(data_root, split_file='train.txt'):
#     """Calculate class weights for imbalanced dataset"""
#     import rasterio
#     from PIL import Image
    
#     print(" Calculating class weights...")
#     list_path = Path(data_root) / 'list' / split_file
#     with open(list_path, 'r') as f:
#         file_names = [line.strip() for line in f.readlines()]
    
#     total_pixels = 0
#     change_pixels = 0
#     label_dir = Path(data_root) / 'label'
    
#     for file_name in file_names:
#         base_name = file_name.replace('.tif', '')
#         label_paths = [label_dir / f"{base_name}.tif", label_dir / f"{base_name}.png"]
#         label_path = next((path for path in label_paths if path.exists()), None)
        
#         if label_path is None:
#             print(f"   Warning: Label not found for {file_name}")
#             continue
            
#         try:
#             if label_path.suffix == '.tif':
#                 with rasterio.open(label_path) as src:
#                     label_data = src.read(1)
#             else:
#                 label_data = np.array(Image.open(label_path))
#                 if len(label_data.shape) > 2:
#                     label_data = label_data[:, :, 0]
            
#             label_data = (label_data > 0).astype(np.uint8)
#             total_pixels += label_data.size
#             change_pixels += np.sum(label_data > 0)
                
#         except Exception as e:
#             print(f"   Error reading {label_path}: {e}")
    
#     if total_pixels > 0:
#         change_ratio = change_pixels / total_pixels
#         no_change_ratio = 1 - change_ratio
#         no_change_weight = 1.0
#         change_weight = no_change_ratio / change_ratio if change_ratio > 0 else 10.0
#         print(f"   Change ratio: {change_ratio:.4f}")
#         print(f"   Weights - No change: {no_change_weight:.2f}, Change: {change_weight:.2f}")
#         return [float(no_change_weight), float(change_weight)]
#     print("   ❌ Could not calculate class weights")
#     return [1.0, 10.0]

# def train(args):
#     """Simplified training function"""
#     if args.fix_nan_values:
#         fix_dataset_nan_values(args.data_root)
    
#     if args.auto_class_weights:
#         class_weights = calculate_class_weights(args.data_root)
#         args.class_weights = [float(w) for w in class_weights]
#         print(f"🎯 Using class weights: {args.class_weights}")
    
#     args.batch_size = int(args.batch_size)
#     args.num_workers = int(args.num_workers)
#     args.max_epochs = int(args.max_epochs)
#     args.img_size = int(args.img_size)
#     args.vis_freq = int(args.vis_freq)
#     args.save_epoch_freq = int(args.save_epoch_freq)
    
#     os.makedirs(args.checkpoint_dir, exist_ok=True)
    
#     visualizer = TrainingVisualizer(args.vis_dir, args.exp_name)
#     args.visualizer = visualizer
    
#     print(f"🔥 Starting training with {args.backbone} backbone")
#     dataloaders = utils.get_loaders(args)
    
#     args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
#     model = CDTrainer(args=args, dataloaders=dataloaders)
#     model.train_models()
    
#     print(" Creating final visualizations...")
#     visualizer.plot_training_curves()
#     visualizer.save_metrics_json()
    
#     return model

# def setup_device(args):
#     """Setup device configuration"""
#     if torch.cuda.is_available() and args.gpu_ids:
#         args.device = f'cuda:{args.gpu_ids[0]}'
#         print(f"🚀 Using GPU: {args.device}")
#     else:
#         args.device = 'cpu'
#         print("⚠️ Using CPU")
#     return args

# def validate_args(args):
#     """Validate arguments"""
#     data_path = Path(args.data_root)
#     if not data_path.exists():
#         print(f"❌ Data root directory does not exist: {args.data_root}")
#         return False
    
#     required_dirs = ['A', 'B', 'label', 'list']
#     missing_dirs = [req_dir for req_dir in required_dirs if not (data_path / req_dir).exists()]
#     if missing_dirs:
#         print(f"❌ Missing directories: {missing_dirs}")
#         return False
    
#     train_list = data_path / 'list' / 'train.txt'
#     if not train_list.exists():
#         print(f"❌ Training list not found: {train_list}")
#         return False
    
#     if len(args.rgb_bands) != 3:
#         print("⚠️ RGB bands should be 3 values. Using default")
#         args.rgb_bands = [3, 2, 1]
    
#     if args.focal_alpha < 0 or args.focal_alpha > 1:
#         print("⚠️ focal_alpha should be between 0 and 1. Setting to 0.75")
#         args.focal_alpha = 0.75
    
#     if len(args.class_weights) != 2:
#         print("⚠️ Class weights should have 2 values. Using default")
#         args.class_weights = [1.0, 10.0]
    
#     print("✅ Arguments validated")
#     return True

# if __name__ == '__main__':
#     parser = ArgumentParser(description="Simplified ChangeFormer Training")
    
#     parser.add_argument('--data_name', type=str, default='cartoCustom')
#     parser.add_argument('--data_root', type=str, default='./data/cartoCustom')
#     parser.add_argument('--data_format', type=str, default='tif')
#     parser.add_argument('--split', type=str, default='list')
#     parser.add_argument('--dataset', type=str, default='CDDataset')
#     parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_Simplified')
    
#     parser.add_argument('--satellite_type', type=str, default='cartosat', 
#                        choices=['cartosat', 'sentinel2', 'landsat8', 'planetscope'])
#     parser.add_argument('--rgb_bands', type=int, nargs=3, default=[3, 2, 1])
#     parser.add_argument('--normalize_method', type=str, default='adaptive_percentile')
    
#     parser.add_argument('--embed_dim', type=int, default=256)
#     parser.add_argument('--net_G', type=str, default='ChangeFormerV6')
#     parser.add_argument('--backbone', type=str, default='resnet18', 
#                        choices=['resnet18', 'resnet34', 'resnet50'])
#     parser.add_argument('--n_class', type=int, default=2)
    
#     parser.add_argument('--max_epochs', type=int, default=100)
#     parser.add_argument('--batch_size', type=int, default=4)
#     parser.add_argument('--lr', type=float, default=1e-5)
#     parser.add_argument('--img_size', type=int, default=256)
#     parser.add_argument('--num_workers', type=int, default=2)
#     parser.add_argument('--optimizer', type=str, default='adamw')
#     parser.add_argument('--lr_policy', type=str, default='cosine')
    
#     parser.add_argument('--loss', type=str, default='combined')
#     parser.add_argument('--auto_class_weights', type=bool, default=True)
#     parser.add_argument('--class_weights', type=float, nargs='+', default=[1.0, 15.0])
#     parser.add_argument('--focal_alpha', type=float, default=0.75)
#     parser.add_argument('--focal_gamma', type=float, default=2.0)
#     parser.add_argument('--dice_weight', type=float, default=0.6)
#     parser.add_argument('--focal_weight', type=float, default=0.4)
    
#     parser.add_argument('--use_simple_augmentation', type=bool, default=True)
#     parser.add_argument('--horizontal_flip_prob', type=float, default=0.5)
#     parser.add_argument('--vertical_flip_prob', type=float, default=0.5)
#     parser.add_argument('--rotation_prob', type=float, default=0.3)
#     parser.add_argument('--shuffle_AB', type=bool, default=True)
    
#     parser.add_argument('--min_component_size', type=int, default=15)
#     parser.add_argument('--morphology_kernel_size', type=int, default=3)
    
#     parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
#     parser.add_argument('--checkpoint_dir', type=str, 
#                        default='./checkpoints/ChangeFormer_myCustom/ChangeFormerV6_Simplified')
#     parser.add_argument('--vis_dir', type=str, default='./vis')
#     parser.add_argument('--vis_freq', type=int, default=10)
#     parser.add_argument('--save_epoch_freq', type=int, default=10)
    
#     parser.add_argument('--pretrain', type=bool, default=True)
#     parser.add_argument('--pretrain_path', type=str, 
#                        default=r'D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_LEVIR\ChangeFormerV6_LEVIR\LEVIR_WEIGHT\best_ckpt.pt')
    
#     parser.add_argument('--early_stopping_patience', type=int, default=20)
#     parser.add_argument('--weight_decay', type=float, default=1e-4)
#     parser.add_argument('--fix_nan_values', type=bool, default=True)
#     parser.add_argument('--val_freq', type=int, default=5)
#     parser.add_argument('--augment_factor', type=int, default=1)
#     parser.add_argument('--multi_scale_train', type=bool, default=False)
#     parser.add_argument('--multi_scale_infer', type=bool, default=False)
#     parser.add_argument('--multi_pred_weights', type=float, nargs='+', default=[1.0])

#     parser.add_argument('--gpu', type=str, default='0')
    
#     args = parser.parse_args()
    
#     args.gpu_ids = [int(i) for i in args.gpu.split(',')] if torch.cuda.is_available() else []
    
#     rgb_configs = {
#         'sentinel2': [3, 2, 1],
#         'landsat8': [3, 2, 1],
#         'planetscope': [2, 1, 0],
#         'cartosat': [3, 2, 1]
#     }
#     args.rgb_bands = rgb_configs.get(args.satellite_type, [3, 2, 1])
    
#     args = setup_device(args)
    
#     if not validate_args(args):
#         print("❌ Argument validation failed. Exiting.")
#         exit(1)
    
#     print("🚀 Simplified ChangeFormer Training")
#     print(f"Dataset: {args.data_name}")
#     print(f"Model: {args.net_G} with {args.backbone} backbone")
#     print(f"Loss: {args.loss}")
#     print(f"Device: {args.device}")
    
#     try:
#         train(args)
#         print("\n✅ Training completed!")
#         print(f" Visualizations saved in: {args.vis_dir}/{args.exp_name}")
#     except Exception as e:
#         print(f"\n❌ Training failed: {e}")
#         import traceback
#         traceback.print_exc()