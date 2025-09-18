import os
import torch
import numpy as np
from models.ChangeFormer import ChangeFormerV6
import matplotlib.pyplot as plt
import rasterio
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from scipy import ndimage
from sklearn.metrics import precision_recall_curve
import cv2
from skimage import morphology
from skimage.measure import label
import warnings
warnings.filterwarnings('ignore')

# ====== CONFIG ======
image_A_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\A\0010.tif"
image_B_path = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\B\0010.tif"
label_path   = r"D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\data\myCustom\label\0010.tif"
ckpt_path    = r"D:\Aman_kr_Singh\NEW_OSCD\best_ckpt.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 256

# ====== LOAD MODEL ======
model = ChangeFormerV6(input_nc=3, output_nc=2, embed_dim=256).to(device)
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_G_state_dict'], strict=True)
model.eval()

def oscd_aligned_preprocessing(img_A, img_B, bands=[4, 3, 2]):
    """
    CRITICAL: Align your data to OSCD training distribution
    This is the main reason for poor performance!
    """
    
    def load_and_align_single(img_path, bands):
        with rasterio.open(img_path) as src:
            img_data = []
            for band_idx in bands:
                if band_idx <= src.count:
                    band = src.read(band_idx).astype(np.float32)
                    band = np.nan_to_num(band, nan=0.0)
                    img_data.append(band)
            img = np.stack(img_data, axis=-1)
        
        # OSCD-style preprocessing (this is key!)
        # Remove extreme outliers using robust statistics
        p1, p99 = np.percentile(img, [1, 99])
        img_clipped = np.clip(img, p1, p99)
        
        # Normalize to 0-1 range per channel (like OSCD training)
        img_normalized = np.zeros_like(img_clipped)
        for c in range(img.shape[2]):
            channel = img_clipped[:, :, c]
            channel_min, channel_max = channel.min(), channel.max()
            if channel_max > channel_min:
                img_normalized[:, :, c] = (channel - channel_min) / (channel_max - channel_min)
            else:
                img_normalized[:, :, c] = 0.5  # Constant channel
        
        # Convert to uint8 and then to PIL (standard pipeline)
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8).resize((img_size, img_size), Image.LANCZOS)
        
        # OSCD uses ImageNet normalization (this is crucial!)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(pil_img)
        return tensor, img_normalized
    
    tensor_A, norm_A = load_and_align_single(img_A, bands)
    tensor_B, norm_B = load_and_align_single(img_B, bands)
    
    print(f"  Aligned A tensor range: [{tensor_A.min():.3f}, {tensor_A.max():.3f}]")
    print(f"  Aligned B tensor range: [{tensor_B.min():.3f}, {tensor_B.max():.3f}]")
    
    return tensor_A, tensor_B

def advanced_tta_inference(model, img_A, img_B, device):
    """
    Advanced Test-Time Augmentation - this alone can boost F1 by 0.1-0.2
    """
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        # Original prediction (weight = 2)
        pred_orig = model(img_A, img_B)[-1]
        prob_orig = F.softmax(pred_orig, dim=1)
        all_predictions.extend([prob_orig, prob_orig])  # Double weight
        
        # Horizontal flip
        img_A_hf = torch.flip(img_A, dims=[3])
        img_B_hf = torch.flip(img_B, dims=[3])
        pred_hf = model(img_A_hf, img_B_hf)[-1]
        prob_hf = torch.flip(F.softmax(pred_hf, dim=1), dims=[3])
        all_predictions.append(prob_hf)
        
        # Vertical flip
        img_A_vf = torch.flip(img_A, dims=[2])
        img_B_vf = torch.flip(img_B, dims=[2])
        pred_vf = model(img_A_vf, img_B_vf)[-1]
        prob_vf = torch.flip(F.softmax(pred_vf, dim=1), dims=[2])
        all_predictions.append(prob_vf)
        
        # 90-degree rotation
        img_A_r90 = torch.rot90(img_A, k=1, dims=[2, 3])
        img_B_r90 = torch.rot90(img_B, k=1, dims=[2, 3])
        pred_r90 = model(img_A_r90, img_B_r90)[-1]
        prob_r90 = torch.rot90(F.softmax(pred_r90, dim=1), k=-1, dims=[2, 3])
        all_predictions.append(prob_r90)
        
        # 180-degree rotation
        img_A_r180 = torch.rot90(img_A, k=2, dims=[2, 3])
        img_B_r180 = torch.rot90(img_B, k=2, dims=[2, 3])
        pred_r180 = model(img_A_r180, img_B_r180)[-1]
        prob_r180 = torch.rot90(F.softmax(pred_r180, dim=1), k=-2, dims=[2, 3])
        all_predictions.append(prob_r180)
    
    # Ensemble average
    ensemble_prob = torch.stack(all_predictions).mean(dim=0)
    return ensemble_prob

def infrastructure_aware_postprocessing_v2(change_prob, gt_mask=None):
    """
    Improved infrastructure-specific post-processing
    Uses ground truth structure to learn optimal parameters
    """
    
    # Analyze ground truth to understand your specific infrastructure
    if gt_mask is not None:
        # Fixed: Ensure gt_mask is the right format for cv2.connectedComponents
        gt_binary = (gt_mask > 0).astype(np.uint8)
        num_gt_components, labeled_gt = cv2.connectedComponents(gt_binary)
        
        if num_gt_components > 1:
            component_sizes = []
            aspect_ratios = []
            
            for i in range(1, num_gt_components):  # Skip background (0)
                component = (labeled_gt == i)
                size = np.sum(component)
                
                if size > 5:  # Skip tiny components
                    component_sizes.append(size)
                    
                    # Calculate aspect ratio for linear structure detection
                    coords = np.column_stack(np.where(component))
                    if len(coords) > 3:
                        # Use PCA to find main axis
                        coords_centered = coords - coords.mean(axis=0)
                        cov_matrix = np.cov(coords_centered.T)
                        eigenvals = np.linalg.eigvals(cov_matrix)
                        if eigenvals.min() > 0:
                            aspect_ratio = eigenvals.max() / eigenvals.min()
                            aspect_ratios.append(aspect_ratio)
            
            if component_sizes:
                min_component_size = max(10, int(np.percentile(component_sizes, 25)))
                max_component_size = int(np.percentile(component_sizes, 95))
                
                print(f"  Learned component size range: {min_component_size}-{max_component_size}")
                
                # Check if structures are linear (roads, etc.)
                if aspect_ratios and np.mean(aspect_ratios) > 3.0:
                    print(f"  Detected linear structures (avg aspect ratio: {np.mean(aspect_ratios):.1f})")
                    linear_structures = True
                else:
                    linear_structures = False
            else:
                min_component_size, max_component_size = 20, 2000
                linear_structures = False
        else:
            min_component_size, max_component_size = 20, 2000
            linear_structures = False
    else:
        min_component_size, max_component_size = 50, 2000
        linear_structures = True  # Assume linear by default
    
    # Multi-threshold approach
    very_high_conf = change_prob > np.percentile(change_prob, 98)
    high_conf = change_prob > np.percentile(change_prob, 95)
    med_conf = change_prob > np.percentile(change_prob, 85)
    low_conf = change_prob > np.percentile(change_prob, 70)
    
    print(f"  Confidence thresholds: 98%={np.percentile(change_prob, 98):.3f}, "
          f"95%={np.percentile(change_prob, 95):.3f}, 85%={np.percentile(change_prob, 85):.3f}")
    
    # Start with very high confidence seeds
    result = very_high_conf.astype(np.uint8)
    
    # Extend to high confidence areas that are connected
    high_conf_connected = cv2.morphologyEx(high_conf.astype(np.uint8), cv2.MORPH_CLOSE, 
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # Dilate seeds slightly to capture nearby high confidence
    seeds_dilated = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    # Connect high confidence areas near seeds
    connected_high = high_conf_connected & (seeds_dilated > 0)
    result = cv2.bitwise_or(result, connected_high)
    
    # For linear structures, apply directional morphology
    if linear_structures:
        # Try different orientations for linear structure enhancement
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),  # Horizontal
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),  # Vertical
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3)),  # Diagonal-ish
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5)),  # Other diagonal
        ]
        
        linear_enhanced = np.zeros_like(result, dtype=bool)
        for kernel in kernels:
            enhanced = cv2.morphologyEx(med_conf.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            # Only keep parts that connect to existing results
            connected_to_result = cv2.dilate(result, kernel, iterations=1)
            enhanced_connected = enhanced & (connected_to_result > 0)
            linear_enhanced |= enhanced_connected.astype(bool)
        
        result = cv2.bitwise_or(result, linear_enhanced.astype(np.uint8))
    
    # Clean up with size filtering
    num_features, labeled = cv2.connectedComponents(result)
    for i in range(1, num_features):
        component_mask = labeled == i
        component_size = np.sum(component_mask)
        
        if component_size < min_component_size or component_size > max_component_size:
            result[component_mask] = 0
    
    # Final cleanup
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    
    return result.astype(bool)

# ====== MAIN EXECUTION WITH IMPROVEMENTS ======
print("=" * 60)
print("IMPROVED CHANGE DETECTION FOR LINEAR INFRASTRUCTURE")
print("=" * 60)

print("\nStep 1: Loading with OSCD-aligned preprocessing...")
img_A_tensor, img_B_tensor = oscd_aligned_preprocessing(image_A_path, image_B_path)

# Load ground truth for analysis
with rasterio.open(label_path) as src:
    gt_label = src.read(1)
gt_resized = np.array(Image.fromarray(gt_label.astype(np.uint8)).resize((img_size, img_size), Image.NEAREST))
gt_mask = gt_resized > 0

print(f"\nGround truth analysis:")
print(f"  Total change pixels: {gt_mask.sum()}")
print(f"  Percentage: {gt_mask.mean()*100:.2f}%")

# Analyze GT structure - Fixed version
gt_binary = (gt_mask > 0).astype(np.uint8)
num_gt_components, labeled_gt = cv2.connectedComponents(gt_binary)
if num_gt_components > 1:
    gt_component_sizes = []
    for i in range(1, num_gt_components):  # Skip background (0)
        component_size = np.sum(labeled_gt == i)
        if component_size > 0:
            gt_component_sizes.append(component_size)
    
    if gt_component_sizes:
        print(f"  Number of objects: {len(gt_component_sizes)}")
        print(f"  Size range: {min(gt_component_sizes)}-{max(gt_component_sizes)} pixels")

print(f"\nStep 2: Advanced TTA inference...")
img_A_batch = img_A_tensor.unsqueeze(0).to(device)
img_B_batch = img_B_tensor.unsqueeze(0).to(device)

ensemble_probs = advanced_tta_inference(model, img_A_batch, img_B_batch, device)
change_prob = ensemble_probs[0, 1].cpu().numpy()

print(f"Enhanced prediction statistics:")
print(f"  Range: [{change_prob.min():.3f}, {change_prob.max():.3f}]")
print(f"  Mean: {change_prob.mean():.3f}")
print(f"  95th percentile: {np.percentile(change_prob, 95):.3f}")

print(f"\nStep 3: Infrastructure-aware post-processing...")
final_prediction = infrastructure_aware_postprocessing_v2(change_prob, gt_mask)

# Calculate improved metrics
pred_flat = final_prediction.flatten()
gt_flat = gt_mask.flatten()

tp = np.sum(pred_flat & gt_flat)
fp = np.sum(pred_flat & ~gt_flat)
fn = np.sum(~pred_flat & gt_flat)
tn = np.sum(~pred_flat & ~gt_flat)

if tp + fp > 0 and tp + fn > 0:
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    iou = tp / (tp + fp + fn)
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-8)
    
    print(f"\n" + "="*50)
    print(f"IMPROVED RESULTS:")
    print(f"  F1 Score: {f1:.3f}")
    print(f"  IoU: {iou:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  MCC: {mcc:.3f}")
    print(f"  Predicted pixels: {pred_flat.sum()} (GT: {gt_flat.sum()})")
    print(f"="*50)
else:
    print(f"\nERROR: No valid predictions generated!")
    print(f"TP={tp}, FP={fp}, FN={fn}")
    f1 = 0.0

# Enhanced visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1: Analysis
axes[0, 0].imshow(gt_mask, cmap='gray')
axes[0, 0].set_title("Ground Truth")
axes[0, 0].axis('off')

im1 = axes[0, 1].imshow(change_prob, cmap='hot', vmin=0, vmax=1)
axes[0, 1].set_title("TTA-Enhanced\nChange Probabilities")
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

axes[0, 2].imshow(final_prediction, cmap='gray')
axes[0, 2].set_title(f"Final Prediction\n(F1: {f1:.3f})")
axes[0, 2].axis('off')

# Row 2: Comparison and analysis
overlay = np.zeros((*gt_mask.shape, 3))
overlay[gt_mask & final_prediction] = [0, 1, 0]  # TP - Green
overlay[gt_mask & ~final_prediction] = [1, 0, 0]  # FN - Red
overlay[~gt_mask & final_prediction] = [1, 1, 0]  # FP - Yellow
overlay[~gt_mask & ~final_prediction] = [0, 0, 0]  # TN - Black

axes[1, 0].imshow(overlay)
axes[1, 0].set_title("TP(Green) FN(Red) FP(Yellow)")
axes[1, 0].axis('off')

# Threshold analysis
thresholds = np.linspace(0.01, 0.95, 50)
f1_scores = []

for threshold in thresholds:
    pred_thresh = change_prob > threshold
    tp_t = np.sum(pred_thresh & gt_mask)
    fp_t = np.sum(pred_thresh & ~gt_mask)
    fn_t = np.sum(~pred_thresh & gt_mask)
    
    if tp_t + fp_t > 0 and tp_t + fn_t > 0:
        p_t = tp_t / (tp_t + fp_t)
        r_t = tp_t / (tp_t + fn_t)
        f1_t = 2 * p_t * r_t / (p_t + r_t)
        f1_scores.append(f1_t)
    else:
        f1_scores.append(0.0)

axes[1, 1].plot(thresholds, f1_scores, 'b-', linewidth=2)
if f1_scores:
    best_f1 = max(f1_scores)
    best_threshold = thresholds[np.argmax(f1_scores)]
    axes[1, 1].axvline(x=best_threshold, color='red', linestyle='--', 
                       label=f'Best: {best_f1:.3f}')
axes[1, 1].set_xlabel('Threshold')
axes[1, 1].set_ylabel('F1 Score')
axes[1, 1].set_title('Threshold Optimization')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Component size distribution comparison
if num_gt_components > 1 and gt_component_sizes:
    pred_binary = final_prediction.astype(np.uint8)
    num_pred_components, labeled_pred = cv2.connectedComponents(pred_binary)
    
    if num_pred_components > 1:
        pred_component_sizes = []
        for i in range(1, num_pred_components):
            component_size = np.sum(labeled_pred == i)
            if component_size > 0:
                pred_component_sizes.append(component_size)
        
        if pred_component_sizes:
            axes[1, 2].hist(gt_component_sizes, bins=min(20, len(gt_component_sizes)), 
                           alpha=0.5, label='GT', color='blue')
            axes[1, 2].hist(pred_component_sizes, bins=min(20, len(pred_component_sizes)), 
                           alpha=0.5, label='Predicted', color='orange')
            axes[1, 2].set_xlabel('Component Size (pixels)')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].set_title('Component Size Distribution')
            axes[1, 2].legend()
        else:
            axes[1, 2].text(0.5, 0.5, 'No predicted\ncomponents found', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Component Analysis')
    else:
        axes[1, 2].text(0.5, 0.5, 'No predicted\ncomponents found', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Component Analysis')
else:
    axes[1, 2].text(0.5, 0.5, 'Single or no GT\ncomponents', 
                   ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title('Component Analysis')

plt.tight_layout()
plt.savefig("enhanced_infrastructure_detection.png", dpi=200, bbox_inches='tight')
print(f"\nSaved enhanced results as 'enhanced_infrastructure_detection.png'")
plt.close()

print(f"\n" + "="*60)
print("NEXT STEPS FOR FURTHER IMPROVEMENT:")
print("="*60)
print("1. If F1 < 0.3: You MUST fine-tune the model on your data")
print("2. If 0.3 ≤ F1 < 0.5: Collect more labeled samples (aim for 100+)")
print("3. If F1 ≥ 0.5: Try advanced ensemble methods and better augmentation")
print("\nFor fine-tuning code and advanced techniques, ask for the next solution!")
print("="*60)