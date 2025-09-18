import os
import torch
import numpy as np
from argparse import ArgumentParser
from models.trainer import CDTrainer
import utils
from pathlib import Path
import shutil
from sklearn.model_selection import StratifiedKFold
import json
import matplotlib.pyplot as plt

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
        change_weight = no_change_ratio / change_ratio if change_ratio > 0 else 50.0
        
        print(f"   Change ratio: {change_ratio:.4f} ({change_ratio*100:.2f}%)")
        print(f"   Weights - No change: {no_change_weight:.2f}, Change: {change_weight:.2f}")
        
        # Convert to Python float to avoid np.float64
        return [float(no_change_weight), float(change_weight)]
    else:
        print("   ❌ Could not calculate class weights")
        return [1.0, 20.0]

def create_cross_validation_splits(data_root, n_splits=5, random_state=42):
    """Create stratified cross-validation splits"""
    import rasterio
    from PIL import Image
    
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
    
    fold_info = []
    
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
        
        fold_info.append({
            'fold': fold,
            'train_samples': len(train_samples),
            'train_changes': int(train_changes),
            'val_samples': len(val_samples),
            'val_changes': int(val_changes)
        })
        
        print(f"Fold {fold}: Train {len(train_samples)} samples ({train_changes} changes), "
              f"Val {len(val_samples)} samples ({val_changes} changes)")
    
    # Save fold information
    with open(cv_dir / 'fold_info.json', 'w') as f:
        json.dump(fold_info, f, indent=2)
    
    print(f"✅ Cross-validation splits created in {cv_dir}")
    return True

def train_single_fold(args, fold_num=None, original_data_root=None):
    """Train a single fold"""
    
    if fold_num is not None and original_data_root is not None:
        args.data_root = str(Path(original_data_root) / 'cv_splits' / f'fold_{fold_num}')
        base_exp_name = args.exp_name.split('_fold')[0]
        args.exp_name = f"{base_exp_name}_fold{fold_num}"
        args.checkpoint_dir = str(Path(args.checkpoint_root) / args.data_name / args.exp_name)
        
        print(f"🔥 Training fold {fold_num}")
        print(f"   Data root: {args.data_root}")
        print(f"   Checkpoint dir: {args.checkpoint_dir}")
    
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
    
    print(f"🔥 Using HEAVY AUGMENTATION: {args.augment_factor}x dataset expansion")
    
    dataloaders = utils.get_loaders(args)
    
    # Convert class weights to tensor
    args.class_weights = torch.tensor(args.class_weights, dtype=torch.float32).to(args.device)
    
    model = CDTrainer(args=args, dataloaders=dataloaders)
    
    # Train the model
    model.train_models()
    
    return model

def train_cross_validation(args):
    """Train with cross-validation"""
    
    # Store original data root to prevent path accumulation
    original_data_root = args.data_root
    original_exp_name = args.exp_name
    
    # Create cross-validation splits if they don't exist
    cv_dir = Path(original_data_root) / 'cv_splits'
    if not cv_dir.exists():
        print("Creating cross-validation splits...")
        create_cross_validation_splits(original_data_root, n_splits=args.cv_folds)
    
    # Load fold information
    fold_info_path = cv_dir / 'fold_info.json'
    if fold_info_path.exists():
        with open(fold_info_path, 'r') as f:
            fold_info = json.load(f)
    else:
        fold_info = []
    
    # Train each fold
    fold_results = []
    
    for fold in range(args.cv_folds):
        print(f"\n{'='*60}")
        print(f"🚀 TRAINING FOLD {fold + 1}/{args.cv_folds}")
        print(f"{'='*60}")
        
        try:
            # Reset args to original values before each fold
            args.data_root = original_data_root
            args.exp_name = original_exp_name
            
            # Train this fold with original data root
            model = train_single_fold(args, fold_num=fold, original_data_root=original_data_root)
            
            # TODO: Extract final metrics from model
            # This would need to be implemented based on your CDTrainer class
            fold_results.append({
                'fold': fold,
                'status': 'completed',
                # Add metrics here when available
            })
            
        except Exception as e:
            print(f"❌ Error training fold {fold}: {e}")
            fold_results.append({
                'fold': fold,
                'status': 'failed',
                'error': str(e)
            })
    
    # Save cross-validation results
    cv_results_path = Path(args.checkpoint_root) / args.data_name / f"{original_exp_name}_cv_results.json"
    cv_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cv_results_path, 'w') as f:
        json.dump({
            'fold_info': fold_info,
            'fold_results': fold_results,
            'args': vars(args)
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🎯 CROSS-VALIDATION COMPLETED")
    print(f"Results saved to: {cv_results_path}")
    print(f"{'='*60}")
    
    # Print summary
    completed_folds = [r for r in fold_results if r['status'] == 'completed']
    failed_folds = [r for r in fold_results if r['status'] == 'failed']
    
    print(f"✅ Completed folds: {len(completed_folds)}/{args.cv_folds}")
    if failed_folds:
        print(f"❌ Failed folds: {len(failed_folds)}")
        for failed in failed_folds:
            print(f"   Fold {failed['fold']}: {failed['error']}")

def train(args):
    """Main training function"""
    
    if args.cross_validation:
        train_cross_validation(args)
    else:
        train_single_fold(args)

def setup_device(args):
    """Setup device and GPU configuration"""
    if torch.cuda.is_available() and args.gpu_ids:
        args.device = f'cuda:{args.gpu_ids[0]}'
        print(f"🚀 Using GPU: {args.device}")
        print(f"   Available GPUs: {args.gpu_ids}")
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
        args.class_weights = [1.0, 20.0]
    
    print("✅ Arguments validated successfully")
    return True

if __name__ == '__main__':
    parser = ArgumentParser(description="Train ChangeFormer with HEAVY AUGMENTATION and Cross-Validation")

    # Data and experiment settings - UPDATED FOR HEAVY AUGMENTATION
    parser.add_argument('--data_name', type=str, default='myCustom')
    parser.add_argument('--data_root', type=str, default='./data/myCustom', 
                       help='Root directory containing A/, B/, label/, list/ folders')
    parser.add_argument('--data_format', type=str, default='tif', 
                       help='Data format: tif or standard')
    parser.add_argument('--split', type=str, default='list', help='Folder containing train.txt/val.txt')
    parser.add_argument('--dataset', type=str, default='CDDatasetHeavy', help='Dataset class name - USE HEAVY VERSION!')
    parser.add_argument('--exp_name', type=str, default='ChangeFormerV6_HeavyAug_75to7500', help='Experiment name')

    # HEAVY AUGMENTATION SETTINGS - NEW!
    parser.add_argument('--augment_factor', type=int, default=100, 
                       help='Augmentation factor: 75 samples -> 7500 samples with 100x')
    parser.add_argument('--cross_validation', type=bool, default=True, 
                       help='Use cross-validation training')
    parser.add_argument('--cv_folds', type=int, default=3, help='Number of CV folds')

    # TIF-specific parameters
    parser.add_argument('--satellite_type', type=str, default='sentinel2', 
                       choices=['sentinel2', 'landsat8', 'planetscope', 'false_color', 'vegetation', 'default'],
                       help='Satellite type for RGB band selection')
    parser.add_argument('--rgb_bands', type=int, nargs=3, default=[3, 2, 1],
                       help='RGB band indices (0-based) to extract from 10-band data')
    parser.add_argument('--normalize_method', type=str, default='adaptive_percentile',
                       choices=['percentile', 'zscore', 'minmax', 'adaptive_percentile'],
                       help='Normalization method for TIF data')

    # Model config - LIGHTER MODEL FOR SMALL DATASET
    parser.add_argument('--embed_dim', type=int, default=128, help='Smaller embedding for small dataset')
    parser.add_argument('--net_G', type=str, default='ChangeFormerV6', help='Network architecture')
    parser.add_argument('--backbone', type=str, default='resnet18', help='Lighter backbone')
    parser.add_argument('--n_class', type=int, default=2)

    # Training params - OPTIMIZED FOR SMALL HEAVILY AUGMENTED DATASET
    parser.add_argument('--max_epochs', type=int, default=50, help='More epochs for augmented data')
    parser.add_argument('--batch_size', type=int, default=2, help='Small batch for stability') 
    parser.add_argument('--lr', type=float, default=1e-5, help='Very low LR for fine-tuning')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=0, help='Fewer workers')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr_policy', type=str, default='cosine', 
                       choices=['linear', 'step', 'plateau', 'cosine'])
    
    # IMPROVED: Class imbalance handling - STRONGER
    parser.add_argument('--loss', type=str, default='focal', 
                       choices=['ce', 'bce', 'fl', 'focal', 'weighted_bce', 'dice', 'combined'])
    parser.add_argument('--auto_class_weights', type=bool, default=True)
    parser.add_argument('--class_weights', type=float, nargs='+', default=[1.0, 20.0], 
                       help='Higher change weight for very imbalanced data')
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.5, help='Higher gamma for hard examples')
    
    # Data augmentation - ALREADY HANDLED BY HEAVY DATASET
    parser.add_argument('--multi_scale_infer', type=bool, default=True)
    parser.add_argument('--multi_pred_weights', type=float, nargs='+', default=[1.0, 0.8, 0.6])
    parser.add_argument('--shuffle_AB', type=bool, default=True)
    parser.add_argument('--multi_scale_train', type=bool, default=True)

    # Checkpointing and visualization
    parser.add_argument('--checkpoint_root', type=str, default='./checkpoints/')
    parser.add_argument('--checkpoint_dir', type=str, 
                       default='./checkpoints/ChangeFormer_myCustom/ChangeFormerV6_HeavyAug_75to7500')
    parser.add_argument('--vis_dir', type=str, default='./vis')
    parser.add_argument('--vis_freq', type=int, default=50)
    parser.add_argument('--save_vis_during_train', type=bool, default=True)

    # Pretraining - CRITICAL: USE LEVIR/DSIFN WEIGHTS, NOT OSCD!
    parser.add_argument('--pretrain', type=bool, default=True)
    parser.add_argument('--pretrain_path', type=str, 
                       default=r'D:\Aman_kr_Singh\NEW_OSCD\ChangeFormer\checkpoints\ChangeFormer_LEVIR\ChangeFormerV6_LEVIR\LEVIR_WEIGHT\best_ckpt.pt',
                       help='IMPORTANT: Use LEVIR/DSIFN weights, NOT OSCD weights!')
    parser.add_argument('--freeze_backbone_epochs', type=int, default=50, 
                       help='Freeze backbone longer for better fine-tuning')

    # Training strategy for small augmented dataset
    parser.add_argument('--early_stopping_patience', type=int, default=30, help='More patience for CV')
    parser.add_argument('--save_epoch_freq', type=int, default=20)
    parser.add_argument('--gradient_clip_val', type=float, default=0.5, help='Aggressive gradient clipping')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='Longer warmup')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Regularization')
    
    # Dataset preprocessing
    parser.add_argument('--fix_nan_values', type=bool, default=True)
    parser.add_argument('--label_transform', type=str, default='binary', 
                       choices=['none', 'binary', 'norm'])

    # Validation and testing
    parser.add_argument('--val_freq', type=int, default=10, help='Less frequent validation')
    parser.add_argument('--test_during_train', type=bool, default=False)

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
            'vegetation': [7, 5, 3],    # Vegetation-focused bands
        }
        args.rgb_bands = rgb_configs.get(args.satellite_type, [3, 2, 1])

    # Setup device
    args = setup_device(args)

    # Validate arguments
    if not validate_args(args):
        print("❌ Argument validation failed. Exiting.")
        exit(1)

    # Print configuration summary
    print("🚀 HEAVY AUGMENTATION + CROSS-VALIDATION TRAINING")
    print("=" * 60)
    print(f"Dataset: {args.data_name}")
    print(f"Data root: {args.data_root}")
    print(f"Data format: {args.data_format}")
    print(f"Satellite type: {args.satellite_type}")
    print(f"RGB bands: {args.rgb_bands}")
    print(f"Original dataset size: ~75 samples")
    print(f"Augmented dataset size: ~{75 * args.augment_factor} samples (x{args.augment_factor})")
    print(f"Cross-validation: {args.cross_validation} ({args.cv_folds} folds)")
    print(f"Model: {args.net_G} with {args.backbone} backbone")
    print(f"Loss function: {args.loss}")
    print(f"Class weights: {args.class_weights}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Device: {args.device}")
    print("=" * 60)

    # Start training
    try:
        train(args)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🏁 Training script completed")