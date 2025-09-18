import numpy as np
import matplotlib.pyplot as plt
import os

import utils
from models.networks import *

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import cross_entropy
import models.losses as losses
from models.losses import get_alpha, softmax_helper, FocalLoss, mIoULoss, mmIoULoss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import torch.nn.functional as F

from misc.logger_tool import Logger, Timer

from utils import de_norm

from tqdm import tqdm


# ===================== ENHANCED LOSS FUNCTIONS =====================

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, num_classes=2):
        """
        Args:
            alpha (float or list): Weighting factor for classes.
            gamma (float): Focusing parameter.
            num_classes (int): Number of classes.
        """
        super(EnhancedFocalLoss, self).__init__()

        # ✅ Force int for safety
        num_classes = int(num_classes)

        # ✅ Store gamma as float
        self.gamma = float(gamma)
        self.num_classes = num_classes

        # ✅ Prepare alpha safely
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(
                [float(a) for a in alpha],
                dtype=torch.float32
            )
        else:
            self.alpha = torch.tensor(
                [float(alpha)] * num_classes,
                dtype=torch.float32
            )

    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits (N, C, H, W)
            targets: ground truth labels (N, H, W) or one-hot (N, C, H, W)
        """
        if inputs.dim() > 2:
            # Flatten to (N*H*W, C)
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
            inputs = inputs.transpose(1, 2)
            inputs = inputs.contiguous().view(-1, inputs.size(2))

        targets = targets.view(-1, 1)
        
        # CRITICAL FIX: Convert targets to long dtype BEFORE any operations
        targets = targets.long()

        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        # ✅ Move alpha to the same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # CRITICAL FIX: Use long targets for indexing
        alpha_t = self.alpha[targets.view(-1)]
        loss = -alpha_t * (1 - pt) ** self.gamma * logpt

        return loss.mean()


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross Entropy Loss
    """
    def __init__(self, pos_weight=None, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.float()
        
        if inputs.dim() > 1 and inputs.size(1) == 2:
            # Convert to binary logits
            inputs = inputs[:, 1] - inputs[:, 0]
        
        return F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )


class DiceLoss(nn.Module):
    """
    Dice Loss for better boundary detection
    """
    def __init__(self, smooth=1.0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert to probabilities
        if inputs.dim() > 1 and inputs.size(1) == 2:
            inputs = F.softmax(inputs, dim=1)[:, 1]  # Take positive class
        else:
            inputs = torch.sigmoid(inputs)
        
        # CRITICAL FIX: Convert targets to float (not long for Dice)
        targets = targets.float()
        
        # Flatten tensors
        inputs_flat = inputs.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # Calculate intersection and union
        intersection = (inputs_flat * targets_flat).sum()
        total = inputs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        dice_loss = 1 - dice
        
        return dice_loss

class CombinedLoss(nn.Module):
    """
    Combined loss function using multiple losses
    """
    def __init__(self, focal_weight=0.6, dice_weight=0.4, 
                 focal_alpha=0.75, focal_gamma=2.0, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
        self.focal_loss = EnhancedFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        # CRITICAL FIX: Ensure targets are long dtype at the beginning
        targets = targets.long()
        
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        
        combined = self.focal_weight * focal + self.dice_weight * dice
        return combined


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    Good for imbalanced data by controlling false positives and false negatives
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Convert to probabilities
        if inputs.dim() > 1 and inputs.size(1) == 2:
            inputs = F.softmax(inputs, dim=1)[:, 1]
        else:
            inputs = torch.sigmoid(inputs)
            
        targets = targets.float()
        
        # Flatten tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax Loss for semantic segmentation
    """
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if inputs.dim() > 1 and inputs.size(1) == 2:
            inputs = F.softmax(inputs, dim=1)
        else:
            # Convert binary to 2-class
            probs_0 = 1 - torch.sigmoid(inputs)
            probs_1 = torch.sigmoid(inputs)
            inputs = torch.stack([probs_0, probs_1], dim=1)
        
        return self.lovasz_softmax(inputs, targets.long(), 
                                 classes=self.classes, 
                                 per_image=self.per_image, 
                                 ignore=self.ignore_index)
    
    def lovasz_softmax(self, probas, labels, classes='present', per_image=False, ignore=None):
        """
        Multi-class Lovász-Softmax loss
        """
        if per_image:
            loss = self._lovasz_softmax_flat(*self._flatten_probas(probas[i:i+1], labels[i:i+1], ignore), classes=classes)
            for i in range(1, probas.size(0)):
                loss += self._lovasz_softmax_flat(*self._flatten_probas(probas[i:i+1], labels[i:i+1], ignore), classes=classes)
            return loss / probas.size(0)
        else:
            return self._lovasz_softmax_flat(*self._flatten_probas(probas, labels, ignore), classes=classes)
    
    def _lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Lovász-Softmax loss for flattened predictions and labels
        """
        if probas.numel() == 0:
            return probas * 0.
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, self._lovasz_grad(fg_sorted)))
        return sum(losses) / len(losses)
    
    def _flatten_probas(self, probas, labels, ignore=None):
        """
        Flatten predictions and labels
        """
        if probas.dim() == 3:
            # Assume NxHxW
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)
        B, C, H, W = probas.size()
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)
        vprobas = probas[valid.nonzero().squeeze()]
        vlabels = labels[valid]
        return vprobas, vlabels
    
    def _lovasz_grad(self, gt_sorted):
        """
        Compute gradient of Lovász extension w.r.t sorted errors
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if gts == 0:
            return gt_sorted.new_zeros(p)
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard


def get_enhanced_loss_function(loss_type, class_weights=None, **kwargs):
    """
    Factory function to create enhanced loss functions
    
    Args:
        loss_type: 'enhanced_focal', 'weighted_bce', 'dice', 'combined', 'tversky', 'lovasz'
        class_weights: [no_change_weight, change_weight]
        **kwargs: Additional arguments for specific losses
    """
    
    if loss_type == 'enhanced_focal':
        alpha = kwargs.get('focal_alpha', 0.75)
        gamma = kwargs.get('focal_gamma', 2.0)
        return EnhancedFocalLoss(alpha=alpha, gamma=gamma)
    
    elif loss_type == 'weighted_bce':
        pos_weight = None
        if class_weights:
            pos_weight = torch.tensor(
                [float(class_weights[1]) / float(class_weights[0])],
                dtype=torch.float32
            )
        return WeightedBCELoss(pos_weight=pos_weight)
    
    elif loss_type == 'dice':
        smooth = kwargs.get('smooth', 1.0)
        return DiceLoss(smooth=smooth)
    
    elif loss_type == 'combined':
        focal_weight = kwargs.get('focal_weight', 0.6)
        dice_weight = kwargs.get('dice_weight', 0.4)
        focal_alpha = kwargs.get('focal_alpha', 0.75)
        focal_gamma = kwargs.get('focal_gamma', 2.0)
        
        pos_weight = None
        if class_weights:
            pos_weight = torch.tensor(
                [float(class_weights[1]) / float(class_weights[0])],
                dtype=torch.float32
            )
            
        return CombinedLoss(
            focal_weight=focal_weight,
            dice_weight=dice_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            pos_weight=pos_weight
        )
    
    elif loss_type == 'tversky':
        alpha = kwargs.get('tversky_alpha', 0.3)
        beta = kwargs.get('tversky_beta', 0.7)
        return TverskyLoss(alpha=alpha, beta=beta)
    
    elif loss_type == 'lovasz':
        return LovaszSoftmaxLoss()
    
    else:
        # Default to weighted BCE
        pos_weight = None
        if class_weights:
            pos_weight = torch.tensor(
                [float(class_weights[1]) / float(class_weights[0])],
                dtype=torch.float32
            )
        return WeightedBCELoss(pos_weight=pos_weight)


class LossScheduler:
    """
    Dynamic loss weight scheduling during training
    """
    def __init__(self, initial_weights, schedule_type='linear', total_epochs=200):
        self.initial_weights = initial_weights
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        
    def get_weights(self, epoch):
        if self.schedule_type == 'linear':
            # Gradually reduce the class weight imbalance
            progress = epoch / self.total_epochs
            decay_factor = 1.0 - 0.5 * progress  # Reduce to 50% of initial weight
            
            return [1.0, self.initial_weights[1] * decay_factor]
        
        elif self.schedule_type == 'cosine':
            # Cosine annealing
            progress = epoch / self.total_epochs
            decay_factor = 0.5 * (1 + np.cos(np.pi * progress))
            
            return [1.0, self.initial_weights[1] * decay_factor]
        
        else:
            return self.initial_weights


# ===================== TRAINING CODE =====================

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler"""
    if args.lr_policy == 'cosine':
        # Use a less aggressive cosine schedule or step scheduler
        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)  # Reduce LR by half every 50 epochs
    elif args.lr_policy == 'step':
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        # Keep constant LR
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
    return scheduler


class CDTrainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders
        self.visualizer = getattr(args, 'visualizer', None)

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                     weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self.shuffle_AB = args.shuffle_AB


        self.last_train_f1 = 0.0
        self.last_val_f1 = 0.0

        # define the loss functions
        self.multi_scale_train = args.multi_scale_train
        self.multi_scale_infer = args.multi_scale_infer
        self.weights = tuple(args.multi_pred_weights)

        # Initialize loss scheduler if specified
        self.loss_scheduler = None
        if hasattr(args, 'use_loss_scheduler') and args.use_loss_scheduler:
            initial_weights = [1.0, 10.0]  # Default weights
            if hasattr(args, 'initial_class_weights'):
                initial_weights = args.initial_class_weights
            
            schedule_type = getattr(args, 'loss_schedule_type', 'linear')
            self.loss_scheduler = LossScheduler(
                initial_weights=initial_weights,
                schedule_type=schedule_type,
                total_epochs=self.max_num_epochs
            )

        # Enhanced loss function selection
        if args.loss == 'ce':
            self._pxl_loss = cross_entropy
        elif args.loss == 'bce':
            self._pxl_loss = losses.binary_ce
        elif args.loss == 'fl' or args.loss == 'focal':
            print('\n Calculating alpha in Focal-Loss (FL) ...')
            alpha = get_alpha(dataloaders['train'])
            print(f"alpha-0 (no-change)={alpha[0]}, alpha-1 (change)={alpha[1]}")
            # Increase gamma for stronger focus on hard examples
            self._pxl_loss = FocalLoss(apply_nonlin=softmax_helper, alpha=alpha, gamma=3, smooth=1e-5)
        elif args.loss == 'enhanced_focal':
            print('\n Using Enhanced Focal Loss...')
            alpha = get_alpha(dataloaders['train']) if hasattr(args, 'auto_alpha') and args.auto_alpha else [0.25, 0.75]
            gamma = getattr(args, 'focal_gamma', 2.0)
            print(f"Enhanced Focal Loss - alpha: {alpha[1]}, gamma: {gamma}")
            self._pxl_loss = get_enhanced_loss_function(
                'enhanced_focal',
                focal_alpha=alpha[1],
                focal_gamma=gamma
            )
        elif args.loss == 'weighted_bce':
            print('\n Using Enhanced Weighted BCE Loss...')
            alpha = get_alpha(dataloaders['train'])
            class_weights = [alpha[0], alpha[1]]
            print(f"Class weights: {class_weights}")
            self._pxl_loss = get_enhanced_loss_function(
                'weighted_bce',
                class_weights=class_weights
            )
        elif args.loss == 'dice':
            print('\n Using Dice Loss...')
            smooth = getattr(args, 'dice_smooth', 1.0)
            self._pxl_loss = get_enhanced_loss_function('dice', smooth=smooth)
        elif args.loss == 'combined':
            print('\n Using Combined Loss (Focal + Dice)...')
            alpha = get_alpha(dataloaders['train']) if hasattr(args, 'auto_alpha') and args.auto_alpha else [0.25, 0.75]
            class_weights = [alpha[0], alpha[1]]
            focal_weight = getattr(args, 'focal_weight', 0.6)
            dice_weight = getattr(args, 'dice_weight', 0.4)
            focal_gamma = getattr(args, 'focal_gamma', 2.0)
            print(f"Combined Loss - Focal weight: {focal_weight}, Dice weight: {dice_weight}")
            print(f"Focal alpha: {alpha[1]}, gamma: {focal_gamma}")
            self._pxl_loss = get_enhanced_loss_function(
                'combined',
                class_weights=class_weights,
                focal_weight=focal_weight,
                dice_weight=dice_weight,
                focal_alpha=alpha[1],
                focal_gamma=focal_gamma
            )
        elif args.loss == 'tversky':
            print('\n Using Tversky Loss...')
            tversky_alpha = getattr(args, 'tversky_alpha', 0.3)
            tversky_beta = getattr(args, 'tversky_beta', 0.7)
            print(f"Tversky Loss - alpha: {tversky_alpha}, beta: {tversky_beta}")
            self._pxl_loss = get_enhanced_loss_function(
                'tversky',
                tversky_alpha=tversky_alpha,
                tversky_beta=tversky_beta
            )
        elif args.loss == 'lovasz':
            print('\n Using Lovász-Softmax Loss...')
            self._pxl_loss = get_enhanced_loss_function('lovasz')
        elif args.loss == "miou":
            print('\n Calculating Class occurances in training set...')
            alpha   = np.asarray(get_alpha(dataloaders['train'])) # calculare class occurences
            alpha   = alpha/np.sum(alpha)
            weights = 1-torch.from_numpy(alpha).cuda()
            print(f"Weights = {weights}")
            self._pxl_loss = mIoULoss(weight=weights, size_average=True, n_classes=args.n_class).cuda()
        elif args.loss == "mmiou":
            self._pxl_loss = mmIoULoss(n_classes=args.n_class).cuda()
        else:
            raise NotImplementedError(f"Loss function '{args.loss}' is not implemented")

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.npy')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.npy'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.npy')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.npy'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
        if os.path.exists(self.vis_dir) is False:
            os.mkdir(self.vis_dir)

    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(
                checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start)*self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        elif self.args.pretrain is not None:
            print("Initializing backbone weights from: " + self.args.pretrain_path)
            self.net_G.load_state_dict(torch.load(self.args.pretrain_path, weights_only=False), strict=False)
            self.net_G.to(self.device)
            self.net_G.eval()
        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

    def _visualize_pred(self):
        """Create visualization of predictions with proper color mapping"""
        pred = torch.argmax(self.G_final_pred, dim=1, keepdim=True)
    
        # Convert to float and normalize to [0, 1] range for visualization
        pred_vis = pred.float()  # Convert to float first
        return pred_vis  # Return in [0, 1] range for make_numpy_grid

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self):
        """
        update metric
        """
        target = self.batch['L'].to(self.device).detach()
        G_pred = self.G_final_pred.detach()

        G_pred = torch.argmax(G_pred, dim=1)

        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 100) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, G_loss: %.5f, running_mf1: %.5f\n' %\
                    (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m,
                    imps*self.batch_size, est,
                    self.G_loss.item(), running_acc)
            self.logger.write(message)

    # Save visualizations every 500 batches
        if np.mod(self.batch_id, 500) == 1:
            # Create proper visualizations
            vis_input = utils.make_numpy_grid(de_norm(self.batch['A']))
            vis_input2 = utils.make_numpy_grid(de_norm(self.batch['B']))
        
        # Fixed prediction visualization
            pred_vis_tensor = self._visualize_pred()
            vis_pred = utils.make_numpy_grid(pred_vis_tensor)
        
        # Ground truth visualization
            gt_vis = utils.make_numpy_grid(self.batch['L'].float())
        
        # Create a comprehensive visualization
            vis = np.concatenate([vis_input, vis_input2, vis_pred, gt_vis], axis=0)
            vis = np.clip(vis, a_min=0.0, a_max=1.0)
        
        # Save with better naming
            file_name = os.path.join(
                self.vis_dir, f'epoch_{self.epoch_id}_batch_{self.batch_id}_train_{self.is_training}.jpg')
            plt.imsave(file_name, vis)
        
        # Also save individual components for better analysis
            if self.batch_id == 1:  # Save detailed view for first batch of each epoch
                self._save_detailed_visualization()

    def _save_detailed_visualization(self):
        """Save detailed visualization with labels and color-coded predictions"""
        import matplotlib.pyplot as plt
    
        batch_size = min(4, self.batch['A'].size(0))  # Show up to 4 samples
    
        fig, axes = plt.subplots(batch_size, 4, figsize=(16, batch_size * 4))
        if batch_size == 1:
            axes = axes.reshape(1, -1)
    
        for i in range(batch_size):
        # Input A (Image 1)
            img_a = de_norm(self.batch['A'][i]).cpu().numpy().transpose(1, 2, 0)
            img_a = np.clip(img_a, 0, 1)
            axes[i, 0].imshow(img_a)
            axes[i, 0].set_title(f'Image A (Sample {i+1})')
            axes[i, 0].axis('off')
        
        # Input B (Image 2)  
            img_b = de_norm(self.batch['B'][i]).cpu().numpy().transpose(1, 2, 0)
            img_b = np.clip(img_b, 0, 1)
            axes[i, 1].imshow(img_b)
            axes[i, 1].set_title(f'Image B (Sample {i+1})')
            axes[i, 1].axis('off')
        
        # Ground Truth
            gt = self.batch['L'][i, 0].cpu().numpy()
            axes[i, 2].imshow(gt, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i, 2].set_title(f'Ground Truth (Sample {i+1})')
            axes[i, 2].axis('off')
        
        # Prediction
            pred = torch.argmax(self.G_final_pred[i], dim=0).cpu().numpy()
            axes[i, 3].imshow(pred, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[i, 3].set_title(f'Prediction (Sample {i+1})')
            axes[i, 3].axis('off')
    
        plt.tight_layout()
        detailed_file = os.path.join(
            self.vis_dir, f'detailed_epoch_{self.epoch_id}_train_{self.is_training}.jpg')
        plt.savefig(detailed_file, dpi=150, bbox_inches='tight')
        plt.close()



    def _calculate_f1_score(self, predictions, targets):
        """Calculate F1 score from predictions and targets"""
        try:
            # Convert to numpy arrays
            if torch.is_tensor(predictions):
                preds_np = torch.argmax(predictions, dim=1).cpu().numpy().flatten()
            else:
                preds_np = predictions.flatten()
                
            if torch.is_tensor(targets):
                targets_np = targets.cpu().numpy().flatten()
            else:
                targets_np = targets.flatten()
            
            # Calculate F1 score
            f1 = f1_score(targets_np, preds_np, average='binary', zero_division=0)
            return f1
        except Exception as e:
            print(f"Warning: Could not calculate F1 score: {e}")
            return 0.0


    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        
        # ADD THIS: Store F1 for visualization
        if self.is_training:
            self.last_train_f1 = self.epoch_acc
        else:
            self.last_val_f1 = self.epoch_acc
        
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1, self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
        self.logger.write(message+'\n')
        self.logger.write('\n')


    def _update_checkpoints(self):

        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def _update_training_acc_curve(self):
        # update train acc curve
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        # update val acc curve
        self.VAL_ACC = np.append(self.VAL_ACC, [self.epoch_acc])
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()


    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

        if self.multi_scale_infer == "True":
            self.G_final_pred = torch.zeros(self.G_pred[-1].size()).to(self.device)
            for pred in self.G_pred:
                if pred.size(2) != self.G_pred[-1].size(2):
                    self.G_final_pred = self.G_final_pred + F.interpolate(pred, size=self.G_pred[-1].size(2), mode="nearest")
                else:
                    self.G_final_pred = self.G_final_pred + pred
            self.G_final_pred = self.G_final_pred/len(self.G_pred)
        else:
            self.G_final_pred = self.G_pred[-1]

            
    def _backward_G(self):
        gt = self.batch['L'].to(self.device).float()
        if self.multi_scale_train == "True":
            i         = 0
            temp_loss = 0.0
            for pred in self.G_pred:
                if pred.size(2) != gt.size(2):
                    temp_loss = temp_loss + self.weights[i]*self._pxl_loss(pred, F.interpolate(gt, size=pred.size(2), mode="nearest"))
                else:
                    temp_loss = temp_loss + self.weights[i]*self._pxl_loss(pred, gt)
                i+=1
            self.G_loss = temp_loss
        else:
            self.G_loss = self._pxl_loss(self.G_pred[-1], gt)

        self.G_loss.backward()


    def visualize_current_predictions(self, epoch):
        """Create prediction visualizations for the visualizer"""
        if not hasattr(self, 'visualizer') or not self.visualizer:
            return
            
        try:
            # Get a validation batch
            val_iter = iter(self.dataloaders['val'])
            val_batch = next(val_iter)
            
            # Move to device and limit to 4 samples
            images_A = val_batch['A'].to(self.device)[:4]
            images_B = val_batch['B'].to(self.device)[:4]
            targets = val_batch['L'].to(self.device)[:4]
            
            # Get predictions
            self.net_G.eval()
            with torch.no_grad():
                preds = self.net_G(images_A, images_B)
                
                # Handle multi-scale predictions
                if isinstance(preds, list):
                    final_pred = preds[-1]  # Take the final prediction
                else:
                    final_pred = preds
                
                # Convert to probabilities
                if final_pred.shape[1] == 2:  # 2-class output
                    predictions = torch.softmax(final_pred, dim=1)[:, 1]  # Probability of change
                else:  # Single channel output
                    predictions = torch.sigmoid(final_pred.squeeze())
            
            # Create visualization
            self.visualizer.plot_predictions(
                images_A=images_A.cpu(),
                images_B=images_B.cpu(),
                predictions=predictions.cpu(),
                ground_truths=targets.cpu(),
                epoch=epoch
            )
            
        except Exception as e:
            print(f"Warning: Could not create prediction visualization: {e}")
        finally:
            # Set back to training mode if we were training
            if self.is_training:
                self.net_G.train()



    def train_models(self):
        self._load_checkpoint()

        # loop over the dataset multiple times
        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            ################## train #################
            ##########################################
            self._clear_cache()
            self.is_training = True
            self.net_G.train()
            
            # Get current learning rate for visualization
            current_lr = self.optimizer_G.param_groups[0]['lr']
            
            # Iterate over data.
            total = len(self.dataloaders['train'])
            self.logger.write('lr: %0.7f\n \n' % current_lr)
            
            train_loss_sum = 0.0
            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                self._forward_pass(batch)
                # update G
                self.optimizer_G.zero_grad()
                self._backward_G()
                self.optimizer_G.step()
                self._collect_running_batch_states()
                self._timer_update()
                
                # ADD THIS: Accumulate loss for averaging
                train_loss_sum += self.G_loss.item()

            # Calculate average training loss
            avg_train_loss = train_loss_sum / len(self.dataloaders['train'])
            
            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            ################## Eval ##################
            ##########################################
            self.logger.write('Begin evaluation...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            # ADD THIS: Track validation loss
            val_loss_sum = 0.0
            
            # Iterate over data.
            for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                with torch.no_grad():
                    self._forward_pass(batch)
                    # Calculate validation loss (ADD THIS)
                    gt = batch['L'].to(self.device).float()
                    val_loss = self._pxl_loss(self.G_final_pred, gt)
                    val_loss_sum += val_loss.item()
                    
                self._collect_running_batch_states()
                
            # Calculate average validation loss
            avg_val_loss = val_loss_sum / len(self.dataloaders['val'])
            
            self._collect_epoch_states()

            # ADD THIS BLOCK: Update visualizer
            if self.visualizer:
                self.visualizer.update_metrics(
                    epoch=self.epoch_id,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    train_f1=self.last_train_f1,
                    val_f1=self.last_val_f1,
                    lr=current_lr
                )
                
                # Plot training curves every few epochs
                plot_freq = getattr(self.args, 'plot_freq', 5)
                if self.epoch_id % plot_freq == 0 and self.epoch_id > 0:
                    self.visualizer.plot_training_curves()
                
                # Visualize predictions
                vis_freq = getattr(self.args, 'vis_freq', 10)
                if self.epoch_id % vis_freq == 0:
                    self.visualize_current_predictions(self.epoch_id)

            ########### Update_Checkpoints ###########
            ##########################################
            self._update_val_acc_curve()
            self._update_checkpoints()


        if self.visualizer:
            print(" Creating final training visualizations...")
            self.visualizer.plot_training_curves()
            self.visualizer.save_metrics_json()
            print(f"✅ Visualizations saved to: {self.visualizer.vis_dir}")


