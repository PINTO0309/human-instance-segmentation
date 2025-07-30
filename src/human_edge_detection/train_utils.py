"""Training utilities for advanced models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from .model import ROIBatchProcessor
def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate IoU between predicted and target masks."""
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def calculate_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3) -> torch.Tensor:
    """Calculate confusion matrix for segmentation.

    Args:
        pred: Predicted class labels (N, H, W)
        target: Target class labels (N, H, W)
        num_classes: Number of classes

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    # Flatten the tensors
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Create confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for t in range(num_classes):
        for p in range(num_classes):
            conf_matrix[t, p] = ((target_flat == t) & (pred_flat == p)).sum()

    return conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list, save_path: Path, epoch: int):
    """Plot and save confusion matrix.

    Args:
        conf_matrix: Confusion matrix
        class_names: List of class names
        save_path: Path to save the figure
        epoch: Current epoch number
    """
    # Normalize confusion matrix by row (true labels)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_norm = np.nan_to_num(conf_matrix_norm)  # Handle division by zero

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot raw counts
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title(f'Confusion Matrix (Counts) - Epoch {epoch}')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')

    # Plot normalized
    sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title(f'Confusion Matrix (Normalized) - Epoch {epoch}')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_detection_metrics(ious: list, thresholds: list = [0.5, 0.7]) -> dict:
    """Calculate detection success rates at different IoU thresholds.

    Args:
        ious: List of IoU values
        thresholds: IoU thresholds for detection success

    Returns:
        Dictionary with detection rates
    """
    metrics = {}
    if not ious:
        for t in thresholds:
            metrics[f'detection_rate_{t}'] = 0.0
        return metrics

    ious_array = np.array(ious)
    for threshold in thresholds:
        success_rate = (ious_array > threshold).mean()
        metrics[f'detection_rate_{threshold}'] = success_rate

    return metrics


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    feature_extractor: Optional[object] = None,
    config: Optional[object] = None,
    epoch: Optional[int] = None,
    output_dir: Optional[Path] = None
) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        loss_fn: Loss function
        device: Device to use
        feature_extractor: Optional feature extractor for base models
        config: Optional experiment config for advanced features
        epoch: Current epoch number (for confusion matrix saving)
        output_dir: Directory to save confusion matrices

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0

    # Auxiliary metrics tracking
    aux_fg_bg_loss = 0
    aux_fg_accuracy = 0
    aux_fg_iou = 0

    # Refinement losses
    active_contour_loss = 0
    boundary_aware_loss = 0
    contour_loss = 0
    distance_transform_loss = 0

    # IoU tracking
    class_ious = {i: [] for i in range(3)}  # Assuming 3 classes
    target_ious = []  # Separate tracking for target IoUs

    # Confusion matrices (keep on CPU for accumulation)
    conf_matrix_total = torch.zeros(3, 3, dtype=torch.int64)
    conf_matrix_bg_target = torch.zeros(2, 2, dtype=torch.int64)  # Background vs Target
    conf_matrix_target_nontarget = torch.zeros(2, 2, dtype=torch.int64)  # Target vs Non-target

    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Validation', dynamic_ncols=True, leave=False)

        for batch in progress_bar:
            # Move to device
            images = batch['image'].to(device)
            rois = batch['roi_boxes'].to(device)
            masks = batch['roi_masks'].to(device)

            # Extract features if using base model
            if feature_extractor is not None:
                features = feature_extractor.extract_features(images)
            else:
                features = images  # Multi-scale model extracts internally

            # Forward pass
            if config and config.model.use_rgb_hierarchical:
                # RGB hierarchical model takes full images and ROIs
                predictions = model(images, rois)
            elif config and config.multiscale.enabled:
                # Multi-scale model
                if config.cascade.enabled and hasattr(model, 'cascade_head'):
                    # Only use return_all_stages if model actually supports cascade
                    predictions = model(features, rois, return_all_stages=True)
                else:
                    predictions = model(features, rois)
            else:
                # Base model - call with named arguments
                output = model(features=features, rois=rois)
                predictions = output['masks']

            # Compute loss
            instance_info = batch.get('instance_info') if config and config.distance_loss.enabled else None

            # Handle hierarchical model output
            is_hierarchical = config and (config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']))
            if is_hierarchical:
                if isinstance(predictions, tuple):
                    logits, aux_outputs = predictions
                    loss, loss_dict = loss_fn(logits, masks, aux_outputs)
                    pred_for_metrics = logits
                else:
                    # Hierarchical model should return tuple, but didn't
                    loss, loss_dict = loss_fn(predictions, masks, {})
                    pred_for_metrics = predictions
            elif config and config.cascade.enabled and isinstance(predictions, tuple):
                # Use final stage predictions for metrics
                loss, loss_dict = loss_fn(predictions, masks, instance_info)
                pred_for_metrics = predictions[-1]  # Last stage
            else:
                # Standard loss - check if distance loss is enabled
                if config and config.distance_loss.enabled:
                    loss, loss_dict = loss_fn(predictions, masks, instance_info)
                else:
                    loss, loss_dict = loss_fn(predictions, masks)
                pred_for_metrics = predictions

            # Update loss metrics
            total_loss += loss.item()
            total_ce_loss += loss_dict.get('ce_loss', 0).item() if isinstance(loss_dict.get('ce_loss', 0), torch.Tensor) else loss_dict.get('ce_loss', 0)
            total_dice_loss += loss_dict.get('dice_loss', 0).item() if isinstance(loss_dict.get('dice_loss', 0), torch.Tensor) else loss_dict.get('dice_loss', 0)

            # Track auxiliary metrics if available
            aux_fg_bg_loss += loss_dict.get('aux_fg_bg_loss', 0)
            aux_fg_accuracy += loss_dict.get('aux_fg_accuracy', 0)
            aux_fg_iou += loss_dict.get('aux_fg_iou', 0)

            # Track refinement losses if available
            active_contour_loss += loss_dict.get('active_contour', 0)
            boundary_aware_loss += loss_dict.get('boundary_aware', 0)
            contour_loss += loss_dict.get('contour', 0)
            distance_transform_loss += loss_dict.get('distance_transform', 0)

            # Calculate IoU metrics
            pred_classes = pred_for_metrics.argmax(dim=1)  # (N, H, W)

            # Move to CPU for confusion matrix calculation
            pred_classes_cpu = pred_classes.cpu()
            masks_cpu = masks.cpu()

            # Update confusion matrices
            batch_conf_matrix = calculate_confusion_matrix(pred_classes_cpu, masks_cpu)
            conf_matrix_total += batch_conf_matrix

            # Create binary masks for specific confusion matrices
            # Background (0) vs Target (1)
            pred_bg_target = (pred_classes_cpu == 1).long()  # 1 if target, 0 otherwise
            target_bg_target = (masks_cpu == 1).long()

            for i in range(pred_classes_cpu.shape[0]):
                # Background vs Target confusion
                for t in range(2):
                    for p in range(2):
                        conf_matrix_bg_target[t, p] += ((target_bg_target[i] == t) & (pred_bg_target[i] == p)).sum().item()

                # Target vs Non-target confusion (only where ground truth is foreground)
                fg_mask = (masks_cpu[i] > 0)  # Foreground pixels only
                if fg_mask.any():
                    pred_tn = (pred_classes_cpu[i] == 2).long()[fg_mask]  # 1 if non-target, 0 if target
                    target_tn = (masks_cpu[i] == 2).long()[fg_mask]
                    for t in range(2):
                        for p in range(2):
                            conf_matrix_target_nontarget[t, p] += ((target_tn == t) & (pred_tn == p)).sum().item()

            # Calculate per-class IoUs
            for cls in range(3):
                pred_mask = (pred_classes == cls)
                target_mask = (masks == cls)

                # Calculate IoU for each sample
                for i in range(pred_mask.shape[0]):
                    iou = calculate_iou(pred_mask[i], target_mask[i])
                    class_ious[cls].append(iou)

                    # Track target IoUs separately
                    if cls == 1:
                        target_ious.append(iou)

            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}'
            })

    # Calculate average metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches
    }

    # Add auxiliary metrics if available
    if aux_fg_bg_loss > 0:
        metrics['aux_fg_bg_loss'] = aux_fg_bg_loss / num_batches
        metrics['aux_fg_accuracy'] = aux_fg_accuracy / num_batches
        metrics['aux_fg_iou'] = aux_fg_iou / num_batches

    # Add refinement losses if available
    if active_contour_loss > 0:
        metrics['active_contour'] = active_contour_loss / num_batches
    if boundary_aware_loss > 0:
        metrics['boundary_aware'] = boundary_aware_loss / num_batches
    if contour_loss > 0:
        metrics['contour'] = contour_loss / num_batches
    if distance_transform_loss > 0:
        metrics['distance_transform'] = distance_transform_loss / num_batches

    # Calculate mean IoU for each class
    for cls in range(3):
        if class_ious[cls]:
            metrics[f'iou_class_{cls}'] = sum(class_ious[cls]) / len(class_ious[cls])
        else:
            metrics[f'iou_class_{cls}'] = 0.0

    # Primary metric: Target IoU only
    metrics['target_iou'] = sum(target_ious) / len(target_ious) if target_ious else 0.0
    metrics['miou'] = metrics['target_iou']  # Use target IoU as primary metric

    # Detection success rates
    detection_metrics = calculate_detection_metrics(target_ious)
    metrics.update(detection_metrics)

    # Confusion matrix metrics
    # Convert to numpy for easier manipulation
    conf_total_np = conf_matrix_total.cpu().numpy()
    conf_bg_target_np = conf_matrix_bg_target.cpu().numpy()
    conf_tn_np = conf_matrix_target_nontarget.cpu().numpy()

    # Calculate accuracies from confusion matrices
    if conf_total_np.sum() > 0:
        metrics['overall_accuracy'] = np.diag(conf_total_np).sum() / conf_total_np.sum()

    # Background vs Target metrics
    if conf_bg_target_np.sum() > 0:
        # Precision and recall for target detection
        target_tp = conf_bg_target_np[1, 1]
        target_fp = conf_bg_target_np[0, 1]
        target_fn = conf_bg_target_np[1, 0]

        if target_tp + target_fp > 0:
            metrics['target_precision'] = target_tp / (target_tp + target_fp)
        else:
            metrics['target_precision'] = 0.0

        if target_tp + target_fn > 0:
            metrics['target_recall'] = target_tp / (target_tp + target_fn)
        else:
            metrics['target_recall'] = 0.0

        # Calculate F1 score
        if metrics['target_precision'] + metrics['target_recall'] > 0:
            metrics['target_f1'] = 2 * (metrics['target_precision'] * metrics['target_recall']) / (metrics['target_precision'] + metrics['target_recall'])
        else:
            metrics['target_f1'] = 0.0

    # Target vs Non-target metrics (instance separation)
    if conf_tn_np.sum() > 0:
        # Accuracy of distinguishing target from non-target in foreground regions
        metrics['instance_separation_accuracy'] = np.diag(conf_tn_np).sum() / conf_tn_np.sum()

    # Save confusion matrices if output_dir is provided
    if output_dir is not None and epoch is not None:
        # Create confusion matrix directory
        cm_dir = output_dir / 'confusion_matrices'
        cm_dir.mkdir(parents=True, exist_ok=True)

        # Plot and save full confusion matrix
        class_names = ['Background', 'Target', 'Non-target']
        plot_confusion_matrix(conf_total_np, class_names,
                            cm_dir / f'confusion_matrix_full_epoch_{epoch+1:04d}.png', epoch)

        # Plot and save background vs target confusion matrix
        bg_target_names = ['Background', 'Target']
        plot_confusion_matrix(conf_bg_target_np, bg_target_names,
                            cm_dir / f'confusion_matrix_bg_target_epoch_{epoch+1:04d}.png', epoch)

        # Plot and save target vs non-target confusion matrix
        tn_names = ['Target', 'Non-target']
        plot_confusion_matrix(conf_tn_np, tn_names,
                            cm_dir / f'confusion_matrix_target_nontarget_epoch_{epoch+1:04d}.png', epoch)

    # Store confusion matrices in metrics for logging
    metrics['conf_matrix_total'] = conf_total_np
    metrics['conf_matrix_bg_target'] = conf_bg_target_np
    metrics['conf_matrix_target_nontarget'] = conf_tn_np

    return metrics