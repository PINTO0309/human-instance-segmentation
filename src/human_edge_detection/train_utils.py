"""Training utilities for advanced models."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from .model import ROIBatchProcessor
def calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate IoU between predicted and target masks."""
    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (intersection / union).item()


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    feature_extractor: Optional[object] = None,
    config: Optional[object] = None
) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        loss_fn: Loss function
        device: Device to use
        feature_extractor: Optional feature extractor for base models
        config: Optional experiment config for advanced features
        
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
    
    # IoU tracking
    class_ious = {i: [] for i in range(3)}  # Assuming 3 classes
    
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
            
            # Calculate IoU metrics
            pred_classes = pred_for_metrics.argmax(dim=1)  # (N, H, W)
            
            for cls in range(3):
                pred_mask = (pred_classes == cls)
                target_mask = (masks == cls)
                
                # Calculate IoU for each sample
                for i in range(pred_mask.shape[0]):
                    iou = calculate_iou(pred_mask[i], target_mask[i])
                    class_ious[cls].append(iou)
                    
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
    
    # Calculate mean IoU for each class
    for cls in range(3):
        if class_ious[cls]:
            metrics[f'iou_class_{cls}'] = sum(class_ious[cls]) / len(class_ious[cls])
        else:
            metrics[f'iou_class_{cls}'] = 0.0
            
    # Calculate mean IoU (excluding background)
    foreground_ious = []
    for cls in range(1, 3):  # Classes 1 and 2
        if class_ious[cls]:
            foreground_ious.extend(class_ious[cls])
            
    metrics['miou'] = sum(foreground_ious) / len(foreground_ious) if foreground_ious else 0.0
    
    return metrics