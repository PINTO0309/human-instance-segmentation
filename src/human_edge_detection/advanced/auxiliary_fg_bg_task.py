"""Auxiliary foreground/background binary mask prediction task."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class AuxiliaryFgBgHead(nn.Module):
    """Auxiliary head for binary foreground/background prediction."""
    
    def __init__(
        self, 
        in_channels: int,
        mid_channels: int = 128,
        mask_size: int = 56
    ):
        super().__init__()
        
        # Lightweight binary segmentation head
        self.fg_bg_head = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1)  # Binary output
        )
        
        # Upsampling if needed
        if mask_size > 28:
            self.upsample = nn.ConvTranspose2d(1, 1, 2, stride=2)
        else:
            self.upsample = nn.Identity()
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass for binary fg/bg prediction."""
        logits = self.fg_bg_head(features)
        return self.upsample(logits)


class MultiTaskSegmentationModel(nn.Module):
    """Segmentation model with auxiliary fg/bg task."""
    
    def __init__(
        self,
        base_segmentation_head: nn.Module,
        in_channels: int,
        mask_size: int = 56,
        aux_weight: float = 0.3
    ):
        super().__init__()
        
        # Main segmentation head (3-class)
        self.main_head = base_segmentation_head
        
        # Auxiliary fg/bg head
        self.aux_head = AuxiliaryFgBgHead(
            in_channels=in_channels,
            mask_size=mask_size
        )
        
        self.aux_weight = aux_weight
        
    def forward(
        self, 
        features: torch.Tensor,
        rois: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with main and auxiliary predictions.
        
        Args:
            features: Feature tensor or dict of features
            rois: Optional ROI tensor for models that need it
        """
        
        # For multi-scale models, we need to handle the case where the model
        # internally extracts ROI features
        roi_features = None
        
        # Main 3-class segmentation
        if hasattr(self.main_head, 'forward'):
            # Check if main head expects rois
            import inspect
            sig = inspect.signature(self.main_head.forward)
            params = list(sig.parameters.keys())
            
            if len(params) > 1 and rois is not None:
                # Model expects both features and rois
                main_output = self.main_head(features, rois)
            else:
                # Model only expects features
                main_output = self.main_head(features)
                
            if isinstance(main_output, tuple):
                main_logits, main_aux = main_output
            else:
                main_logits = main_output
                main_aux = {}
                
            # Check if model has exposed ROI features for auxiliary task
            if hasattr(self.main_head, 'last_roi_features'):
                roi_features = self.main_head.last_roi_features
        else:
            # Fallback for non-standard models
            if rois is not None:
                main_logits = self.main_head(features, rois)
            else:
                main_logits = self.main_head(features)
            main_aux = {}
        
        # Auxiliary binary fg/bg prediction
        # We need ROI-aligned features for the auxiliary head
        if roi_features is not None:
            # Use the ROI features from the main model
            aux_logits = self.aux_head(roi_features)
        else:
            # Fallback: assume features are already ROI-aligned
            # This works for models that output ROI features directly
            if isinstance(features, dict):
                # Should not happen if model is properly integrated
                raise ValueError("Auxiliary task requires ROI-aligned features, but received feature dict. "
                               "The model needs to expose ROI features for auxiliary task.")
            aux_logits = self.aux_head(features)
        
        # Add auxiliary output
        aux_outputs = {
            **main_aux,
            'fg_bg_binary': aux_logits
        }
        
        return main_logits, aux_outputs


class MultiTaskLoss(nn.Module):
    """Loss function combining main task and auxiliary fg/bg task."""
    
    def __init__(
        self,
        main_loss_fn: nn.Module,
        aux_weight: float = 0.3,
        aux_pos_weight: Optional[float] = None
    ):
        """Initialize multi-task loss.
        
        Args:
            main_loss_fn: Main segmentation loss function
            aux_weight: Weight for auxiliary task loss
            aux_pos_weight: Positive class weight for binary loss
        """
        super().__init__()
        self.main_loss_fn = main_loss_fn
        self.aux_weight = aux_weight
        
        # Binary cross-entropy for auxiliary task
        if aux_pos_weight is not None:
            self.aux_loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([aux_pos_weight])
            )
        else:
            self.aux_loss_fn = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss."""
        
        # Main segmentation loss
        if 'fg_bg_binary' in aux_outputs:
            # Remove auxiliary prediction before passing to main loss
            aux_pred = aux_outputs.pop('fg_bg_binary')
            main_loss, main_dict = self.main_loss_fn(predictions, targets, aux_outputs)
            
            # Create binary fg/bg targets from 3-class targets
            fg_targets = (targets > 0).float()  # 0=bg, 1,2=fg
            
            # Auxiliary binary loss
            aux_loss = self.aux_loss_fn(aux_pred.squeeze(1), fg_targets)
            
            # Combined loss
            total_loss = main_loss + self.aux_weight * aux_loss
            
            # Update loss dict
            loss_dict = {
                **main_dict,
                'aux_fg_bg_loss': aux_loss.item(),
                'total_loss': total_loss.item()
            }
            
            # Compute auxiliary metrics
            with torch.no_grad():
                aux_probs = torch.sigmoid(aux_pred.squeeze(1))
                aux_preds = (aux_probs > 0.5).float()
                aux_accuracy = (aux_preds == fg_targets).float().mean()
                
                # IoU for foreground
                intersection = (aux_preds * fg_targets).sum()
                union = aux_preds.sum() + fg_targets.sum() - intersection
                fg_iou = (intersection / union.clamp(min=1)).item()
                
                loss_dict['aux_fg_accuracy'] = aux_accuracy.item()
                loss_dict['aux_fg_iou'] = fg_iou
            
            # Restore auxiliary output for potential visualization
            aux_outputs['fg_bg_binary'] = aux_pred
            
            return total_loss, loss_dict
        else:
            # Fallback to main loss only
            return self.main_loss_fn(predictions, targets, aux_outputs)


def create_multitask_model(
    base_model: nn.Module,
    feature_channels: int,
    mask_size: int = 56,
    aux_weight: float = 0.3
) -> nn.Module:
    """Wrap existing model with auxiliary fg/bg task.
    
    Args:
        base_model: Base segmentation model
        feature_channels: Number of feature channels
        mask_size: Output mask size
        aux_weight: Weight for auxiliary task
        
    Returns:
        Multi-task model
    """
    
    # Extract segmentation head from base model
    if hasattr(base_model, 'segmentation_head'):
        seg_head = base_model.segmentation_head
    else:
        # Assume the whole model is the segmentation head
        seg_head = base_model
    
    # Create multi-task model
    mt_model = MultiTaskSegmentationModel(
        base_segmentation_head=seg_head,
        in_channels=feature_channels,
        mask_size=mask_size,
        aux_weight=aux_weight
    )
    
    return mt_model