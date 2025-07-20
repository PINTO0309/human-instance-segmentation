"""Hierarchical segmentation architecture for stable multi-class learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..model import LayerNorm2d, ResidualBlock


class HierarchicalSegmentationHead(nn.Module):
    """Hierarchical segmentation head that separates background vs foreground from target vs non-target."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head.
        
        Args:
            in_channels: Input feature channels
            mid_channels: Intermediate channel count
            num_classes: Total number of classes (should be 3)
            mask_size: Output mask size
        """
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"
        
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
        )
        
        # Branch 1: Background vs Foreground (binary)
        self.bg_vs_fg_branch = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, 2, 1)  # 2 classes: bg, fg
        )
        
        # Branch 2: Target vs Non-target (within foreground)
        self.target_vs_nontarget_branch = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, 2, 1)  # 2 classes: target, non-target
        )
        
        # Gating mechanism to focus target/non-target branch on foreground regions
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical prediction.
        
        Args:
            features: Input features (N, C, H, W)
            
        Returns:
            final_logits: Combined 3-class logits (N, 3, H, W)
            aux_outputs: Dictionary with intermediate predictions
        """
        # Shared feature extraction
        shared = self.shared_features(features)
        
        # Branch 1: Background vs Foreground
        bg_fg_logits = self.bg_vs_fg_branch(shared)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        # Create foreground attention gate
        fg_attention = self.fg_gate(bg_fg_logits)
        
        # Branch 2: Target vs Non-target (modulated by foreground attention)
        gated_features = shared * fg_attention
        target_nontarget_logits = self.target_vs_nontarget_branch(gated_features)
        
        # Combine predictions hierarchically
        # Final logits: [background, target, non-target]
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_size, self.mask_size, 
                                  device=features.device)
        
        # Background probability from first branch
        final_logits[:, 0] = bg_fg_logits[:, 0]  # Background logit
        
        # Target and non-target from second branch, scaled by foreground probability
        # This ensures target/non-target predictions are only strong in foreground regions
        fg_mask = bg_fg_probs[:, 1:2]  # Foreground probability
        final_logits[:, 1] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 0] * fg_mask.squeeze(1)
        final_logits[:, 2] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 1] * fg_mask.squeeze(1)
        
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention
        }
        
        return final_logits, aux_outputs


class HierarchicalLoss(nn.Module):
    """Loss function for hierarchical segmentation."""
    
    def __init__(
        self,
        bg_weight: float = 1.0,
        fg_weight: float = 1.0,
        target_weight: float = 1.0,
        consistency_weight: float = 0.1
    ):
        """Initialize hierarchical loss.
        
        Args:
            bg_weight: Weight for background vs foreground loss
            fg_weight: Weight for target vs non-target loss
            target_weight: Additional weight for target class
            consistency_weight: Weight for consistency between branches
        """
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.target_weight = target_weight
        self.consistency_weight = consistency_weight
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hierarchical loss.
        
        Args:
            predictions: Final 3-class predictions (N, 3, H, W)
            targets: Ground truth labels (N, H, W)
            aux_outputs: Auxiliary outputs from hierarchical head
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Create binary masks for hierarchical supervision
        bg_mask = (targets == 0).long()
        fg_mask = (targets > 0).long()
        target_mask = (targets == 1).long()
        nontarget_mask = (targets == 2).long()
        
        # Loss 1: Background vs Foreground
        bg_fg_targets = fg_mask  # 0 for background, 1 for foreground
        bg_fg_loss = F.cross_entropy(
            aux_outputs['bg_fg_logits'], 
            bg_fg_targets,
            weight=torch.tensor([1.0, self.target_weight]).to(predictions.device)
        )
        
        # Loss 2: Target vs Non-target (only on foreground pixels)
        if fg_mask.any():
            # Create targets for foreground pixels: 0 for target, 1 for non-target
            fg_indices = fg_mask.bool()
            target_nontarget_targets = torch.zeros_like(targets)
            target_nontarget_targets[target_mask.bool()] = 0
            target_nontarget_targets[nontarget_mask.bool()] = 1
            
            # Masked loss on foreground pixels only
            tn_logits_masked = aux_outputs['target_nontarget_logits'][:, :, fg_indices]
            tn_targets_masked = target_nontarget_targets[fg_indices]
            
            target_nontarget_loss = F.cross_entropy(
                tn_logits_masked.transpose(1, 2),
                tn_targets_masked
            )
        else:
            target_nontarget_loss = torch.tensor(0.0, device=predictions.device)
        
        # Loss 3: Final 3-class cross entropy
        final_loss = F.cross_entropy(predictions, targets)
        
        # Loss 4: Consistency regularization
        # Ensure that background prediction from branch 1 matches final background prediction
        consistency_loss = F.mse_loss(
            F.softmax(aux_outputs['bg_fg_logits'], dim=1)[:, 0],
            F.softmax(predictions, dim=1)[:, 0]
        )
        
        # Combine losses
        total_loss = (
            self.bg_weight * bg_fg_loss +
            self.fg_weight * target_nontarget_loss +
            final_loss +
            self.consistency_weight * consistency_loss
        )
        
        loss_dict = {
            'bg_fg_loss': bg_fg_loss.item(),
            'target_nontarget_loss': target_nontarget_loss.item(),
            'final_loss': final_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


def create_hierarchical_model(base_model: nn.Module) -> nn.Module:
    """Wrap a base model with hierarchical segmentation head.
    
    Args:
        base_model: Base segmentation model
        
    Returns:
        Model with hierarchical segmentation
    """
    class HierarchicalSegmentationModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            
            # Replace the segmentation head
            if hasattr(base_model, 'segmentation_head'):
                # Get the input channels from the original head
                old_head = base_model.segmentation_head
                if hasattr(old_head, 'decoder'):
                    # Find the first conv layer to get input channels
                    first_layer = old_head.decoder[0]
                    if isinstance(first_layer, ResidualBlock):
                        in_channels = first_layer.conv1.in_channels
                    else:
                        in_channels = 256  # Default
                else:
                    in_channels = 256
                
                # Create hierarchical head
                self.hierarchical_head = HierarchicalSegmentationHead(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )
                
                # Keep original components for feature extraction
                if hasattr(old_head, 'var_roi_align'):
                    self.roi_align = old_head.var_roi_align
                if hasattr(old_head, 'feature_fusion'):
                    self.feature_fusion = old_head.feature_fusion
                    
        def forward(self, features, rois, rgb_features=None):
            # Extract ROI features using original logic
            if hasattr(self, 'roi_align'):
                roi_features = self.roi_align(features, rois)
                
                # Handle RGB features if present
                if rgb_features is not None and hasattr(self.base_model, 'segmentation_head'):
                    old_head = self.base_model.segmentation_head
                    if hasattr(old_head, 'rgb_roi_aligns'):
                        # Process RGB features for enhanced layers
                        for layer_name in old_head.rgb_enhanced_layers:
                            if layer_name in roi_features and layer_name in old_head.rgb_roi_aligns:
                                rgb_roi_features = old_head.rgb_roi_aligns[layer_name](
                                    rgb_features, rois,
                                    old_head.rgb_roi_sizes[layer_name],
                                    old_head.rgb_roi_sizes[layer_name]
                                )
                                rgb_roi_features = old_head.rgb_projections[layer_name](rgb_roi_features)
                                roi_features[layer_name] = torch.cat([
                                    roi_features[layer_name],
                                    rgb_roi_features
                                ], dim=1)
                
                # Fuse features
                if hasattr(self, 'feature_fusion'):
                    fused_features = self.feature_fusion(roi_features)
                else:
                    fused_features = list(roi_features.values())[0]
            else:
                # Fallback for simpler models
                output = self.base_model(features=features, rois=rois)
                fused_features = output['features'] if isinstance(output, dict) else features
            
            # Apply hierarchical head
            logits, aux_outputs = self.hierarchical_head(fused_features)
            
            # Return in expected format
            if self.training:
                return logits, aux_outputs
            else:
                return logits
    
    return HierarchicalSegmentationModel(base_model)