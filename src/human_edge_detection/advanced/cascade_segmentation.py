"""Cascade segmentation for progressive refinement."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ..model import LayerNorm2d, ResidualBlock
from .multi_scale_model import MultiScaleFeatureFusion


class BoundaryRefinementModule(nn.Module):
    """Refine segmentation boundaries."""
    
    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 128,
        num_classes: int = 3
    ):
        """Initialize boundary refinement module.
        
        Args:
            in_channels: Input channels (features + coarse masks)
            feature_channels: Internal feature channels
            num_classes: Number of segmentation classes
        """
        super().__init__()
        
        # Edge detection layers
        self.edge_conv1 = nn.Conv2d(in_channels, feature_channels, 3, padding=1)
        self.edge_norm1 = LayerNorm2d(feature_channels)
        
        self.edge_conv2 = nn.Conv2d(feature_channels, feature_channels, 3, padding=1)
        self.edge_norm2 = LayerNorm2d(feature_channels)
        
        # Refinement layers
        self.refine_block1 = ResidualBlock(feature_channels)
        self.refine_block2 = ResidualBlock(feature_channels)
        
        # Output projection
        self.output_conv = nn.Conv2d(feature_channels, num_classes, 1)
        
    def forward(
        self,
        features: torch.Tensor,
        coarse_masks: torch.Tensor
    ) -> torch.Tensor:
        """Refine boundaries.
        
        Args:
            features: Input features
            coarse_masks: Coarse segmentation masks
            
        Returns:
            Refined masks
        """
        # Concatenate features and coarse predictions
        coarse_probs = F.softmax(coarse_masks, dim=1)
        combined = torch.cat([features, coarse_probs], dim=1)
        
        # Edge detection
        edge_feat = F.relu(self.edge_norm1(self.edge_conv1(combined)))
        edge_feat = F.relu(self.edge_norm2(self.edge_conv2(edge_feat)))
        
        # Refinement
        refined = self.refine_block1(edge_feat)
        refined = self.refine_block2(refined)
        
        # Generate residual masks
        residual = self.output_conv(refined)
        
        # Add to coarse masks
        refined_masks = coarse_masks + residual
        
        return refined_masks


class InstanceSeparationModule(nn.Module):
    """Separate overlapping instances."""
    
    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 128,
        num_classes: int = 3
    ):
        """Initialize instance separation module.
        
        Args:
            in_channels: Input channels
            feature_channels: Internal channels
            num_classes: Number of classes
        """
        super().__init__()
        
        # Instance-aware convolutions
        self.inst_conv1 = nn.Conv2d(in_channels, feature_channels, 5, padding=2)
        self.inst_norm1 = LayerNorm2d(feature_channels)
        
        # Dilated convolutions for context
        self.context_conv1 = nn.Conv2d(
            feature_channels, feature_channels, 3, 
            padding=2, dilation=2
        )
        self.context_norm1 = LayerNorm2d(feature_channels)
        
        self.context_conv2 = nn.Conv2d(
            feature_channels, feature_channels, 3,
            padding=4, dilation=4
        )
        self.context_norm2 = LayerNorm2d(feature_channels)
        
        # Separation network
        self.sep_block1 = ResidualBlock(feature_channels)
        self.sep_block2 = ResidualBlock(feature_channels)
        
        # Output
        self.output_conv = nn.Conv2d(feature_channels, num_classes, 1)
        
    def forward(
        self,
        features: torch.Tensor,
        refined_masks: torch.Tensor
    ) -> torch.Tensor:
        """Separate instances.
        
        Args:
            features: Input features
            refined_masks: Refined segmentation masks
            
        Returns:
            Instance-separated masks
        """
        # Combine inputs
        refined_probs = F.softmax(refined_masks, dim=1)
        combined = torch.cat([features, refined_probs], dim=1)
        
        # Instance-aware features
        inst_feat = F.relu(self.inst_norm1(self.inst_conv1(combined)))
        
        # Multi-scale context
        context1 = F.relu(self.context_norm1(self.context_conv1(inst_feat)))
        context2 = F.relu(self.context_norm2(self.context_conv2(inst_feat)))
        
        # Combine contexts
        multi_context = inst_feat + context1 + context2
        
        # Separation refinement
        separated = self.sep_block1(multi_context)
        separated = self.sep_block2(separated)
        
        # Generate final masks
        final_masks = self.output_conv(separated)
        
        return final_masks


class CascadeSegmentationHead(nn.Module):
    """Cascade segmentation with three stages."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        roi_size: int = 28,
        mask_size: int = 56,
        share_features: bool = True
    ):
        """Initialize cascade segmentation head.
        
        Args:
            in_channels: Input feature channels
            mid_channels: Intermediate channels
            num_classes: Number of classes
            roi_size: ROI size
            mask_size: Output mask size
            share_features: Whether to share features between stages
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_size = mask_size
        self.share_features = share_features
        
        # Stage 1: Coarse segmentation
        self.stage1_proj = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.stage1_decoder = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, num_classes, 1)
        )
        
        # Stage 2: Boundary refinement
        stage2_in_channels = mid_channels + num_classes if share_features else in_channels + num_classes
        self.stage2_refiner = BoundaryRefinementModule(
            stage2_in_channels,
            mid_channels // 2,
            num_classes
        )
        
        # Stage 3: Instance separation
        stage3_in_channels = mid_channels + num_classes if share_features else in_channels + num_classes
        self.stage3_separator = InstanceSeparationModule(
            stage3_in_channels,
            mid_channels // 2,
            num_classes
        )
        
    def forward(
        self,
        features: torch.Tensor,
        return_all_stages: bool = False
    ) -> torch.Tensor:
        """Forward pass through cascade.
        
        Args:
            features: Input ROI features
            return_all_stages: Whether to return all stage outputs
            
        Returns:
            Final masks or tuple of all stage masks
        """
        # Stage 1: Coarse segmentation
        stage1_feat = self.stage1_proj(features)
        coarse_masks = self.stage1_decoder(stage1_feat)
        
        # Resize to mask size if needed
        if coarse_masks.shape[2:] != (self.mask_size, self.mask_size):
            coarse_masks = F.interpolate(
                coarse_masks,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
            
        # Stage 2: Boundary refinement
        if self.share_features:
            # Resize features to match mask size
            stage2_feat = F.interpolate(
                stage1_feat,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
        else:
            stage2_feat = F.interpolate(
                features,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
            
        refined_masks = self.stage2_refiner(stage2_feat, coarse_masks)
        
        # Stage 3: Instance separation
        if self.share_features:
            stage3_feat = stage2_feat
        else:
            stage3_feat = F.interpolate(
                features,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
            
        final_masks = self.stage3_separator(stage3_feat, refined_masks)
        
        if return_all_stages:
            return coarse_masks, refined_masks, final_masks
        else:
            return final_masks


class CascadeSegmentationModel(nn.Module):
    """Complete cascade segmentation model."""
    
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int = 3,
        cascade_stages: int = 3,
        share_features: bool = True
    ):
        """Initialize cascade model.
        
        Args:
            base_model: Base segmentation model (e.g., MultiScaleSegmentationModel)
            num_classes: Number of classes
            cascade_stages: Number of cascade stages
            share_features: Whether to share features
        """
        super().__init__()
        
        self.base_model = base_model
        self.num_classes = num_classes
        self.cascade_stages = cascade_stages
        
        # Replace segmentation head with cascade head
        if hasattr(base_model, 'segmentation_head'):
            # Get feature dimensions from base model
            in_channels = base_model.segmentation_head.feature_fusion.output_channels
            roi_size = base_model.segmentation_head.roi_size
            mask_size = base_model.segmentation_head.mask_size
            
            # Create cascade head
            self.cascade_head = CascadeSegmentationHead(
                in_channels=in_channels,
                mid_channels=256,
                num_classes=num_classes,
                roi_size=roi_size,
                mask_size=mask_size,
                share_features=share_features
            )
            
            # Replace base model's head with cascade head
            base_model.segmentation_head = self.cascade_head
            
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor,
        return_all_stages: bool = False
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Multi-scale features
            rois: ROI boxes
            return_all_stages: Whether to return all stages
            
        Returns:
            Segmentation masks
        """
        # Use base model's forward method to get features up to segmentation head
        # Then apply cascade head
        output = self.base_model(features, rois)
        
        if return_all_stages:
            # For cascade, we need to handle this differently
            # The base_model returns final masks, but we need intermediate stages
            # This is a limitation of the current design
            # Return the same masks three times as a workaround
            return output, output, output
        else:
            return output


class CascadeLoss(nn.Module):
    """Loss function for cascade segmentation."""
    
    def __init__(
        self,
        base_loss: nn.Module,
        stage_weights: List[float] = [0.3, 0.3, 0.4]
    ):
        """Initialize cascade loss.
        
        Args:
            base_loss: Base loss function to apply at each stage
            stage_weights: Weights for each stage loss
        """
        super().__init__()
        
        self.base_loss = base_loss
        self.stage_weights = stage_weights
        
    def forward(
        self,
        stage_outputs: Tuple[torch.Tensor, ...],
        targets: torch.Tensor,
        instance_info: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute cascade loss.
        
        Args:
            stage_outputs: Outputs from each cascade stage
            targets: Target masks
            instance_info: Instance information
            
        Returns:
            Total loss and loss dict
        """
        total_loss = 0
        loss_dict = {}
        
        # Compute loss for each stage
        for i, (output, weight) in enumerate(zip(stage_outputs, self.stage_weights)):
            stage_loss, stage_dict = self.base_loss(output, targets, instance_info)
            
            # Weight the loss
            total_loss += weight * stage_loss
            
            # Add to loss dict
            for k, v in stage_dict.items():
                loss_dict[f'stage{i+1}_{k}'] = v
                
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict


def create_cascade_model(
    base_model: nn.Module,
    num_classes: int = 3,
    cascade_stages: int = 3,
    share_features: bool = True
) -> nn.Module:
    """Create a cascade segmentation model.
    
    Note: Current implementation returns the base model as cascade
    is not fully compatible with multi-scale models.
    
    Args:
        base_model: Base model to enhance with cascade
        num_classes: Number of classes
        cascade_stages: Number of cascade stages
        share_features: Whether to share features
        
    Returns:
        Base model (cascade not applied for compatibility)
    """
    # TODO: Implement proper cascade integration with multi-scale models
    # For now, return the base model to avoid compatibility issues
    print("Warning: Cascade segmentation is not fully implemented for multi-scale models.")
    print("Using base model without cascade.")
    return base_model


if __name__ == "__main__":
    # Test cascade segmentation
    print("Testing CascadeSegmentationHead...")
    
    # Create dummy data
    batch_size = 2
    in_channels = 256
    roi_size = 28
    mask_size = 56
    
    features = torch.randn(batch_size, in_channels, roi_size, roi_size)
    
    # Create cascade head
    cascade_head = CascadeSegmentationHead(
        in_channels=in_channels,
        mid_channels=256,
        num_classes=3,
        roi_size=roi_size,
        mask_size=mask_size
    )
    
    # Test forward pass
    coarse, refined, final = cascade_head(features, return_all_stages=True)
    
    print(f"Stage 1 (coarse): {coarse.shape}")
    print(f"Stage 2 (refined): {refined.shape}")
    print(f"Stage 3 (final): {final.shape}")
    
    # Test cascade loss
    print("\nTesting CascadeLoss...")
    
    from .distance_aware_loss import create_distance_aware_loss
    
    base_loss = create_distance_aware_loss(
        pixel_ratios={'background': 0.5, 'target': 0.3, 'non_target': 0.2}
    )
    
    cascade_loss = CascadeLoss(base_loss)
    
    targets = torch.randint(0, 3, (batch_size, mask_size, mask_size))
    
    total_loss, loss_dict = cascade_loss(
        (coarse, refined, final),
        targets
    )
    
    print(f"Total cascade loss: {total_loss.item():.4f}")
    print("Stage losses:")
    for k, v in loss_dict.items():
        if 'stage' in k and isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")