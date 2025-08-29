"""Multi-scale segmentation model without integrated feature extraction."""

import torch
import torch.nn as nn
from typing import Dict, List

from .multi_scale_model import MultiScaleROISegmentationHead


class MultiScaleSegmentationHeadOnly(nn.Module):
    """Multi-scale segmentation model that accepts pre-extracted features."""
    
    def __init__(
        self,
        feature_specs: Dict[str, dict],
        num_classes: int = 3,
        roi_size: int = 28,
        mask_size: int = 56,
        fusion_method: str = 'adaptive'
    ):
        """Initialize model.
        
        Args:
            feature_specs: Feature specifications
            num_classes: Number of classes
            roi_size: ROI size
            mask_size: Output mask size
            fusion_method: Feature fusion method
        """
        super().__init__()
        
        self.feature_specs = feature_specs
        
        # Extract channel and stride info
        feature_channels = {
            lid: spec['channels'] for lid, spec in feature_specs.items()
        }
        feature_strides = {
            lid: spec['stride'] for lid, spec in feature_specs.items()
        }
        
        # Segmentation head
        self.segmentation_head = MultiScaleROISegmentationHead(
            feature_channels=feature_channels,
            feature_strides=feature_strides,
            num_classes=num_classes,
            roi_size=roi_size,
            mask_size=mask_size,
            fusion_method=fusion_method
        )
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Pre-extracted multi-scale features
            rois: ROI boxes (N, 5)
            
        Returns:
            Segmentation masks (N, num_classes, mask_size, mask_size)
        """
        # Generate masks
        masks = self.segmentation_head(features, rois)
        
        return masks
    
    def get_trainable_parameters(self):
        """Get only trainable parameters."""
        return self.segmentation_head.parameters()


def create_multiscale_head_only(
    target_layers: List[str] = ['layer_3', 'layer_22', 'layer_34'],
    num_classes: int = 3,
    roi_size: int = 28,
    mask_size: int = 56,
    fusion_method: str = 'adaptive',
    return_features: bool = False
) -> nn.Module:
    """Create a multi-scale segmentation model without feature extraction.
    
    Args:
        target_layers: Layers to use features from
        num_classes: Number of segmentation classes
        roi_size: ROI size after alignment
        mask_size: Output mask size
        fusion_method: Feature fusion method
        return_features: If True, wrap to return intermediate features
        
    Returns:
        Initialized model
    """
    # Feature specifications for YOLO layers
    FEATURE_SPECS = {
        'layer_3': {'channels': 256, 'stride': 4},    # 160x160
        'layer_19': {'channels': 256, 'stride': 4},   # 160x160
        'layer_5': {'channels': 512, 'stride': 8},    # 80x80
        'layer_22': {'channels': 512, 'stride': 8},   # 80x80
        'layer_34': {'channels': 1024, 'stride': 8}   # 80x80
    }
    
    # Get feature specs for target layers
    feature_specs = {
        lid: FEATURE_SPECS[lid] for lid in target_layers if lid in FEATURE_SPECS
    }
    
    # Create model
    model = MultiScaleSegmentationHeadOnly(
        feature_specs=feature_specs,
        num_classes=num_classes,
        roi_size=roi_size,
        mask_size=mask_size,
        fusion_method=fusion_method
    )
    
    # If return_features is True, wrap to return intermediate features
    if return_features:
        from .multi_scale_head_wrapper import MultiScaleHeadWithFeatures
        model = MultiScaleHeadWithFeatures(model)
    
    return model