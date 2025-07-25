"""Variable ROI segmentation model without integrated feature extraction."""

import torch
import torch.nn as nn
from typing import Dict, List

from .variable_roi_model import VariableROISegmentationHead


class VariableROISegmentationHeadOnly(nn.Module):
    """Variable ROI segmentation model that accepts pre-extracted features."""
    
    def __init__(
        self,
        feature_channels: Dict[str, int],
        feature_strides: Dict[str, int],
        roi_sizes: Dict[str, int],
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize model.
        
        Args:
            feature_channels: Channels for each feature layer
            feature_strides: Strides for each feature layer
            roi_sizes: ROI sizes for each layer
            num_classes: Number of classes
            mask_size: Output mask size
        """
        super().__init__()
        
        # Segmentation head
        self.segmentation_head = VariableROISegmentationHead(
            feature_channels=feature_channels,
            feature_strides=feature_strides,
            roi_sizes=roi_sizes,
            num_classes=num_classes,
            mask_size=mask_size
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


class VariableROIHeadWithFeatures(nn.Module):
    """Wrapper that extracts intermediate features from variable ROI head."""
    
    def __init__(self, base_head: VariableROISegmentationHeadOnly):
        """Initialize wrapper.
        
        Args:
            base_head: The variable ROI segmentation head
        """
        super().__init__()
        self.segmentation_head = base_head.segmentation_head
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward pass that returns intermediate features.
        
        Args:
            features: Pre-extracted multi-scale features
            rois: ROI boxes (N, 5)
            
        Returns:
            Dictionary with 'features' containing fused features
        """
        # Extract variable-sized ROI features
        roi_features = self.segmentation_head.var_roi_align(features, rois)
        
        # Fuse features hierarchically
        fused_features = self.segmentation_head.feature_fusion(roi_features)
        
        # Return in expected format
        return {'features': fused_features}


def create_variable_roi_head_only(
    target_layers: List[str],
    roi_sizes: Dict[str, int],
    num_classes: int = 3,
    mask_size: int = 56,
    return_features: bool = False
) -> nn.Module:
    """Create a variable ROI segmentation model without feature extraction.
    
    Args:
        target_layers: Layers to use features from
        roi_sizes: Dictionary mapping layer IDs to ROI sizes
        num_classes: Number of segmentation classes
        mask_size: Output mask size
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
    
    # Get feature info for selected layers
    feature_channels = {
        layer: FEATURE_SPECS[layer]['channels']
        for layer in target_layers if layer in FEATURE_SPECS
    }
    feature_strides = {
        layer: FEATURE_SPECS[layer]['stride']
        for layer in target_layers if layer in FEATURE_SPECS
    }
    
    # Create model
    model = VariableROISegmentationHeadOnly(
        feature_channels=feature_channels,
        feature_strides=feature_strides,
        roi_sizes=roi_sizes,
        num_classes=num_classes,
        mask_size=mask_size
    )
    
    # If return_features is True, wrap to return intermediate features
    if return_features:
        model = VariableROIHeadWithFeatures(model)
    
    return model