"""Wrapper for multi-scale segmentation head to extract intermediate features."""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .multi_scale_head_only import MultiScaleSegmentationHeadOnly


class MultiScaleHeadWithFeatures(nn.Module):
    """Wrapper that extracts intermediate features from multi-scale head."""
    
    def __init__(self, base_head: MultiScaleSegmentationHeadOnly):
        """Initialize wrapper.
        
        Args:
            base_head: The multi-scale segmentation head
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
        # Extract ROI features
        roi_features = self.segmentation_head.ms_roi_align(features, rois)
        
        # Fuse features
        fused_features = self.segmentation_head.feature_fusion(roi_features)
        
        # Return in expected format
        return {'features': fused_features}