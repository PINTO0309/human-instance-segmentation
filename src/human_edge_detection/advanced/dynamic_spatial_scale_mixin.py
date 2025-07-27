"""Mixin for dynamic spatial scale calculation based on input image size."""

import torch
from typing import Tuple, Union


class DynamicSpatialScaleMixin:
    """Mixin to calculate spatial scale dynamically based on input dimensions."""
    
    @staticmethod
    def get_spatial_scale(
        input_height: int, 
        input_width: int,
        target_size: Union[int, Tuple[int, int]] = (640, 640),
        feature_stride: int = 1
    ) -> Tuple[float, float]:
        """Calculate spatial scale for ROI alignment.
        
        Args:
            input_height: Height of the input image/feature map
            input_width: Width of the input image/feature map
            target_size: Expected size that normalized coordinates refer to
                        Can be int (square) or tuple (height, width)
            feature_stride: Stride of the feature extractor (e.g., 16 for conv5)
        
        Returns:
            Tuple of (scale_h, scale_w) to convert normalized [0,1] coordinates
            to the input feature map space
        """
        if isinstance(target_size, int):
            target_h = target_w = target_size
        else:
            target_h, target_w = target_size
        
        # Calculate scale factors
        # If ROIs are normalized to [0,1] based on target_size,
        # we need to scale them to actual feature map size
        scale_h = input_height / feature_stride
        scale_w = input_width / feature_stride
        
        return (scale_h, scale_w)
    
    def update_roi_align_scales(self, images: torch.Tensor):
        """Update ROI align modules with correct spatial scales.
        
        Args:
            images: Input images tensor (B, C, H, W)
        """
        batch_size, channels, height, width = images.shape
        
        # Update each ROI align module if it exists
        if hasattr(self, 'roi_align'):
            self.roi_align.spatial_scale_h = height
            self.roi_align.spatial_scale_w = width
            
        if hasattr(self, 'image_roi_align'):
            self.image_roi_align.spatial_scale_h = height
            self.image_roi_align.spatial_scale_w = width
            
        if hasattr(self, 'roi_align_mask'):
            self.roi_align_mask.spatial_scale_h = height
            self.roi_align_mask.spatial_scale_w = width
            
        if hasattr(self, 'unet_roi_align'):
            self.unet_roi_align.spatial_scale_h = height
            self.unet_roi_align.spatial_scale_w = width
            
        if hasattr(self, 'roi_align_rgb'):
            self.roi_align_rgb.spatial_scale_h = height
            self.roi_align_rgb.spatial_scale_w = width
            
        # For feature extractors with different strides
        if hasattr(self, 'conv5_roi_align'):
            # Conv5 typically has stride 16
            self.conv5_roi_align.spatial_scale_h = height / 16
            self.conv5_roi_align.spatial_scale_w = width / 16