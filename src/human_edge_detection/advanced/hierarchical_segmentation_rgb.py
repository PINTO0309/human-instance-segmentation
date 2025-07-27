"""RGB-based Hierarchical Segmentation Models.

This module implements hierarchical segmentation models that work directly 
with RGB image inputs instead of pre-extracted YOLOv9 features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
from ..dynamic_roi_align import DynamicRoIAlign

from .hierarchical_segmentation_unet import (
    HierarchicalSegmentationHeadUNetV2,
    EnhancedUNet,
    ResidualBlock,
    LayerNorm2d
)


class RGBFeatureExtractor(nn.Module):
    """Extract features from RGB ROI images."""
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        roi_size: int = 28,
        num_layers: int = 4,
        normalization_type: str = 'layernorm2d',
        normalization_groups: int = 8
    ):
        """Initialize RGB feature extractor.
        
        Args:
            in_channels: Number of input channels (3 for RGB)
            out_channels: Number of output feature channels
            roi_size: Expected input ROI size
            num_layers: Number of convolutional layers
        """
        super().__init__()
        self.roi_size = roi_size
        
        # Build feature extraction layers
        layers = []
        channels = [in_channels, 64, 128, 192, out_channels][:num_layers + 1]
        
        # Import normalization utilities
        from .normalization_comparison import get_normalization_layer
        
        for i in range(len(channels) - 1):
            # No strided convolutions - keep spatial size constant
            out_ch = channels[i + 1]
            
            # Ensure channels are compatible with GroupNorm
            if normalization_type.lower() == 'groupnorm' and out_ch % normalization_groups != 0:
                out_ch = ((out_ch + normalization_groups - 1) // normalization_groups) * normalization_groups
                channels[i + 1] = out_ch
            
            layers.extend([
                nn.Conv2d(channels[i], out_ch, 3, padding=1, stride=1),
                get_normalization_layer(normalization_type, out_ch, num_groups=normalization_groups),
                nn.ReLU(inplace=True)
            ])
            
            # Add residual blocks for deeper features
            if i >= 1:
                # Create normalization-compatible ResidualBlock
                if normalization_type.lower() == 'groupnorm':
                    from .enhanced_unet_groupnorm import ResidualBlockGroupNorm
                    layers.append(ResidualBlockGroupNorm(out_ch, min(normalization_groups, out_ch)))
                elif normalization_type.lower() in ['batchnorm', 'batchnorm2d']:
                    # Use the flexible ResidualBlock from hierarchical_segmentation_unet
                    from .hierarchical_segmentation_unet import ResidualBlock as FlexibleResidualBlock
                    layers.append(FlexibleResidualBlock(out_ch, normalization_type, normalization_groups))
                else:
                    layers.append(ResidualBlock(out_ch))
        
        self.features = nn.Sequential(*layers)
        
        # Output size is same as input size (no downsampling)
        self.output_size = roi_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from RGB ROIs.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Features tensor of shape (B, out_channels, H', W')
        """
        return self.features(x)


class HierarchicalRGBSegmentationModel(nn.Module):
    """Hierarchical segmentation model with RGB input."""
    
    def __init__(
        self,
        roi_size: Union[int, Tuple[int, int]] = 28,
        mask_size: Union[int, Tuple[int, int]] = 56,
        feature_channels: int = 256,
        num_classes: int = 3,
        use_attention_module: bool = False,
        # Refinement module flags
        use_boundary_refinement: bool = False,
        use_progressive_upsampling: bool = False,
        use_subpixel_conv: bool = False,
        use_contour_detection: bool = False,
        use_distance_transform: bool = False,
        **kwargs  # For additional configuration like normalization
    ):
        """Initialize RGB-based hierarchical segmentation model.
        
        Args:
            roi_size: Size of input ROIs - int for square or (height, width) tuple for non-square
            mask_size: Size of output masks - int for square or (height, width) tuple for non-square
            feature_channels: Number of feature channels after extraction
            num_classes: Number of output classes (3 for hierarchical)
            use_attention_module: Whether to use attention modules
            use_boundary_refinement: Enable boundary refinement
            use_progressive_upsampling: Enable progressive upsampling
            use_subpixel_conv: Enable sub-pixel convolution
            use_contour_detection: Enable contour detection branch
            use_distance_transform: Enable distance transform prediction
        """
        super().__init__()
        
        # Support non-square ROI sizes
        if isinstance(roi_size, (tuple, list)):
            self.roi_height = int(roi_size[0])
            self.roi_width = int(roi_size[1])
            self.roi_size = roi_size[0]  # For compatibility
        else:
            self.roi_height = self.roi_width = self.roi_size = int(roi_size)
            
        # Support non-square mask sizes
        if isinstance(mask_size, (tuple, list)):
            self.mask_height = int(mask_size[0])
            self.mask_width = int(mask_size[1])
            self.mask_size = mask_size  # Keep original for compatibility checks
        else:
            self.mask_height = self.mask_width = int(mask_size)
            self.mask_size = mask_size
        
        # RGB feature extractor
        # Get normalization configuration
        normalization_type = kwargs.get('normalization_type', 'layernorm2d')
        normalization_groups = kwargs.get('normalization_groups', 8)
        
        self.rgb_extractor = RGBFeatureExtractor(
            in_channels=3,
            out_channels=feature_channels,
            roi_size=roi_size,
            num_layers=4,
            normalization_type=normalization_type,
            normalization_groups=normalization_groups
        )
        
        # Check if any refinement modules are enabled
        use_refinement = any([
            use_boundary_refinement,
            use_progressive_upsampling,
            use_subpixel_conv,
            use_contour_detection,
            use_distance_transform
        ])
        
        if use_refinement:
            # Use refined hierarchical segmentation head
            from .hierarchical_segmentation_refinement import RefinedHierarchicalSegmentationHead
            # Get normalization configuration
            normalization_type = kwargs.get('normalization_type', 'layernorm2d')
            normalization_groups = kwargs.get('normalization_groups', 8)
            
            self.segmentation_head = RefinedHierarchicalSegmentationHead(
                in_channels=feature_channels,
                mid_channels=256,
                num_classes=num_classes,
                mask_size=mask_size,
                use_attention_module=use_attention_module,
                use_boundary_refinement=use_boundary_refinement,
                use_progressive_upsampling=use_progressive_upsampling,
                use_subpixel_conv=use_subpixel_conv,
                use_contour_detection=use_contour_detection,
                use_distance_transform=use_distance_transform,
                normalization_type=normalization_type,
                normalization_groups=normalization_groups,
            )
        else:
            # Use standard hierarchical segmentation head
            self.segmentation_head = HierarchicalSegmentationHeadUNetV2(
                in_channels=feature_channels,
                mid_channels=256,
                num_classes=num_classes,
                mask_size=mask_size,
                use_attention_module=use_attention_module
            )
        
        # Dynamic ROI Align for extracting regions from full images
        self.roi_align = DynamicRoIAlign(
            spatial_scale=1.0,  # Assuming ROIs are in image coordinates
            sampling_ratio=2,
            aligned=False
        )
        
    def forward(
        self, 
        images: torch.Tensor, 
        rois: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with RGB images and ROIs.
        
        Args:
            images: Full RGB images of shape (B, 3, H, W)
            rois: ROI coordinates of shape (N, 5) where each row is [batch_idx, x1, y1, x2, y2]
                  Coordinates should be in the same scale as images
            
        Returns:
            Tuple of:
                - Final predictions of shape (N, num_classes, mask_size, mask_size)
                - Dictionary of auxiliary outputs
        """
        # Extract ROI regions from images using DynamicRoIAlign
        roi_features = self.roi_align(images, rois, self.roi_height, self.roi_width)
        
        # Extract features from RGB ROIs
        features = self.rgb_extractor(roi_features)
        
        # Apply hierarchical segmentation head
        predictions, aux_outputs = self.segmentation_head(features)
        
        return predictions, aux_outputs


class MultiScaleRGBSegmentationModel(nn.Module):
    """Multi-scale RGB segmentation model with hierarchical head."""
    
    def __init__(
        self,
        roi_sizes: Dict[str, int] = {'scale1': 56, 'scale2': 42, 'scale3': 28},
        mask_size: int = 56,
        feature_channels: int = 256,
        fusion_method: str = 'concat',
        num_classes: int = 3,
        use_attention_module: bool = False
    ):
        """Initialize multi-scale RGB segmentation model.
        
        Args:
            roi_sizes: Dictionary of scale names to ROI sizes
            mask_size: Size of output masks
            feature_channels: Number of feature channels per scale
            fusion_method: How to fuse multi-scale features ('concat', 'sum', 'adaptive')
            num_classes: Number of output classes
            use_attention_module: Whether to use attention modules
        """
        super().__init__()
        
        self.roi_sizes = roi_sizes
        self.mask_size = mask_size
        self.fusion_method = fusion_method
        self.scales = list(roi_sizes.keys())
        
        # Create RGB extractors for each scale
        self.rgb_extractors = nn.ModuleDict({
            scale: RGBFeatureExtractor(
                in_channels=3,
                out_channels=feature_channels,
                roi_size=roi_size,
                num_layers=4
            )
            for scale, roi_size in roi_sizes.items()
        })
        
        # Dynamic ROI Align for each scale
        self.roi_aligns = nn.ModuleDict({
            scale: DynamicRoIAlign(
                spatial_scale=1.0,
                sampling_ratio=2,
                aligned=False
            )
            for scale, roi_size in roi_sizes.items()
        })
        
        # Feature fusion
        if fusion_method == 'concat':
            fused_channels = feature_channels * len(roi_sizes)
        else:
            fused_channels = feature_channels
            
        if fusion_method == 'adaptive':
            self.fusion_weights = nn.Parameter(torch.ones(len(roi_sizes)))
            
        # Project fused features to expected input size
        self.fusion_proj = nn.Sequential(
            nn.Conv2d(fused_channels, feature_channels, 1),
            LayerNorm2d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Hierarchical segmentation head
        self.segmentation_head = HierarchicalSegmentationHeadUNetV2(
            in_channels=feature_channels,
            mid_channels=256,
            num_classes=num_classes,
            mask_size=mask_size,
            use_attention_module=use_attention_module
        )
        
    def forward(
        self,
        images: torch.Tensor,
        rois: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with multi-scale RGB extraction.
        
        Args:
            images: Full RGB images of shape (B, 3, H, W)
            rois: ROI coordinates of shape (N, 5)
            
        Returns:
            Tuple of predictions and auxiliary outputs
        """
        # Extract features at multiple scales
        scale_features = []
        
        for scale in self.scales:
            # Extract ROIs at this scale
            roi_size = self.roi_sizes[scale]
            roi_regions = self.roi_aligns[scale](images, rois, roi_size, roi_size)
            
            # Extract features
            features = self.rgb_extractors[scale](roi_regions)
            
            # Resize to common size if needed
            if features.shape[-1] != 28:
                features = F.interpolate(
                    features, 
                    size=(28, 28), 
                    mode='bilinear', 
                    align_corners=False
                )
            
            scale_features.append(features)
        
        # Fuse multi-scale features
        if self.fusion_method == 'concat':
            fused = torch.cat(scale_features, dim=1)
        elif self.fusion_method == 'sum':
            fused = sum(scale_features)
        elif self.fusion_method == 'adaptive':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, scale_features))
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        # Project to expected channels
        fused = self.fusion_proj(fused)
        
        # Apply hierarchical segmentation head
        predictions, aux_outputs = self.segmentation_head(fused)
        
        return predictions, aux_outputs


def create_rgb_hierarchical_model(
    roi_size: Union[int, Tuple[int, int]] = 28,
    mask_size: Union[int, Tuple[int, int]] = 56,
    multi_scale: bool = False,
    activation_function: str = 'relu',
    activation_beta: float = 1.0,
    normalization_type: str = 'layernorm2d',
    normalization_groups: int = 8,
    **kwargs
) -> nn.Module:
    """Create RGB-based hierarchical segmentation model.
    
    Args:
        roi_size: Size of input ROIs - int for square or (height, width) tuple for non-square
        mask_size: Size of output masks - int for square or (height, width) tuple for non-square
        multi_scale: Whether to use multi-scale feature extraction
        **kwargs: Additional arguments
        
    Returns:
        Hierarchical segmentation model
    """
    # Extract common parameters
    use_attention_module = kwargs.get('use_attention_module', False)
    
    # Extract refinement parameters
    use_boundary_refinement = kwargs.get('use_boundary_refinement', False)
    use_progressive_upsampling = kwargs.get('use_progressive_upsampling', False)
    use_subpixel_conv = kwargs.get('use_subpixel_conv', False)
    use_contour_detection = kwargs.get('use_contour_detection', False)
    use_distance_transform = kwargs.get('use_distance_transform', False)
    
    if multi_scale:
        roi_sizes = kwargs.get('roi_sizes', {'scale1': 56, 'scale2': 42, 'scale3': 28})
        fusion_method = kwargs.get('fusion_method', 'concat')
        
        return MultiScaleRGBSegmentationModel(
            roi_sizes=roi_sizes,
            mask_size=mask_size,
            fusion_method=fusion_method,
            use_attention_module=use_attention_module,
            # TODO: Add refinement support to MultiScaleRGBSegmentationModel
        )
    else:
        return HierarchicalRGBSegmentationModel(
            roi_size=roi_size,
            mask_size=mask_size,
            use_attention_module=use_attention_module,
            # Binary mask refinement modules
            use_boundary_refinement=use_boundary_refinement,
            use_progressive_upsampling=use_progressive_upsampling,
            use_subpixel_conv=use_subpixel_conv,
            use_contour_detection=use_contour_detection,
            use_distance_transform=use_distance_transform,
            # Pass normalization config
            normalization_type=normalization_type,
            normalization_groups=normalization_groups,
        )