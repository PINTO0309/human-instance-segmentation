"""Variable ROI size multi-scale model for experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ..dynamic_roi_align import DynamicRoIAlign
from ..model import LayerNorm2d, ResidualBlock


class VariableROIAlign(nn.Module):
    """Extract ROI features with different sizes for different scales."""
    
    def __init__(
        self,
        feature_strides: Dict[str, int],
        roi_sizes: Dict[str, int],
        sampling_ratio: int = 2
    ):
        """Initialize variable ROI align.
        
        Args:
            feature_strides: Dictionary mapping layer IDs to feature strides
            roi_sizes: Dictionary mapping layer IDs to ROI sizes
            sampling_ratio: Number of sampling points
        """
        super().__init__()
        
        self.feature_strides = feature_strides
        self.roi_sizes = roi_sizes
        
        # Create ROI align for each scale
        self.roi_aligns = nn.ModuleDict()
        
        for layer_id, stride in feature_strides.items():
            self.roi_aligns[layer_id] = DynamicRoIAlign(
                spatial_scale=1.0 / stride,
                sampling_ratio=sampling_ratio,
                aligned=True
            )
            
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract ROI features with variable sizes.
        
        Args:
            features: Dictionary of feature maps
            rois: ROI boxes (N, 5) - [batch_idx, x1, y1, x2, y2]
            
        Returns:
            Dictionary of ROI features for each scale
        """
        roi_features = {}
        
        for layer_id, feat in features.items():
            if layer_id in self.roi_aligns:
                roi_size = self.roi_sizes.get(layer_id, 28)  # Default to 28
                roi_features[layer_id] = self.roi_aligns[layer_id](
                    feat, rois, roi_size, roi_size
                )
                
        return roi_features


class HierarchicalFeatureFusion(nn.Module):
    """Fuse features with different ROI sizes hierarchically."""
    
    def __init__(
        self,
        input_channels: Dict[str, int],
        roi_sizes: Dict[str, int],
        output_channels: int = 256,
        target_size: int = 28
    ):
        """Initialize hierarchical fusion.
        
        Args:
            input_channels: Dictionary of input channels for each layer
            roi_sizes: Dictionary of ROI sizes for each layer
            output_channels: Number of output channels
            target_size: Target size for all features
        """
        super().__init__()
        
        self.roi_sizes = roi_sizes
        self.target_size = target_size
        
        # Create size adjustment and channel reduction for each layer
        self.size_adjusters = nn.ModuleDict()
        self.channel_reducers = nn.ModuleDict()
        
        for layer_id, in_channels in input_channels.items():
            roi_size = roi_sizes.get(layer_id, target_size)
            
            # Channel reduction
            self.channel_reducers[layer_id] = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, 1),
                LayerNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            
            # Size adjustment if needed
            if roi_size != target_size:
                if roi_size > target_size:
                    # Downsample larger ROIs
                    if roi_size == 56 and target_size == 28:
                        # 56 -> 28: use stride 2 convolution for exact 2x downsampling
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                    elif roi_size == 42 and target_size == 28:
                        # 42 -> 28: use learned downsampling with interpolation
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Conv2d(output_channels, output_channels * 2, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels * 2),
                            nn.ReLU(inplace=True),
                            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                    else:
                        # General case: use interpolation-based downsampling
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True),
                            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                else:
                    # Upsample smaller ROIs
                    self.size_adjusters[layer_id] = nn.Sequential(
                        nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                        nn.Conv2d(output_channels, output_channels, 3, padding=1),
                        LayerNorm2d(output_channels),
                        nn.ReLU(inplace=True)
                    )
                    
        # Fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(input_channels)))
        
        # Final fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            LayerNorm2d(output_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(output_channels)
        )
        
    def forward(self, roi_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse variable-sized ROI features.
        
        Args:
            roi_features: Dictionary of ROI features from different scales
            
        Returns:
            Fused feature tensor (N, C, H, W)
        """
        adjusted_features = []
        layer_ids = sorted(roi_features.keys())
        
        for layer_id in layer_ids:
            if layer_id in roi_features:
                # Apply channel reduction
                feat = self.channel_reducers[layer_id](roi_features[layer_id])
                
                # Apply size adjustment if needed
                if layer_id in self.size_adjusters:
                    feat = self.size_adjusters[layer_id](feat)
                    
                adjusted_features.append(feat)
                
        if not adjusted_features:
            raise ValueError("No features to fuse")
            
        # Weighted fusion
        weights = F.softmax(self.fusion_weights[:len(adjusted_features)], dim=0)
        stacked_features = torch.stack(adjusted_features, dim=0)
        weights_expanded = weights.view(-1, 1, 1, 1, 1)
        fused = (stacked_features * weights_expanded).sum(dim=0)
        
        # Final fusion processing
        fused = self.fusion_conv(fused)
        
        return fused


class VariableROISegmentationHead(nn.Module):
    """Segmentation head with variable ROI sizes."""
    
    def __init__(
        self,
        feature_channels: Dict[str, int],
        feature_strides: Dict[str, int],
        roi_sizes: Dict[str, int],
        num_classes: int = 3,
        mask_size: int = 56,
        mid_channels: int = 256
    ):
        """Initialize variable ROI segmentation head.
        
        Args:
            feature_channels: Channels for each feature layer
            feature_strides: Strides for each feature layer
            roi_sizes: ROI sizes for each layer
            num_classes: Number of output classes
            mask_size: Output mask size
            mid_channels: Intermediate channel count
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Variable ROI align
        self.var_roi_align = VariableROIAlign(
            feature_strides=feature_strides,
            roi_sizes=roi_sizes
        )
        
        # Hierarchical feature fusion
        self.feature_fusion = HierarchicalFeatureFusion(
            input_channels=feature_channels,
            roi_sizes=roi_sizes,
            output_channels=mid_channels,
            target_size=28  # Base size for fusion
        )
        
        # Enhanced decoder for better detail preservation
        self.decoder = nn.Sequential(
            # Initial processing
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            
            # First upsampling: 28x28 -> 56x56
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Refinement
            ResidualBlock(mid_channels),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            
            # Final prediction
            nn.Conv2d(mid_channels // 2, num_classes, 1)
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Multi-scale feature maps
            rois: ROI boxes
            
        Returns:
            Mask predictions (N, num_classes, mask_size, mask_size)
        """
        # Extract variable-sized ROI features
        roi_features = self.var_roi_align(features, rois)
        
        # Fuse features hierarchically
        fused_features = self.feature_fusion(roi_features)
        
        # Decode to masks
        masks = self.decoder(fused_features)
        
        return masks


def create_variable_roi_model(
    onnx_model_path: str,
    target_layers: List[str],
    roi_sizes: Dict[str, int],
    num_classes: int = 3,
    mask_size: int = 56,
    execution_provider: str = 'cpu'
) -> nn.Module:
    """Create variable ROI multi-scale model.
    
    Args:
        onnx_model_path: Path to YOLO ONNX model
        target_layers: List of layer IDs to extract
        roi_sizes: Dictionary mapping layer IDs to ROI sizes
        num_classes: Number of segmentation classes
        mask_size: Output mask size
        execution_provider: ONNX execution provider
        
    Returns:
        Complete model
    """
    from .multi_scale_extractor import MultiScaleYOLOFeatureExtractor
    
    # Feature configurations
    FEATURE_INFO = {
        'layer_3': {'channels': 256, 'stride': 4},    # 160x160
        'layer_19': {'channels': 256, 'stride': 4},   # 160x160
        'layer_5': {'channels': 512, 'stride': 8},    # 80x80
        'layer_22': {'channels': 512, 'stride': 8},   # 80x80
        'layer_34': {'channels': 1024, 'stride': 8}   # 80x80
    }
    
    # Create feature extractor
    extractor = MultiScaleYOLOFeatureExtractor(
        model_path=onnx_model_path,
        target_layers=target_layers,
        execution_provider=execution_provider
    )
    
    # Get feature info for selected layers
    feature_channels = {
        layer: FEATURE_INFO[layer]['channels'] 
        for layer in target_layers
    }
    feature_strides = {
        layer: FEATURE_INFO[layer]['stride'] 
        for layer in target_layers
    }
    
    # Create segmentation head
    segmentation_head = VariableROISegmentationHead(
        feature_channels=feature_channels,
        feature_strides=feature_strides,
        roi_sizes=roi_sizes,
        num_classes=num_classes,
        mask_size=mask_size
    )
    
    # Combine into complete model
    class VariableROIModel(nn.Module):
        def __init__(self, extractor, segmentation_head):
            super().__init__()
            self.extractor = extractor
            self.segmentation_head = segmentation_head
            
        def forward(self, images: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
            # Extract multi-scale features
            features = self.extractor.extract_features(images)
            
            # Generate masks
            masks = self.segmentation_head(features, rois)
            
            return masks
            
    return VariableROIModel(extractor, segmentation_head)