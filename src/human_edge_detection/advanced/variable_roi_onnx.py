"""ONNX-exportable variable ROI model with fixed output sizes."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from torchvision.ops import roi_align

from ..model import LayerNorm2d, ResidualBlock


class ONNXVariableROIAlign(nn.Module):
    """ONNX-compatible variable ROI align with fixed output sizes."""
    
    def __init__(
        self,
        feature_strides: Dict[str, int],
        roi_sizes: Dict[str, int],
        sampling_ratio: int = 2
    ):
        """Initialize ONNX-compatible variable ROI align.
        
        Args:
            feature_strides: Dictionary mapping layer IDs to feature strides
            roi_sizes: Dictionary mapping layer IDs to ROI sizes
            sampling_ratio: Number of sampling points
        """
        super().__init__()
        
        self.feature_strides = feature_strides
        self.roi_sizes = roi_sizes
        self.sampling_ratio = sampling_ratio
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract ROI features with variable sizes using ONNX-compatible ops.
        
        Args:
            features: Dictionary of feature maps
            rois: ROI boxes (N, 5) - [batch_idx, x1, y1, x2, y2]
            
        Returns:
            Dictionary of ROI features for each scale
        """
        roi_features = {}
        
        for layer_id, feat in features.items():
            if layer_id in self.roi_sizes:
                roi_size = self.roi_sizes[layer_id]
                spatial_scale = 1.0 / self.feature_strides[layer_id]
                
                # Use torchvision's roi_align which is ONNX-compatible
                aligned_feat = roi_align(
                    feat,
                    rois,
                    output_size=(roi_size, roi_size),
                    spatial_scale=spatial_scale,
                    sampling_ratio=self.sampling_ratio,
                    aligned=True
                )
                
                roi_features[layer_id] = aligned_feat
                
        return roi_features


class ONNXHierarchicalFeatureFusion(nn.Module):
    """ONNX-compatible hierarchical feature fusion."""
    
    def __init__(
        self,
        input_channels: Dict[str, int],
        roi_sizes: Dict[str, int],
        output_channels: int = 256,
        target_size: int = 28
    ):
        """Initialize ONNX-compatible hierarchical fusion.
        
        Args:
            input_channels: Dictionary of input channels for each layer
            roi_sizes: Dictionary of ROI sizes for each layer
            output_channels: Number of output channels
            target_size: Target size for all features
        """
        super().__init__()
        
        self.roi_sizes = roi_sizes
        self.target_size = target_size
        self.layer_names = sorted(input_channels.keys())  # Fixed order
        
        # Create channel reduction for each layer
        self.channel_reducers = nn.ModuleDict()
        
        for layer_id, in_channels in input_channels.items():
            self.channel_reducers[layer_id] = nn.Sequential(
                nn.Conv2d(in_channels, output_channels, 1),
                LayerNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            
        # Create size adjustment layers
        self.size_adjusters = nn.ModuleDict()
        
        for layer_id, roi_size in roi_sizes.items():
            if roi_size != target_size:
                if roi_size > target_size:
                    # Downsample using strided convolution for ONNX compatibility
                    # Calculate appropriate stride and kernel size
                    if roi_size == 56 and target_size == 28:
                        # 56 -> 28: use stride 2
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=2, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                    elif roi_size == 42 and target_size == 28:
                        # 42 -> 28: use interpolation with fixed size
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                    else:
                        # General case: use interpolation
                        self.size_adjusters[layer_id] = nn.Sequential(
                            nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                            LayerNorm2d(output_channels),
                            nn.ReLU(inplace=True)
                        )
                else:
                    # Upsample using interpolate
                    self.size_adjusters[layer_id] = nn.Sequential(
                        nn.Upsample(size=target_size, mode='bilinear', align_corners=False),
                        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
                        LayerNorm2d(output_channels),
                        nn.ReLU(inplace=True)
                    )
                    
        # Fusion weights as a linear layer for ONNX compatibility
        self.fusion_linear = nn.Linear(len(input_channels), len(input_channels))
        
        # Final fusion layers
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            LayerNorm2d(output_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(output_channels)
        )
        
    def forward(self, roi_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse variable-sized ROI features in ONNX-compatible way.
        
        Args:
            roi_features: Dictionary of ROI features from different scales
            
        Returns:
            Fused feature tensor (N, C, H, W)
        """
        adjusted_features = []
        
        # Process in fixed order for ONNX
        for layer_id in self.layer_names:
            if layer_id in roi_features:
                # Apply channel reduction
                feat = self.channel_reducers[layer_id](roi_features[layer_id])
                
                # Apply size adjustment if needed
                if layer_id in self.size_adjusters:
                    feat = self.size_adjusters[layer_id](feat)
                    
                adjusted_features.append(feat)
                
        if not adjusted_features:
            raise ValueError("No features to fuse")
            
        # Stack features
        batch_size = adjusted_features[0].shape[0]
        num_features = len(adjusted_features)
        
        # Weighted fusion using linear layer (ONNX-compatible)
        stacked = torch.stack(adjusted_features, dim=1)  # (N, F, C, H, W)
        
        # Compute fusion weights
        ones = torch.ones(batch_size, num_features).to(stacked.device)
        weights = F.softmax(self.fusion_linear(ones), dim=1)  # (N, F)
        
        # Apply weights
        weights = weights.view(batch_size, num_features, 1, 1, 1)
        fused = (stacked * weights).sum(dim=1)  # (N, C, H, W)
        
        # Final fusion processing
        fused = self.fusion_conv(fused)
        
        return fused


class ONNXVariableROISegmentationHead(nn.Module):
    """ONNX-exportable segmentation head with variable ROI sizes."""
    
    def __init__(
        self,
        feature_channels: Dict[str, int],
        feature_strides: Dict[str, int],
        roi_sizes: Dict[str, int],
        num_classes: int = 3,
        mask_size: int = 56,
        mid_channels: int = 256
    ):
        """Initialize ONNX-exportable variable ROI segmentation head.
        
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
        
        # ONNX-compatible variable ROI align
        self.var_roi_align = ONNXVariableROIAlign(
            feature_strides=feature_strides,
            roi_sizes=roi_sizes
        )
        
        # ONNX-compatible hierarchical feature fusion
        self.feature_fusion = ONNXHierarchicalFeatureFusion(
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


def create_onnx_variable_roi_segmentation_head(
    feature_channels: Dict[str, int],
    feature_strides: Dict[str, int],
    roi_sizes: Dict[str, int],
    num_classes: int = 3,
    mask_size: int = 56
) -> nn.Module:
    """Create ONNX-exportable variable ROI segmentation head.
    
    Args:
        feature_channels: Channels for each feature layer
        feature_strides: Strides for each feature layer
        roi_sizes: ROI sizes for each layer
        num_classes: Number of output classes
        mask_size: Output mask size
        
    Returns:
        ONNX-exportable segmentation head
    """
    return ONNXVariableROISegmentationHead(
        feature_channels=feature_channels,
        feature_strides=feature_strides,
        roi_sizes=roi_sizes,
        num_classes=num_classes,
        mask_size=mask_size
    )