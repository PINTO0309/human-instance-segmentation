"""Multi-scale ROI segmentation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from ..dynamic_roi_align import DynamicRoIAlign
from ..model import LayerNorm2d, ResidualBlock


class MultiScaleRoIAlign(nn.Module):
    """Extract ROI features from multiple scale feature maps."""
    
    def __init__(
        self,
        feature_strides: Dict[str, int],
        roi_size: int = 28,
        sampling_ratio: int = 2
    ):
        """Initialize multi-scale ROI align.
        
        Args:
            feature_strides: Dictionary mapping layer IDs to feature strides
            roi_size: Output ROI size
            sampling_ratio: Number of sampling points
        """
        super().__init__()
        
        self.feature_strides = feature_strides
        self.roi_size = roi_size
        
        # Create ROI align for each scale
        self.roi_aligns = nn.ModuleDict()
        
        for layer_id, stride in feature_strides.items():
            self.roi_aligns[layer_id] = DynamicRoIAlign(
                spatial_scale=1.0 / stride,
                output_size=(roi_size, roi_size),
                sampling_ratio=sampling_ratio,
                aligned=True
            )
            
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract ROI features from multiple scales.
        
        Args:
            features: Dictionary of feature maps
            rois: ROI boxes (N, 5) - [batch_idx, x1, y1, x2, y2]
            
        Returns:
            Dictionary of ROI features for each scale
        """
        roi_features = {}
        
        for layer_id, feat in features.items():
            if layer_id in self.roi_aligns:
                roi_features[layer_id] = self.roi_aligns[layer_id](feat, rois)
                
        return roi_features
    
    def assign_rois_to_levels(
        self,
        rois: torch.Tensor,
        canonical_scale: int = 224,
        canonical_level: int = 4
    ) -> Dict[str, torch.Tensor]:
        """Assign ROIs to FPN levels based on their size.
        
        Args:
            rois: ROI boxes (N, 5)
            canonical_scale: Canonical box scale
            canonical_level: Canonical FPN level
            
        Returns:
            Dictionary mapping layer IDs to assigned ROI indices
        """
        # Calculate ROI areas
        areas = (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2])
        
        # Assign to levels based on area
        levels = torch.floor(canonical_level + torch.log2(
            torch.sqrt(areas) / canonical_scale + 1e-6
        ))
        
        # Map levels to layer IDs
        level_assignments = {}
        stride_to_level = {4: 2, 8: 3, 16: 4, 32: 5}
        
        for layer_id, stride in self.feature_strides.items():
            level = stride_to_level.get(stride, 3)
            mask = (levels == level)
            if mask.any():
                level_assignments[layer_id] = torch.where(mask)[0]
                
        return level_assignments


class MultiScaleFeatureFusion(nn.Module):
    """Fuse ROI features from multiple scales."""
    
    def __init__(
        self,
        input_channels: Dict[str, int],
        output_channels: int = 256,
        fusion_method: str = 'adaptive'
    ):
        """Initialize feature fusion module.
        
        Args:
            input_channels: Dictionary mapping layer IDs to channel counts
            output_channels: Output channel count
            fusion_method: Fusion method ('adaptive', 'concat', 'sum')
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fusion_method = fusion_method
        
        # Channel reduction layers
        self.channel_reducers = nn.ModuleDict()
        
        for layer_id, channels in input_channels.items():
            self.channel_reducers[layer_id] = nn.Sequential(
                nn.Conv2d(channels, output_channels, 1),
                LayerNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            
        if fusion_method == 'adaptive':
            # Learnable fusion weights
            self.fusion_weights = nn.Parameter(
                torch.ones(len(input_channels)) / len(input_channels)
            )
            
        elif fusion_method == 'concat':
            # Projection after concatenation
            total_channels = output_channels * len(input_channels)
            self.fusion_proj = nn.Sequential(
                nn.Conv2d(total_channels, output_channels, 1),
                LayerNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            
    def forward(self, roi_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multi-scale ROI features.
        
        Args:
            roi_features: Dictionary of ROI features from different scales
            
        Returns:
            Fused feature tensor (N, C, H, W)
        """
        # Apply channel reduction
        reduced_features = []
        layer_ids = sorted(roi_features.keys())  # Ensure consistent ordering
        
        for layer_id in layer_ids:
            if layer_id in roi_features:
                feat = self.channel_reducers[layer_id](roi_features[layer_id])
                reduced_features.append(feat)
                
        if not reduced_features:
            raise ValueError("No features to fuse")
            
        # Apply fusion
        if self.fusion_method == 'adaptive':
            # Weighted sum with learnable weights
            weights = F.softmax(self.fusion_weights[:len(reduced_features)], dim=0)
            fused = sum(w * f for w, f in zip(weights, reduced_features))
            
        elif self.fusion_method == 'concat':
            # Concatenate and project
            fused = torch.cat(reduced_features, dim=1)
            fused = self.fusion_proj(fused)
            
        elif self.fusion_method == 'sum':
            # Simple sum
            fused = sum(reduced_features)
            
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
        return fused


class MultiScaleROISegmentationHead(nn.Module):
    """Multi-scale ROI segmentation head."""
    
    def __init__(
        self,
        feature_channels: Dict[str, int],
        feature_strides: Dict[str, int],
        num_classes: int = 3,
        roi_size: int = 28,
        mask_size: int = 56,
        mid_channels: int = 256,
        fusion_method: str = 'adaptive'
    ):
        """Initialize multi-scale segmentation head.
        
        Args:
            feature_channels: Channels for each feature layer
            feature_strides: Strides for each feature layer
            num_classes: Number of output classes
            roi_size: ROI size after ROIAlign
            mask_size: Output mask size
            mid_channels: Intermediate channel count
            fusion_method: Feature fusion method
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_size = mask_size
        
        # Multi-scale ROI align
        self.ms_roi_align = MultiScaleRoIAlign(
            feature_strides=feature_strides,
            roi_size=roi_size
        )
        
        # Feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(
            input_channels=feature_channels,
            output_channels=mid_channels,
            fusion_method=fusion_method
        )
        
        # Decoder (similar to original but with multi-scale input)
        self.decoder = nn.Sequential(
            # Initial processing
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            
            # First upsampling: 28x28 -> 56x56
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            
            # Mid-resolution processing
            ResidualBlock(mid_channels),
            
            # Second upsampling: 56x56 -> 112x112
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            
            # Fine-detail processing
            nn.Conv2d(mid_channels // 2, mid_channels // 2, 3, padding=1),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            
            # Downsample back to target size with detail preservation
            nn.Conv2d(mid_channels // 2, mid_channels // 4, 3, stride=2, padding=1),
            LayerNorm2d(mid_channels // 4),
            nn.ReLU(inplace=True),
            
            # Final classification
            nn.Conv2d(mid_channels // 4, num_classes, 1)
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        rois: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            features: Multi-scale feature dictionary
            rois: ROI boxes (N, 5)
            
        Returns:
            Segmentation logits (N, num_classes, mask_size, mask_size)
        """
        # Extract multi-scale ROI features
        roi_features = self.ms_roi_align(features, rois)
        
        # Fuse features
        fused_features = self.feature_fusion(roi_features)
        
        # Decode to masks
        masks = self.decoder(fused_features)
        
        return masks


class MultiScaleSegmentationModel(nn.Module):
    """Complete multi-scale segmentation model."""
    
    def __init__(
        self,
        feature_extractor,
        feature_specs: Dict[str, dict],
        num_classes: int = 3,
        roi_size: int = 28,
        mask_size: int = 56,
        fusion_method: str = 'adaptive'
    ):
        """Initialize model.
        
        Args:
            feature_extractor: Multi-scale feature extractor
            feature_specs: Feature specifications
            num_classes: Number of classes
            roi_size: ROI size
            mask_size: Output mask size
            fusion_method: Feature fusion method
        """
        super().__init__()
        
        self.feature_extractor = feature_extractor
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
        images: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            images: Input images (B, 3, H, W)
            rois: ROI boxes (N, 5)
            
        Returns:
            Segmentation masks (N, num_classes, mask_size, mask_size)
        """
        # Extract multi-scale features
        features = self.feature_extractor.extract_features(images)
        
        # Generate masks
        masks = self.segmentation_head(features, rois)
        
        return masks
    
    def get_trainable_parameters(self):
        """Get only trainable parameters (excluding feature extractor)."""
        return self.segmentation_head.parameters()


def create_multiscale_model(
    onnx_model_path: str,
    target_layers: List[str] = ['layer_3', 'layer_22', 'layer_34'],
    num_classes: int = 3,
    roi_size: int = 28,
    mask_size: int = 56,
    fusion_method: str = 'adaptive',
    execution_provider: str = 'cuda'
) -> MultiScaleSegmentationModel:
    """Create a multi-scale segmentation model.
    
    Args:
        onnx_model_path: Path to YOLO ONNX model
        target_layers: Layers to extract features from
        num_classes: Number of segmentation classes
        roi_size: ROI size after alignment
        mask_size: Output mask size
        fusion_method: Feature fusion method
        execution_provider: ONNX execution provider
        
    Returns:
        Initialized model
    """
    from .multi_scale_extractor import MultiScaleYOLOFeatureExtractor
    
    # Create feature extractor
    feature_extractor = MultiScaleYOLOFeatureExtractor(
        model_path=onnx_model_path,
        target_layers=target_layers,
        execution_provider=execution_provider
    )
    
    # Get feature specs for target layers
    feature_specs = {
        lid: feature_extractor.FEATURE_SPECS[lid]
        for lid in target_layers
    }
    
    # Create model
    model = MultiScaleSegmentationModel(
        feature_extractor=feature_extractor,
        feature_specs=feature_specs,
        num_classes=num_classes,
        roi_size=roi_size,
        mask_size=mask_size,
        fusion_method=fusion_method
    )
    
    return model


if __name__ == "__main__":
    # Test multi-scale model
    print("Testing MultiScaleSegmentationModel...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create dummy data
    batch_size = 2
    num_rois = 5
    images = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # Create dummy ROIs
    rois = []
    for i in range(num_rois):
        batch_idx = i % batch_size
        x1, y1 = torch.randint(0, 400, (2,))
        x2, y2 = x1 + torch.randint(50, 200, (1,)), y1 + torch.randint(50, 200, (1,))
        rois.append([batch_idx, x1, y1, x2, y2])
    rois = torch.tensor(rois, dtype=torch.float32).to(device)
    
    # Test model creation (will fail without ONNX model)
    try:
        model = create_multiscale_model(
            onnx_model_path="ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx",
            target_layers=['layer_3', 'layer_22', 'layer_34'],
            fusion_method='adaptive'
        )
        model = model.to(device)
        model.eval()
        
        # Test forward pass
        with torch.no_grad():
            masks = model(images, rois)
            
        print(f"Output shape: {masks.shape}")
        print(f"Expected: ({num_rois}, 3, 56, 56)")
        
    except Exception as e:
        print(f"Test failed (expected without model file): {e}")