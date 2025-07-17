"""Alternative ROI-based segmentation model using LayerNorm instead of GroupNorm."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from typing import Dict, Optional, Tuple


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (B, C, H, W)."""
    
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.eps = eps
    
    def forward(self, x):
        # Calculate mean and variance across C, H, W dimensions
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        
        # Normalize
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        x = x * self.weight + self.bias
        
        return x


class ROISegmentationHeadLayerNorm(nn.Module):
    """ROI-based segmentation head using LayerNorm for better ONNX compatibility."""

    def __init__(
        self,
        in_channels: int = 1024,
        mid_channels: int = 256,
        num_classes: int = 3,
        roi_size: int = 14,
        mask_size: int = 56,
        feature_stride: int = 8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_size = mask_size
        self.feature_stride = feature_stride

        # ROIAlign layer
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / feature_stride,
            sampling_ratio=2,
            aligned=True
        )

        # Feature processing blocks with LayerNorm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Upsampling blocks with LayerNorm
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Additional conv for refinement
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Final classifier
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        # Extract ROI features
        roi_features = self.roi_align(features, rois)

        # Process features
        x = self.conv1(roi_features)
        x = self.conv2(x)

        # Upsample to mask size
        x = self.upsample1(x)  # 14x14 -> 28x28
        x = self.upsample2(x)  # 28x28 -> 56x56

        # Refine
        x = self.conv3(x)

        # Classify
        logits = self.classifier(x)

        return logits


# Alternative: Simple normalization without learnable parameters
class SimpleNorm2d(nn.Module):
    """Simple normalization without learnable parameters for maximum ONNX compatibility."""
    
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        # Normalize across spatial dimensions for each channel
        # Shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Reshape to (B*C, H*W)
        x_reshaped = x.view(B * C, H * W)
        
        # Calculate statistics
        mean = x_reshaped.mean(dim=1, keepdim=True)
        var = x_reshaped.var(dim=1, keepdim=True, unbiased=False)
        
        # Normalize
        x_normalized = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        
        # Reshape back
        x_normalized = x_normalized.view(B, C, H, W)
        
        return x_normalized


class ROISegmentationHeadSimpleNorm(nn.Module):
    """ROI-based segmentation head using simple normalization for maximum ONNX compatibility."""

    def __init__(
        self,
        in_channels: int = 1024,
        mid_channels: int = 256,
        num_classes: int = 3,
        roi_size: int = 14,
        mask_size: int = 56,
        feature_stride: int = 8
    ):
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_size = mask_size
        self.feature_stride = feature_stride

        # ROIAlign layer
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / feature_stride,
            sampling_ratio=2,
            aligned=True
        )

        # Feature processing blocks with simple normalization
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.norm1 = SimpleNorm2d()
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.norm2 = SimpleNorm2d()

        # Upsampling blocks
        self.upsample1 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2)
        self.norm3 = SimpleNorm2d()
        
        self.upsample2 = nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2)
        self.norm4 = SimpleNorm2d()

        # Additional conv for refinement
        self.conv3 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.norm5 = SimpleNorm2d()

        # Final classifier
        self.classifier = nn.Conv2d(mid_channels, num_classes, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        # Extract ROI features
        roi_features = self.roi_align(features, rois)

        # Process features
        x = self.conv1(roi_features)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x, inplace=True)

        # Upsample to mask size
        x = self.upsample1(x)
        x = self.norm3(x)
        x = F.relu(x, inplace=True)
        
        x = self.upsample2(x)
        x = self.norm4(x)
        x = F.relu(x, inplace=True)

        # Refine
        x = self.conv3(x)
        x = self.norm5(x)
        x = F.relu(x, inplace=True)

        # Classify
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    # Test both alternatives
    print("Testing alternative normalization approaches...")
    
    # Test LayerNorm version
    print("\n1. Testing LayerNorm version:")
    model_ln = ROISegmentationHeadLayerNorm()
    model_ln.eval()
    
    # Test SimpleNorm version
    print("\n2. Testing SimpleNorm version:")
    model_sn = ROISegmentationHeadSimpleNorm()
    model_sn.eval()
    
    # Create dummy inputs
    features = torch.randn(2, 1024, 80, 80)
    rois = torch.tensor([
        [0, 100, 100, 300, 300],
        [1, 150, 150, 350, 350]
    ], dtype=torch.float32)
    
    # Test forward pass
    with torch.no_grad():
        output_ln = model_ln(features, rois)
        output_sn = model_sn(features, rois)
        
    print(f"\nLayerNorm output shape: {output_ln.shape}")
    print(f"SimpleNorm output shape: {output_sn.shape}")
    print(f"Expected shape: (2, 3, 56, 56)")