"""ROI-based instance segmentation model with enhanced architecture.

This model features:
- Larger ROI size (28x28) for better initial resolution
- Residual blocks for deeper feature extraction
- Progressive upsampling with multi-scale fusion
- Smooth mask generation with reduced artifacts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dynamic_roi_align import DynamicRoIAlign
from typing import Dict, Optional, Tuple
from .advanced.normalization_comparison import get_normalization_layer


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


class ResidualBlock(nn.Module):
    """Residual block with configurable normalization."""

    def __init__(self, channels, normalization_type: str = 'layernorm2d', normalization_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = get_normalization_layer(normalization_type, channels, num_groups=min(normalization_groups, channels))
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = get_normalization_layer(normalization_type, channels, num_groups=min(normalization_groups, channels))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out


class ROISegmentationHead(nn.Module):
    """ROI-based segmentation head with enhanced upsampling and FPN-style features."""

    def __init__(
        self,
        in_channels: int = 1024,
        mid_channels: int = 256,
        num_classes: int = 3,
        roi_size: int = 28,  # Enhanced spatial resolution
        mask_size: int = 56,
        feature_stride: int = 8,
        normalization_type: str = 'layernorm2d',
        normalization_groups: int = 8
    ):
        """Initialize improved segmentation head.

        Args:
            in_channels: Number of input channels from YOLO features
            mid_channels: Number of channels in intermediate layers
            num_classes: Number of output classes (3: bg, target, non-target)
            roi_size: Size of ROI after ROIAlign (increased to 28)
            mask_size: Output mask size
            feature_stride: Stride of feature map relative to input image
        """
        super().__init__()

        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.mask_size = mask_size
        self.feature_stride = feature_stride
        self.normalization_type = normalization_type
        self.normalization_groups = normalization_groups

        # For future extension to non-square ROIs
        if isinstance(roi_size, int):
            self.roi_height = roi_size
            self.roi_width = roi_size
        else:
            self.roi_height, self.roi_width = roi_size

        # DynamicROIAlign layer with larger output
        self.roi_align = DynamicRoIAlign(
            spatial_scale=1.0 / feature_stride,
            sampling_ratio=4,  # Increased sampling for better quality
            aligned=True
        )

        # Initial feature processing with residual blocks
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1),
            get_normalization_layer(normalization_type, mid_channels, num_groups=min(normalization_groups, mid_channels)),
            nn.ReLU(inplace=True)
        )

        # Feature processing with residual connections
        self.res_block1 = ResidualBlock(mid_channels, normalization_type, normalization_groups)
        self.res_block2 = ResidualBlock(mid_channels, normalization_type, normalization_groups)

        # Progressive upsampling with intermediate supervision
        # 28x28 -> 56x56
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels, 4, stride=2, padding=1),
            get_normalization_layer(normalization_type, mid_channels, num_groups=min(normalization_groups, mid_channels)),
            nn.ReLU(inplace=True)
        )

        # Refinement at 56x56
        self.refine1 = ResidualBlock(mid_channels, normalization_type, normalization_groups)

        # 56x56 -> 112x112 (oversample)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 4, stride=2, padding=1),
            get_normalization_layer(normalization_type, mid_channels // 2, num_groups=min(normalization_groups, mid_channels // 2)),
            nn.ReLU(inplace=True)
        )

        # Refinement at 112x112
        self.refine2 = nn.Sequential(
            nn.Conv2d(mid_channels // 2, mid_channels // 2, 3, padding=1),
            get_normalization_layer(normalization_type, mid_channels // 2, num_groups=min(normalization_groups, mid_channels // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels // 2, 3, padding=1),
            get_normalization_layer(normalization_type, mid_channels // 2, num_groups=min(normalization_groups, mid_channels // 2)),
            nn.ReLU(inplace=True)
        )

        # Final projection and downsampling to 56x56
        self.final_conv = nn.Sequential(
            nn.Conv2d(mid_channels // 2, mid_channels // 4, 3, padding=1),
            get_normalization_layer(normalization_type, mid_channels // 4, num_groups=min(normalization_groups, mid_channels // 4)),
            nn.ReLU(inplace=True)
        )

        # Multi-scale feature fusion
        self.fusion = nn.Conv2d(mid_channels + mid_channels // 4, mid_channels // 2, 1)

        # Final classifier
        self.classifier = nn.Conv2d(mid_channels // 2, num_classes, 1)

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
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: YOLO features of shape (B, C, H, W)
            rois: ROI coordinates of shape (N, 5) where each row is [batch_idx, x1, y1, x2, y2]
                  Coordinates should be in the original image space

        Returns:
            Mask logits of shape (N, num_classes, mask_size, mask_size)
        """
        # Extract ROI features using DynamicRoIAlign (28x28)
        roi_features = self.roi_align(features, rois, self.roi_height, self.roi_width)

        # Initial processing
        x = self.conv_in(roi_features)

        # Deep feature extraction with residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Progressive upsampling
        # 28x28 -> 56x56
        x_56 = self.upsample1(x)
        x_56 = self.refine1(x_56)

        # 56x56 -> 112x112
        x_112 = self.upsample2(x_56)
        x_112 = self.refine2(x_112)

        # Project to lower channels
        x_112_proj = self.final_conv(x_112)

        # Downsample back to 56x56 for fusion
        x_112_down = F.interpolate(x_112_proj, size=(56, 56), mode='bilinear', align_corners=False)

        # Multi-scale fusion
        x_fused = torch.cat([x_56, x_112_down], dim=1)
        x_fused = self.fusion(x_fused)

        # Final classification
        logits = self.classifier(x_fused)

        return logits


class ROISegmentationModel(nn.Module):
    """Complete ROI-based instance segmentation model."""

    def __init__(
        self,
        segmentation_head: ROISegmentationHead,
        feature_extractor: Optional[nn.Module] = None
    ):
        """Initialize model.

        Args:
            segmentation_head: Segmentation head module
            feature_extractor: Optional feature extractor (if None, features are provided externally)
        """
        super().__init__()

        self.feature_extractor = feature_extractor
        self.segmentation_head = segmentation_head

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        rois: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images (B, 3, H, W) - required if feature_extractor is set
            features: Pre-extracted features (B, C, H, W) - required if feature_extractor is None
            rois: ROI coordinates (N, 5) format: [batch_idx, x1, y1, x2, y2]

        Returns:
            Dictionary containing:
                - 'masks': Predicted mask logits (N, num_classes, H, W)
                - 'features': Extracted features (B, C, H, W)
        """
        # Extract features if needed
        if self.feature_extractor is not None:
            assert images is not None, "Images required when using feature extractor"
            features = self.feature_extractor(images)
        else:
            assert features is not None, "Features required when not using feature extractor"

        # Get segmentation masks
        masks = self.segmentation_head(features, rois)

        return {
            'masks': masks,
            'features': features
        }


class ROIBatchProcessor:
    """Helper class to process multiple ROIs per image in a batch."""

    @staticmethod
    def prepare_rois_for_batch(
        roi_coords_batch: torch.Tensor,
        image_size: Tuple[int, int] = (640, 640)
    ) -> torch.Tensor:
        """Convert normalized ROI coordinates to format expected by ROIAlign.

        Args:
            roi_coords_batch: Normalized ROI coordinates (B, 4) in format [x1, y1, x2, y2]
                             Values should be in range [0, 1]
            image_size: Size of input images (H, W)

        Returns:
            ROIs in format (B, 5) where each row is [batch_idx, x1, y1, x2, y2]
            with coordinates in pixel space
        """
        batch_size = roi_coords_batch.shape[0]
        device = roi_coords_batch.device

        # Create batch indices
        batch_indices = torch.arange(batch_size, device=device).float().unsqueeze(1)

        # Convert normalized coordinates to pixel coordinates
        roi_coords_pixel = roi_coords_batch.clone()
        roi_coords_pixel[:, [0, 2]] *= image_size[1]  # x coordinates
        roi_coords_pixel[:, [1, 3]] *= image_size[0]  # y coordinates

        # Concatenate batch indices with coordinates
        rois = torch.cat([batch_indices, roi_coords_pixel], dim=1)

        return rois


def create_model(
    num_classes: int = 3,
    in_channels: int = 1024,
    mid_channels: int = 256,
    mask_size: int = 56,
    roi_size: int = 28,  # Increased default
    normalization_type: str = 'layernorm2d',
    normalization_groups: int = 8
) -> ROISegmentationModel:
    """Create ROI segmentation model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels from features
        mid_channels: Number of channels in intermediate layers
        mask_size: Output mask size
        roi_size: ROI size after ROIAlign (default 28)

    Returns:
        ROISegmentationModel instance
    """
    segmentation_head = ROISegmentationHead(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_classes=num_classes,
        mask_size=mask_size,
        roi_size=roi_size,
        normalization_type=normalization_type,
        normalization_groups=normalization_groups
    )

    model = ROISegmentationModel(
        segmentation_head=segmentation_head,
        feature_extractor=None  # Features will be provided externally
    )

    return model


if __name__ == "__main__":
    # Test improved model
    print("Testing ROI Segmentation Model...")

    # Create model
    model = create_model()
    model.eval()

    # Create dummy inputs
    batch_size = 2
    features = torch.randn(batch_size, 1024, 80, 80)

    # Create dummy ROIs (normalized coordinates)
    roi_coords = torch.tensor([
        [0.1, 0.1, 0.5, 0.5],  # First ROI
        [0.3, 0.3, 0.8, 0.8]   # Second ROI
    ])

    # Prepare ROIs for model
    rois = ROIBatchProcessor.prepare_rois_for_batch(roi_coords)
    print(f"ROIs shape: {rois.shape}")

    # Forward pass
    with torch.no_grad():
        output = model(features=features, rois=rois)

    print(f"\nModel output:")
    print(f"Masks shape: {output['masks'].shape}")
    print(f"Expected shape: (2, 3, 56, 56)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")