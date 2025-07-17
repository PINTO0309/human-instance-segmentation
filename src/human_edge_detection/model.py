"""ROI-based instance segmentation model."""

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


class ROISegmentationHead(nn.Module):
    """ROI-based segmentation head for 3-class instance segmentation.

    Takes YOLO features and ROI coordinates to produce segmentation masks.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        mid_channels: int = 256,
        num_classes: int = 3,
        roi_size: int = 14,
        mask_size: int = 56,
        feature_stride: int = 8
    ):
        """Initialize segmentation head.

        Args:
            in_channels: Number of input channels from YOLO features
            mid_channels: Number of channels in intermediate layers
            num_classes: Number of output classes (3: bg, target, non-target)
            roi_size: Size of ROI after ROIAlign
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

        # ROIAlign layer
        self.roi_align = RoIAlign(
            output_size=(roi_size, roi_size),
            spatial_scale=1.0 / feature_stride,
            sampling_ratio=2,
            aligned=True
        )

        # Feature processing blocks
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

        # Upsampling blocks
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


class ROISegmentationModel(nn.Module):
    """Complete ROI-based instance segmentation model.

    Combines feature extraction and segmentation head.
    """

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
    mask_size: int = 56
) -> ROISegmentationModel:
    """Create ROI segmentation model.

    Args:
        num_classes: Number of output classes
        in_channels: Number of input channels from features
        mid_channels: Number of channels in intermediate layers
        mask_size: Output mask size

    Returns:
        ROISegmentationModel instance
    """
    segmentation_head = ROISegmentationHead(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_classes=num_classes,
        mask_size=mask_size
    )

    model = ROISegmentationModel(
        segmentation_head=segmentation_head,
        feature_extractor=None  # Features will be provided externally
    )

    return model


if __name__ == "__main__":
    # Test model
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
    print(f"ROIs: {rois}")

    # Forward pass
    with torch.no_grad():
        output = model(features=features, rois=rois)

    print(f"\nModel output:")
    print(f"Masks shape: {output['masks'].shape}")
    print(f"Expected shape: (2, 3, 56, 56)")

    # Test individual components
    print("\nTesting segmentation head directly...")
    seg_head = model.segmentation_head
    masks = seg_head(features, rois)
    print(f"Direct segmentation head output shape: {masks.shape}")