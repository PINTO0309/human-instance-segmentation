"""Test dynamic spatial scale calculation."""

import torch
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign
from src.human_edge_detection.advanced.dynamic_spatial_scale_mixin import DynamicSpatialScaleMixin


class TestModel(torch.nn.Module, DynamicSpatialScaleMixin):
    """Test model that uses dynamic spatial scaling."""
    
    def __init__(self):
        super().__init__()
        # Initialize with default square assumption
        self.roi_align = DynamicRoIAlign(
            spatial_scale=(640.0, 640.0),
            sampling_ratio=2,
            aligned=True
        )
        
    def forward(self, images, roi_boxes):
        # Dynamically update spatial scale based on input size
        self.update_roi_align_scales(images)
        
        # Extract ROIs
        return self.roi_align(images, roi_boxes, 64, 64)


def test_dynamic_scaling():
    """Test dynamic spatial scale updates."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Dynamic Spatial Scale ===\n")
    
    model = TestModel()
    
    # Test different image sizes
    test_cases = [
        ("Square 640x640", (1, 3, 640, 640)),
        ("Non-square 480x640", (1, 3, 480, 640)),
        ("Non-square 720x1280", (1, 3, 720, 1280)),
        ("Small 320x240", (1, 3, 320, 240))
    ]
    
    # Fixed ROI in normalized coordinates
    roi_norm = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32).to(device)
    
    for name, image_shape in test_cases:
        print(f"\nTest: {name}")
        print(f"  Image shape: {image_shape}")
        
        # Create dummy image
        images = torch.randn(image_shape).to(device)
        
        # Before update
        print(f"  Before update: scale_h={model.roi_align.spatial_scale_h}, "
              f"scale_w={model.roi_align.spatial_scale_w}")
        
        # Forward pass (will update scales)
        output = model(images, roi_norm)
        
        # After update
        print(f"  After update: scale_h={model.roi_align.spatial_scale_h}, "
              f"scale_w={model.roi_align.spatial_scale_w}")
        print(f"  Output shape: {output.shape}")
        
        # Calculate expected ROI pixel coordinates
        h, w = image_shape[2], image_shape[3]
        x1 = roi_norm[0, 1] * w
        y1 = roi_norm[0, 2] * h
        x2 = roi_norm[0, 3] * w
        y2 = roi_norm[0, 4] * h
        print(f"  ROI in pixels: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  ROI size: {(x2-x1):.1f}x{(y2-y1):.1f}")
    
    print("\n\n=== Testing with Feature Maps at Different Strides ===")
    
    # Test with conv5 features (stride 16)
    conv5_scale_h = 480.0 / 16  # 30
    conv5_scale_w = 640.0 / 16  # 40
    
    roi_align_conv5 = DynamicRoIAlign(
        spatial_scale=(conv5_scale_h, conv5_scale_w),
        sampling_ratio=2,
        aligned=True
    ).to(device)
    
    # Create conv5 feature map
    conv5_features = torch.randn(1, 512, 30, 40).to(device)
    
    print(f"\nConv5 features:")
    print(f"  Feature shape: {conv5_features.shape}")
    print(f"  Spatial scale: ({conv5_scale_h}, {conv5_scale_w})")
    
    # Extract ROI from conv5
    conv5_roi = roi_align_conv5(conv5_features, roi_norm, 14, 14)
    print(f"  Extracted ROI shape: {conv5_roi.shape}")


if __name__ == "__main__":
    test_dynamic_scaling()