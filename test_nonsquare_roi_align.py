"""Test DynamicRoIAlign with non-square images."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign


def test_nonsquare_roi_align():
    """Test ROI align with non-square images."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Non-Square ROI Align ===\n")
    
    # Test configurations
    test_cases = [
        {
            "name": "Square image (640x640)",
            "image_size": (640, 640),
            "roi_norm": [0.2, 0.2, 0.8, 0.8],  # Normalized coordinates
            "spatial_scale": (640.0, 640.0)
        },
        {
            "name": "Non-square image (480x640)",
            "image_size": (480, 640),
            "roi_norm": [0.2, 0.2, 0.8, 0.8],  # Same normalized coordinates
            "spatial_scale": (480.0, 640.0)
        },
        {
            "name": "Non-square image (640x480)",
            "image_size": (640, 480),
            "roi_norm": [0.2, 0.2, 0.8, 0.8],  # Same normalized coordinates
            "spatial_scale": (640.0, 480.0)
        }
    ]
    
    fig, axes = plt.subplots(len(test_cases), 3, figsize=(15, 5*len(test_cases)))
    if len(test_cases) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, test_case in enumerate(test_cases):
        print(f"\nTest case: {test_case['name']}")
        H, W = test_case['image_size']
        
        # Create test image with grid pattern
        img = torch.zeros(1, 3, H, W).to(device)
        # Add grid pattern
        grid_size = 40
        for i in range(0, H, grid_size):
            img[0, 0, i:i+2, :] = 1.0  # Red horizontal lines
        for j in range(0, W, grid_size):
            img[0, 1, :, j:j+2] = 1.0  # Green vertical lines
        
        # Add center marker
        center_h, center_w = H//2, W//2
        img[0, 2, center_h-20:center_h+20, center_w-20:center_w+20] = 1.0  # Blue square
        
        # Create ROI (normalized coordinates)
        roi_norm = torch.tensor([[0] + test_case['roi_norm']], dtype=torch.float32).to(device)
        
        # Calculate pixel coordinates for visualization
        x1_px = (roi_norm[0, 1] * W).item()
        y1_px = (roi_norm[0, 2] * H).item()
        x2_px = (roi_norm[0, 3] * W).item()
        y2_px = (roi_norm[0, 4] * H).item()
        
        print(f"  Image size: {H}x{W}")
        print(f"  ROI normalized: {roi_norm[0, 1:].tolist()}")
        print(f"  ROI pixels: [{x1_px:.1f}, {y1_px:.1f}, {x2_px:.1f}, {y2_px:.1f}]")
        print(f"  ROI size: {x2_px-x1_px:.1f}x{y2_px-y1_px:.1f}")
        print(f"  Spatial scale: {test_case['spatial_scale']}")
        
        # Show original image
        img_np = img[0].cpu().permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(img_np)
        rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px,
                        linewidth=2, edgecolor='yellow', facecolor='none')
        axes[idx, 0].add_patch(rect)
        axes[idx, 0].set_title(f'{test_case["name"]}\\nOriginal with ROI')
        axes[idx, 0].set_xlabel(f'Width: {W}')
        axes[idx, 0].set_ylabel(f'Height: {H}')
        
        # Test with old single-scale (will be wrong for non-square)
        roi_align_single = DynamicRoIAlign(
            spatial_scale=640.0,  # Single scale assumes square
            sampling_ratio=2,
            aligned=True
        ).to(device)
        
        extracted_single = roi_align_single(img, roi_norm, 64, 64)
        axes[idx, 1].imshow(extracted_single[0].cpu().permute(1, 2, 0).numpy())
        axes[idx, 1].set_title('Single scale=640\\n(Wrong for non-square)')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Test with correct tuple scale
        roi_align_tuple = DynamicRoIAlign(
            spatial_scale=test_case['spatial_scale'],
            sampling_ratio=2,
            aligned=True
        ).to(device)
        
        extracted_tuple = roi_align_tuple(img, roi_norm, 64, 64)
        axes[idx, 2].imshow(extracted_tuple[0].cpu().permute(1, 2, 0).numpy())
        axes[idx, 2].set_title(f'Tuple scale={test_case["spatial_scale"]}\\n(Correct)')
        axes[idx, 2].grid(True, alpha=0.3)
        
        # Calculate extraction accuracy
        # The blue square should be centered if extraction is correct
        blue_channel = extracted_tuple[0, 2].cpu().numpy()
        if blue_channel.max() > 0:
            blue_mass = blue_channel.sum()
            print(f"  Blue channel sum: {blue_mass:.2f}")
    
    plt.tight_layout()
    plt.savefig('nonsquare_roi_align_test.png', dpi=150)
    print("\nVisualization saved as 'nonsquare_roi_align_test.png'")
    
    # Additional test: Feature extraction at different scales
    print("\n\n=== Testing Feature Extraction at Different Scales ===")
    
    # Simulate feature maps at different strides
    img_480x640 = torch.randn(1, 3, 480, 640).to(device)
    roi_norm = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32).to(device)
    
    # Different feature scales
    feature_configs = [
        ("Original", 1, (480.0, 640.0)),
        ("Conv3 (stride=8)", 8, (480.0/8, 640.0/8)),
        ("Conv5 (stride=16)", 16, (480.0/16, 640.0/16)),
    ]
    
    for name, stride, spatial_scale in feature_configs:
        print(f"\n{name}:")
        print(f"  Feature map size: {480//stride}x{640//stride}")
        print(f"  Spatial scale: {spatial_scale}")
        
        # Create feature map
        feat_h, feat_w = 480//stride, 640//stride
        feature_map = torch.randn(1, 256, feat_h, feat_w).to(device)
        
        # Extract ROI
        roi_align = DynamicRoIAlign(
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True
        ).to(device)
        
        extracted = roi_align(feature_map, roi_norm, 14, 14)
        print(f"  Extracted shape: {extracted.shape}")


if __name__ == "__main__":
    test_nonsquare_roi_align()