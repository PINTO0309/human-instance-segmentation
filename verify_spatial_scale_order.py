"""Verify the spatial_scale order convention (height, width) vs (width, height)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign


def verify_spatial_scale_order():
    """Verify whether spatial_scale expects (height, width) or (width, height)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Verifying Spatial Scale Order ===\n")
    
    # Create a distinctive test pattern
    # Non-square image: 480 (height) x 640 (width)
    H, W = 480, 640
    img = torch.zeros(1, 3, H, W).to(device)
    
    # Create distinct patterns for each axis
    # Horizontal stripes (vary with y/height)
    for i in range(0, H, 40):
        img[0, 0, i:i+20, :] = 1.0  # Red horizontal stripes
    
    # Vertical stripes (vary with x/width)  
    for j in range(0, W, 40):
        img[0, 1, :, j:j+20] = 1.0  # Green vertical stripes
    
    # Add markers at specific locations to verify orientation
    # Top-left corner marker (blue)
    img[0, 2, 20:60, 20:60] = 1.0
    # Bottom-right corner marker (blue)
    img[0, 2, H-60:H-20, W-60:W-20] = 1.0
    
    # Define ROI that should capture the center region
    # In normalized coordinates [0,1]
    roi_norm = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32).to(device)
    
    # Calculate expected pixel coordinates
    x1_px = 0.25 * W  # 0.25 * 640 = 160
    y1_px = 0.25 * H  # 0.25 * 480 = 120
    x2_px = 0.75 * W  # 0.75 * 640 = 480
    y2_px = 0.75 * H  # 0.75 * 480 = 360
    
    print(f"Image size: H={H}, W={W}")
    print(f"ROI normalized: {roi_norm[0, 1:].tolist()}")
    print(f"ROI in pixels: x:[{x1_px:.0f}, {x2_px:.0f}], y:[{y1_px:.0f}, {y2_px:.0f}]")
    print(f"ROI size: {x2_px-x1_px:.0f} (width) x {y2_px-y1_px:.0f} (height)")
    
    # Test both possible interpretations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original image
    img_np = img[0].cpu().permute(1, 2, 0).numpy()
    for row in range(2):
        axes[row, 0].imshow(img_np)
        rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px,
                        linewidth=2, edgecolor='yellow', facecolor='none')
        axes[row, 0].add_patch(rect)
        axes[row, 0].set_title('Original Image\n(Red=horizontal, Green=vertical)')
        axes[row, 0].set_xlabel(f'Width: {W}')
        axes[row, 0].set_ylabel(f'Height: {H}')
    
    # Test 1: spatial_scale = (H, W) = (480, 640)
    print("\n\nTest 1: spatial_scale = (H, W) = (480, 640)")
    roi_align_hw = DynamicRoIAlign(
        spatial_scale=(480.0, 640.0),  # (height, width)
        sampling_ratio=2,
        aligned=True
    ).to(device)
    
    extracted_hw = roi_align_hw(img, roi_norm, 64, 64)
    axes[0, 1].imshow(extracted_hw[0].cpu().permute(1, 2, 0).numpy())
    axes[0, 1].set_title('spatial_scale=(480, 640)\nIf correct: H stripes horizontal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Analyze the pattern
    red_channel = extracted_hw[0, 0].cpu().numpy()
    green_channel = extracted_hw[0, 1].cpu().numpy()
    
    # Check if stripes are in correct orientation
    red_horizontal_var = np.var(red_channel.mean(axis=1))  # Variance along height
    red_vertical_var = np.var(red_channel.mean(axis=0))    # Variance along width
    green_horizontal_var = np.var(green_channel.mean(axis=1))
    green_vertical_var = np.var(green_channel.mean(axis=0))
    
    print(f"  Red (horizontal stripes) variance: H={red_horizontal_var:.3f}, V={red_vertical_var:.3f}")
    print(f"  Green (vertical stripes) variance: H={green_horizontal_var:.3f}, V={green_vertical_var:.3f}")
    
    # Show variance analysis
    axes[0, 2].text(0.1, 0.8, f'Red H-var: {red_horizontal_var:.3f}', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.6, f'Red V-var: {red_vertical_var:.3f}', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.4, f'Green H-var: {green_horizontal_var:.3f}', transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.2, f'Green V-var: {green_vertical_var:.3f}', transform=axes[0, 2].transAxes)
    axes[0, 2].set_title('Variance Analysis\n(H-var high = horizontal stripes)')
    axes[0, 2].axis('off')
    
    # Test 2: spatial_scale = (W, H) = (640, 480) 
    print("\n\nTest 2: spatial_scale = (W, H) = (640, 480)")
    roi_align_wh = DynamicRoIAlign(
        spatial_scale=(640.0, 480.0),  # (width, height) - WRONG ORDER
        sampling_ratio=2,
        aligned=True
    ).to(device)
    
    extracted_wh = roi_align_wh(img, roi_norm, 64, 64)
    axes[1, 1].imshow(extracted_wh[0].cpu().permute(1, 2, 0).numpy())
    axes[1, 1].set_title('spatial_scale=(640, 480)\nIf wrong: patterns distorted')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Show what the correct extraction should look like
    # Manual extraction
    manual_roi = img[0, :, int(y1_px):int(y2_px), int(x1_px):int(x2_px)]
    manual_resized = torch.nn.functional.interpolate(
        manual_roi.unsqueeze(0), 
        size=(64, 64), 
        mode='bilinear', 
        align_corners=True
    )[0]
    
    axes[1, 2].imshow(manual_resized.cpu().permute(1, 2, 0).numpy())
    axes[1, 2].set_title('Manual extraction\n(Ground truth)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spatial_scale_order_verification.png', dpi=150)
    print("\nVisualization saved as 'spatial_scale_order_verification.png'")
    
    # Conclusion
    print("\n\n=== CONCLUSION ===")
    print("Based on the documentation and code:")
    print("- ROI format: [batch_idx, x1, y1, x2, y2]")
    print("- x coordinates are scaled by spatial_scale_w")
    print("- y coordinates are scaled by spatial_scale_h")
    print("- Therefore, spatial_scale tuple should be (scale_h, scale_w)")
    print("\nFor a 480x640 image:")
    print("- Correct: spatial_scale = (480, 640)  # (height, width)")
    print("- Wrong:   spatial_scale = (640, 480)  # (width, height)")


if __name__ == "__main__":
    verify_spatial_scale_order()