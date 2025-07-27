"""Debug coordinate transformation in DynamicRoIAlign."""

import torch
import torch.nn.functional as F
import numpy as np

# Copy key parts of DynamicRoIAlign logic for debugging
def debug_roi_align_coords(image, roi, output_height, output_width, spatial_scale, aligned):
    """Debug version of ROI align to print intermediate values."""
    
    print(f"\n=== Debug ROI Align ===")
    print(f"Image shape: {image.shape}")
    print(f"ROI: {roi}")
    print(f"Output size: {output_height}x{output_width}")
    print(f"Spatial scale: {spatial_scale}")
    print(f"Aligned: {aligned}")
    
    # Scale ROI coordinates
    boxes = roi[:, 1:] * spatial_scale
    print(f"\nScaled boxes: {boxes}")
    
    x1, y1, x2, y2 = boxes[0, 0], boxes[0, 1], boxes[0, 2], boxes[0, 3]
    print(f"x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")
    
    roi_width = x2 - x1
    roi_height = y2 - y1
    print(f"ROI width={roi_width:.2f}, height={roi_height:.2f}")
    
    # Bin dimensions
    bin_width = roi_width / output_width
    bin_height = roi_height / output_height
    print(f"Bin width={bin_width:.2f}, height={bin_height:.2f}")
    
    # Sample a few grid points
    print("\nSampling grid points:")
    for i in range(min(3, output_height)):
        for j in range(min(3, output_width)):
            # Grid coordinates in [0, 1]
            grid_x = j / (output_width - 1) if output_width > 1 else 0.5
            grid_y = i / (output_height - 1) if output_height > 1 else 0.5
            
            # Feature map coordinates
            fx = x1 + (grid_x + 0.5) * bin_width
            fy = y1 + (grid_y + 0.5) * bin_height
            
            # Normalize for grid_sample
            H_feat, W_feat = image.shape[2], image.shape[3]
            if aligned:
                norm_fx = (fx / (W_feat - 1)) * 2 - 1
                norm_fy = (fy / (H_feat - 1)) * 2 - 1
            else:
                norm_fx = (fx / W_feat) * 2 - 1
                norm_fy = (fy / H_feat) * 2 - 1
            
            print(f"  Grid[{i},{j}]: feature=({fx:.2f},{fy:.2f}) -> normalized=({norm_fx:.3f},{norm_fy:.3f})")
            
            # Check if in valid range
            if norm_fx < -1 or norm_fx > 1 or norm_fy < -1 or norm_fy > 1:
                print(f"    WARNING: Normalized coords out of [-1,1] range!")


def test_coordinate_debug():
    """Test coordinate transformation with different settings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Small image (8x8)
    print("=== Test 1: 8x8 image ===")
    img_small = torch.ones(1, 1, 8, 8).to(device)
    
    # Center ROI in normalized coords
    roi_norm = torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32).to(device)
    debug_roi_align_coords(img_small, roi_norm, 4, 4, spatial_scale=8.0, aligned=False)
    
    # Test 2: Large image (640x640)
    print("\n\n=== Test 2: 640x640 image ===")
    img_large = torch.ones(1, 1, 640, 640).to(device)
    
    # ROI in normalized coords
    roi_norm_large = torch.tensor([[0, 0.28125, 0.28125, 0.71875, 0.71875]], dtype=torch.float32).to(device)
    debug_roi_align_coords(img_large, roi_norm_large, 28, 28, spatial_scale=640.0, aligned=True)
    
    # Test 3: What happens with very small normalized coords
    print("\n\n=== Test 3: Small normalized coords ===")
    roi_small = torch.tensor([[0, 0.1, 0.1, 0.2, 0.2]], dtype=torch.float32).to(device)
    debug_roi_align_coords(img_large, roi_small, 4, 4, spatial_scale=640.0, aligned=True)
    
    # Test 4: Pixel coordinates
    print("\n\n=== Test 4: Pixel coordinates ===")
    roi_pixel = torch.tensor([[0, 180, 180, 460, 460]], dtype=torch.float32).to(device)
    debug_roi_align_coords(img_large, roi_pixel, 28, 28, spatial_scale=1.0, aligned=True)


if __name__ == "__main__":
    test_coordinate_debug()