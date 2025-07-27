"""Simple test to debug ROI align coordinate transformation."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign


def test_simple_roi_align():
    """Test ROI align with a simple synthetic example."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Simple ROI Align Test ===\n")
    
    # Create a simple 8x8 checkerboard pattern
    H, W = 8, 8
    img = torch.zeros(1, 1, H, W).to(device)
    for i in range(H):
        for j in range(W):
            if (i + j) % 2 == 0:
                img[0, 0, i, j] = 1.0
    
    print("Test image (8x8 checkerboard):")
    print(img[0, 0].cpu().numpy())
    
    # Test extracting the center 4x4 region
    # In pixel coordinates: [2, 2, 6, 6]
    # In normalized coordinates: [2/8, 2/8, 6/8, 6/8] = [0.25, 0.25, 0.75, 0.75]
    
    test_cases = [
        ("Normalized [0.25,0.25,0.75,0.75], scale=8", 
         torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32), 8.0),
        ("Normalized [0.25,0.25,0.75,0.75], scale=1", 
         torch.tensor([[0, 0.25, 0.25, 0.75, 0.75]], dtype=torch.float32), 1.0),
        ("Pixel [2,2,6,6], scale=1", 
         torch.tensor([[0, 2, 2, 6, 6]], dtype=torch.float32), 1.0),
        ("Pixel [2,2,6,6], scale=0.125", 
         torch.tensor([[0, 2, 2, 6, 6]], dtype=torch.float32), 0.125),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original
    axes[0, 0].imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original 8x8')
    axes[0, 0].grid(True, alpha=0.3)
    for i in range(9):
        axes[0, 0].axhline(i-0.5, color='red', alpha=0.3, linewidth=0.5)
        axes[0, 0].axvline(i-0.5, color='red', alpha=0.3, linewidth=0.5)
    
    # Manual extraction of center 4x4
    manual = img[0, 0, 2:6, 2:6].cpu().numpy()
    axes[0, 1].imshow(manual, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Manual [2:6, 2:6]')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Test each configuration
    for idx, (name, roi, scale) in enumerate(test_cases):
        print(f"\nTesting {name}:")
        print(f"  ROI: {roi}")
        print(f"  Scale: {scale}")
        
        roi = roi.to(device)
        
        # Test with both aligned settings
        for aligned in [True, False]:
            roi_align = DynamicRoIAlign(
                spatial_scale=scale,
                sampling_ratio=2,
                aligned=aligned
            ).to(device)
            
            extracted = roi_align(img, roi, 4, 4)
            print(f"  Aligned={aligned}: shape={extracted.shape}, "
                  f"sum={extracted.sum().item():.2f}, "
                  f"pattern={'correct' if abs(extracted.sum().item() - 8.0) < 0.1 else 'wrong'}")
            
            if idx < 2 and aligned:  # Show first two with aligned=True
                row = idx // 2 + 1
                col = idx % 2
                extracted_np = extracted[0, 0].cpu().numpy()
                axes[row, col].imshow(extracted_np, cmap='gray', vmin=0, vmax=1)
                axes[row, col].set_title(f'{name}\naligned={aligned}')
                axes[row, col].grid(True, alpha=0.3)
    
    # Test on a larger real-like example
    print("\n\n=== Testing on 640x640 image ===")
    
    # Create 640x640 image with a square in the middle
    img_large = torch.zeros(1, 1, 640, 640).to(device)
    img_large[0, 0, 200:440, 200:440] = 1.0  # White square
    
    # ROI for the square region with some padding
    # Pixel coords: [180, 180, 460, 460]
    # Normalized: [180/640, 180/640, 460/640, 460/640] = [0.28125, 0.28125, 0.71875, 0.71875]
    
    roi_norm = torch.tensor([[0, 0.28125, 0.28125, 0.71875, 0.71875]], dtype=torch.float32).to(device)
    roi_pixel = torch.tensor([[0, 180, 180, 460, 460]], dtype=torch.float32).to(device)
    
    # Test correct configuration
    roi_align_correct = DynamicRoIAlign(spatial_scale=640.0, sampling_ratio=2, aligned=True).to(device)
    extracted_correct = roi_align_correct(img_large, roi_norm, 28, 28)
    
    # Manual extraction
    manual_large = img_large[0, 0, 180:460, 180:460]
    manual_resized = F.interpolate(manual_large.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear', align_corners=True)
    
    axes[1, 2].imshow(img_large[0, 0].cpu().numpy(), cmap='gray')
    rect = Rectangle((180, 180), 280, 280, linewidth=2, edgecolor='red', facecolor='none')
    axes[1, 2].add_patch(rect)
    axes[1, 2].set_title('640x640 with ROI')
    axes[1, 2].set_xlim(100, 540)
    axes[1, 2].set_ylim(540, 100)
    
    print(f"\nExtracted (correct config) sum: {extracted_correct.sum().item():.2f}")
    print(f"Manual extraction + resize sum: {manual_resized.sum().item():.2f}")
    print(f"Difference: {abs(extracted_correct.sum().item() - manual_resized.sum().item()):.2f}")
    
    # Hide unused axes
    axes[0, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('roi_align_simple_test.png', dpi=150)
    print("\nVisualization saved as 'roi_align_simple_test.png'")


if __name__ == "__main__":
    test_simple_roi_align()