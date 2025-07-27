"""Debug DynamicRoIAlign to find why it's not extracting the correct region."""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pycocotools.coco import COCO
from PIL import Image

from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign

def test_roi_align_simple():
    """Test ROI align with a simple synthetic example."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Simple Synthetic Test ===")
    
    # Create a simple test image with a white square
    H, W = 8, 8
    img = torch.zeros(1, 1, H, W).to(device)
    img[0, 0, 2:6, 2:6] = 1.0  # White square in center
    
    print(f"Test image shape: {img.shape}")
    print("Test image (1=white, 0=black):")
    print(img[0, 0].cpu().numpy())
    
    # Test different ROIs
    roi_align = DynamicRoIAlign(spatial_scale=1.0, sampling_ratio=2)
    
    test_rois = [
        [0, 2, 2, 6, 6],  # Exact square
        [0, 1, 1, 7, 7],  # Larger than square
        [0, 2.5, 2.5, 5.5, 5.5],  # Fractional coordinates
        [0, 0, 0, 8, 8],  # Full image
    ]
    
    for i, roi in enumerate(test_rois):
        roi_tensor = torch.tensor([roi], dtype=torch.float32).to(device)
        extracted = roi_align(img, roi_tensor, 4, 4)
        
        print(f"\nROI {i}: {roi}")
        print(f"Extracted shape: {extracted.shape}")
        print(f"Extracted values:")
        print(extracted[0, 0].cpu().numpy())
        print(f"Sum: {extracted.sum().item():.2f}")

def debug_real_data():
    """Debug with real COCO data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n\n=== Real Data Debug ===")
    
    # Load COCO
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    img_ids = coco.getImgIds(catIds=[1])[:1]
    img_info = coco.loadImgs(img_ids)[0]
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[1])
    ann = coco.loadAnns(ann_ids)[0]
    
    # Load and resize image
    img_path = f"data/images/val2017/{img_info['file_name']}"
    img = Image.open(img_path).convert('RGB')
    img_640 = img.resize((640, 640), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img_640)).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Get bbox
    x, y, w, h = ann['bbox']
    scale_x = 640 / img_info['width']
    scale_y = 640 / img_info['height']
    
    x1 = x * scale_x
    y1 = y * scale_y
    x2 = x1 + w * scale_x
    y2 = y1 + h * scale_y
    
    # Add padding
    padding = 0.2
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    
    x1_pad = max(0, x1 - pad_x)
    y1_pad = max(0, y1 - pad_y)
    x2_pad = min(640, x2 + pad_x)
    y2_pad = min(640, y2 + pad_y)
    
    print(f"Original bbox: {ann['bbox']}")
    print(f"Scaled bbox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"Padded bbox: [{x1_pad:.1f}, {y1_pad:.1f}, {x2_pad:.1f}, {y2_pad:.1f}]")
    print(f"Width: {x2_pad-x1_pad:.1f}, Height: {y2_pad-y1_pad:.1f}")
    
    # Create simple test feature map (just grayscale version)
    feature_map = img_tensor.mean(dim=1, keepdim=True)
    
    # Test ROI extraction
    roi_align = DynamicRoIAlign(spatial_scale=1.0, sampling_ratio=2)
    
    # Test both ROIs
    for name, roi_coords in [
        ("Scaled", [0, x1, y1, x2, y2]),
        ("Padded", [0, x1_pad, y1_pad, x2_pad, y2_pad])
    ]:
        roi_tensor = torch.tensor([roi_coords], dtype=torch.float32).to(device)
        extracted = roi_align(feature_map, roi_tensor, 64, 48)
        
        print(f"\n{name} ROI extraction:")
        print(f"  ROI: {roi_coords}")
        print(f"  Extracted shape: {extracted.shape}")
        print(f"  Value range: [{extracted.min():.3f}, {extracted.max():.3f}]")
        print(f"  Mean: {extracted.mean():.3f}")
        
        # Manually extract the same region
        x1_int = int(roi_coords[1])
        y1_int = int(roi_coords[2])
        x2_int = int(roi_coords[3])
        y2_int = int(roi_coords[4])
        
        manual_extract = feature_map[0, :, y1_int:y2_int, x1_int:x2_int]
        print(f"  Manual extract shape: {manual_extract.shape}")
        print(f"  Manual extract mean: {manual_extract.mean():.3f}")

def test_coordinate_normalization():
    """Test coordinate normalization in grid_sample."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n\n=== Coordinate Normalization Test ===")
    
    # Create test image
    H, W = 10, 10
    img = torch.arange(H * W, dtype=torch.float32).reshape(1, 1, H, W).to(device)
    
    print(f"Test image (values 0-99):")
    print(img[0, 0, :5, :5].cpu().numpy())
    
    # Test grid_sample with known coordinates
    # grid_sample expects coordinates in [-1, 1] range
    # -1 = left/top edge, 1 = right/bottom edge
    
    # Sample center pixel (4.5, 4.5) in pixel coordinates
    # Should be normalized to (0, 0) in [-1, 1] range
    x_norm = (4.5 / (W - 1)) * 2 - 1  # = 0
    y_norm = (4.5 / (H - 1)) * 2 - 1  # = 0
    
    grid = torch.tensor([[[[x_norm, y_norm]]]], dtype=torch.float32).to(device)
    sampled = F.grid_sample(img, grid, mode='bilinear', align_corners=True)
    
    print(f"\nSampling center (4.5, 4.5):")
    print(f"  Normalized coords: ({x_norm:.3f}, {y_norm:.3f})")
    print(f"  Sampled value: {sampled[0, 0, 0, 0]:.1f}")
    print(f"  Expected: ~44.5 (average of 44 and 45)")

def visualize_roi_extraction():
    """Visualize what's happening with ROI extraction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n\n=== Visual Debug ===")
    
    # Create a test pattern
    H, W = 20, 20
    img = torch.zeros(1, 1, H, W).to(device)
    
    # Create checkerboard pattern
    for i in range(0, H, 2):
        for j in range(0, W, 2):
            img[0, 0, i, j] = 1
            if i+1 < H and j+1 < W:
                img[0, 0, i+1, j+1] = 1
    
    # Define ROI
    roi = [0, 5, 5, 15, 15]  # 10x10 region in center
    roi_tensor = torch.tensor([roi], dtype=torch.float32).to(device)
    
    # Extract with different output sizes
    roi_align = DynamicRoIAlign(spatial_scale=1.0, sampling_ratio=2)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
    rect = Rectangle((roi[1], roi[2]), roi[3]-roi[1], roi[4]-roi[2], 
                     linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original with ROI')
    axes[0, 0].grid(True)
    
    # Extract at different sizes
    sizes = [(10, 10), (5, 5), (20, 20)]
    for i, (h, w) in enumerate(sizes):
        extracted = roi_align(img, roi_tensor, h, w)
        axes[0, i+1].imshow(extracted[0, 0].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        axes[0, i+1].set_title(f'Extracted {h}x{w}')
        axes[0, i+1].grid(True)
    
    # Manual extraction
    manual = img[0, 0, 5:15, 5:15].cpu().numpy()
    axes[1, 0].imshow(manual, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Manual Extract')
    axes[1, 0].grid(True)
    
    # Show difference
    extracted_10x10 = roi_align(img, roi_tensor, 10, 10)
    diff = manual - extracted_10x10[0, 0].cpu().numpy()
    axes[1, 1].imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title('Difference (Manual - RoIAlign)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('roi_align_debug.png')
    print("\nVisualization saved as 'roi_align_debug.png'")

def main():
    print("=== Debugging DynamicRoIAlign ===\n")
    
    test_roi_align_simple()
    debug_real_data()
    test_coordinate_normalization()
    visualize_roi_extraction()

if __name__ == "__main__":
    main()