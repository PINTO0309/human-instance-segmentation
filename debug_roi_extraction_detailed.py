"""Detailed debug of ROI extraction with normalized coordinates."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign


def debug_roi_extraction():
    """Debug ROI extraction step by step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Detailed ROI Extraction Debug ===\n")
    
    # Load dataset
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        roi_padding=0.2
    )
    
    # Get a sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)  # [1, 3, 640, 640]
    roi_coords_norm = sample['roi_coords']  # Normalized [0, 1]
    
    print(f"Image shape: {image.shape}")
    print(f"ROI coords (normalized): {roi_coords_norm}")
    
    # Convert to pixel coordinates for visualization
    x1_px = roi_coords_norm[0] * 640
    y1_px = roi_coords_norm[1] * 640
    x2_px = roi_coords_norm[2] * 640
    y2_px = roi_coords_norm[3] * 640
    
    print(f"\nROI in pixel coordinates: [{x1_px:.1f}, {y1_px:.1f}, {x2_px:.1f}, {y2_px:.1f}]")
    print(f"ROI width: {x2_px - x1_px:.1f}, height: {y2_px - y1_px:.1f}")
    
    # Manual extraction in pixel space
    print("\n1. Manual extraction (pixel space):")
    manual_roi = image[0, :, int(y1_px):int(y2_px), int(x1_px):int(x2_px)]
    print(f"   Manual ROI shape: {manual_roi.shape}")
    
    # Create ROI boxes for DynamicRoIAlign
    roi_boxes_norm = torch.tensor([[0, roi_coords_norm[0], roi_coords_norm[1], 
                                     roi_coords_norm[2], roi_coords_norm[3]]], 
                                   dtype=torch.float32).to(device)
    
    roi_boxes_pixel = torch.tensor([[0, x1_px, y1_px, x2_px, y2_px]], 
                                    dtype=torch.float32).to(device)
    
    # Test different configurations
    configs = [
        ("Normalized coords, spatial_scale=640", roi_boxes_norm, 640.0),
        ("Normalized coords, spatial_scale=1.0", roi_boxes_norm, 1.0),
        ("Pixel coords, spatial_scale=1.0", roi_boxes_pixel, 1.0),
        ("Pixel coords, spatial_scale=1/640", roi_boxes_pixel, 1.0/640.0),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Show original image with ROI
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img_np)
    rect = Rectangle((x1_px, y1_px), x2_px-x1_px, y2_px-y1_px, 
                     linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original Image with ROI')
    axes[0, 0].axis('off')
    
    # Show manual extraction
    manual_roi_np = manual_roi.cpu().permute(1, 2, 0).numpy()
    axes[0, 1].imshow(manual_roi_np)
    axes[0, 1].set_title(f'Manual Extraction\n{manual_roi.shape[1:]}')
    axes[0, 1].axis('off')
    
    # Show ground truth mask
    roi_mask_np = sample['roi_mask'].numpy()
    axes[0, 2].imshow(roi_mask_np, cmap='tab10', vmin=0, vmax=2)
    axes[0, 2].set_title(f'Ground Truth Mask\n{roi_mask_np.shape}')
    axes[0, 2].axis('off')
    
    # Test each configuration
    for idx, (name, roi_boxes, spatial_scale) in enumerate(configs):
        print(f"\n{idx+2}. Testing {name}:")
        print(f"   ROI boxes: {roi_boxes}")
        print(f"   Spatial scale: {spatial_scale}")
        
        roi_align = DynamicRoIAlign(
            spatial_scale=spatial_scale,
            sampling_ratio=2,
            aligned=True  # Try with aligned=True
        ).to(device)
        
        # Extract at different sizes
        for size_idx, (h, w) in enumerate([(64, 48), (128, 96)]):
            roi_extracted = roi_align(image, roi_boxes, h, w)
            print(f"   Extracted shape ({h}x{w}): {roi_extracted.shape}")
            
            # Visualize
            row = 1 + idx // 2
            col = (idx % 2) * 2 + size_idx
            if row < 3 and col < 3:
                roi_np = roi_extracted[0].cpu().permute(1, 2, 0).numpy()
                axes[row, col].imshow(roi_np)
                axes[row, col].set_title(f'{name}\n{h}x{w}')
                axes[row, col].axis('off')
    
    # Fill remaining axes
    for i in range(2, 3):
        for j in range(3):
            if i == 2 and j >= 2:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('roi_extraction_debug.png', dpi=150)
    print("\nVisualization saved as 'roi_extraction_debug.png'")
    
    # Additional coordinate space analysis
    print("\n\n=== Coordinate Space Analysis ===")
    print(f"Image tensor shape: {image.shape}")
    print(f"Normalized ROI: {roi_coords_norm}")
    print(f"When spatial_scale=640.0:")
    print(f"  - Input coords are treated as normalized [0,1]")
    print(f"  - They are multiplied by spatial_scale to get feature map coords")
    print(f"  - {roi_coords_norm[0]:.3f} * 640 = {roi_coords_norm[0]*640:.1f}")
    print(f"When spatial_scale=1.0:")
    print(f"  - Input coords are treated as pixel coordinates")
    print(f"  - No scaling is applied")


if __name__ == "__main__":
    debug_roi_extraction()