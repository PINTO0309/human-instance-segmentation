"""Test normalized ROI extraction with correct spatial_scale."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign
from src.human_edge_detection.advanced.hierarchical_segmentation_unet import PreTrainedPeopleSegmentationUNetWrapper


def test_normalized_roi_extraction():
    """Test ROI extraction with normalized coordinates."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Normalized ROI Extraction ===\n")
    
    # Load dataset
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        roi_padding=0.0
    )
    
    # Get a sample
    sample = dataset[0]
    image = sample['image'].unsqueeze(0).to(device)  # Add batch dimension
    roi_coords_norm = sample['roi_coords'].unsqueeze(0)  # Normalized [0, 1]
    
    print(f"Image shape: {image.shape}")
    print(f"ROI coords (normalized): {roi_coords_norm}")
    
    # Create ROI boxes with batch index
    roi_boxes = torch.cat([
        torch.zeros(1, 1),  # batch index
        roi_coords_norm
    ], dim=1).to(device)
    
    print(f"ROI boxes shape: {roi_boxes.shape}")
    print(f"ROI boxes: {roi_boxes}")
    
    # Test with different spatial_scale values
    test_cases = [
        ("Wrong (pixel coords)", 1.0),
        ("Correct (normalized coords)", 640.0)
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show original image with ROI
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(img_np)
    
    # Draw ROI box
    x1, y1, x2, y2 = roi_coords_norm[0].numpy() * 640
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                     linewidth=2, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original Image with ROI')
    axes[0, 0].axis('off')
    
    # Test Pre-trained UNet on full image
    print("\nTesting Pre-trained UNet on full image...")
    unet = PreTrainedPeopleSegmentationUNetWrapper(in_channels=3).to(device)
    unet.eval()
    
    with torch.no_grad():
        full_unet_output = unet(image)
        if isinstance(full_unet_output, tuple):
            full_unet_output = full_unet_output[0]  # Get the main output
        full_unet_prob = torch.sigmoid(full_unet_output)
    
    axes[1, 0].imshow(full_unet_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'UNet Full Image\n(max={full_unet_prob.max():.3f})')
    axes[1, 0].axis('off')
    
    # Test ROI extraction with different spatial scales
    for i, (name, spatial_scale) in enumerate(test_cases, 1):
        print(f"\nTesting {name} with spatial_scale={spatial_scale}")
        
        roi_align = DynamicRoIAlign(
            spatial_scale=spatial_scale,
            sampling_ratio=2
        ).to(device)
        
        # Extract ROI from image
        roi_extracted = roi_align(image, roi_boxes, 64, 48)
        print(f"  Extracted ROI shape: {roi_extracted.shape}")
        
        # Show extracted ROI
        roi_np = roi_extracted[0].cpu().permute(1, 2, 0).numpy()
        axes[0, i].imshow(roi_np)
        axes[0, i].set_title(f'{name}\n(spatial_scale={spatial_scale})')
        axes[0, i].axis('off')
        
        # Test UNet on extracted ROI
        with torch.no_grad():
            roi_unet_output = unet(roi_extracted)
            if isinstance(roi_unet_output, tuple):
                roi_unet_output = roi_unet_output[0]  # Get the main output
            roi_unet_prob = torch.sigmoid(roi_unet_output)
        
        axes[1, i].imshow(roi_unet_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[1, i].set_title(f'UNet ROI Output\n(max={roi_unet_prob.max():.3f})')
        axes[1, i].axis('off')
        
        print(f"  UNet output range: [{roi_unet_prob.min():.3f}, {roi_unet_prob.max():.3f}]")
        print(f"  UNet output mean: {roi_unet_prob.mean():.3f}")
    
    plt.tight_layout()
    plt.savefig('normalized_roi_test.png', dpi=150)
    print("\nVisualization saved as 'normalized_roi_test.png'")
    
    # Test with batch processing
    print("\n\n=== Testing Batch Processing ===")
    
    # Create dataloader with collate function
    from torch.utils.data import DataLoader
    collate_fn = create_collate_fn()
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
    
    # Get a batch
    batch = next(iter(dataloader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Images shape: {batch['image'].shape}")
    print(f"ROI boxes shape: {batch['roi_boxes'].shape}")
    print(f"ROI boxes (first 2):\n{batch['roi_boxes'][:2]}")
    
    # Verify all ROI coordinates are normalized
    assert (batch['roi_boxes'][:, 1:] >= 0).all() and (batch['roi_boxes'][:, 1:] <= 1).all(), \
        "ROI coordinates should be normalized to [0, 1]"
    print("\n✓ All ROI coordinates are properly normalized to [0, 1]")


if __name__ == "__main__":
    test_normalized_roi_extraction()