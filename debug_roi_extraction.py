"""Debug ROI extraction and preprocessing in the model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pycocotools.coco import COCO
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign

def debug_roi_extraction():
    """Debug the ROI extraction process step by step."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    model.eval()
    
    # Load COCO data
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    img_ids = coco.getImgIds(catIds=[1])[:1]  # Just one image
    img_info = coco.loadImgs(img_ids)[0]
    
    # Load image
    img_path = f"data/images/val2017/{img_info['file_name']}"
    image = Image.open(img_path).convert('RGB')
    img_np = np.array(image)
    print(f"Original image shape: {img_np.shape}")
    
    # Convert to tensor (NO normalization for now to match training)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Image tensor range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Get annotations and create ROI
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[1])
    anns = coco.loadAnns(ann_ids)
    
    if len(anns) == 0:
        print("No person annotations found!")
        return
    
    # Use first annotation
    ann = anns[0]
    bbox = ann['bbox']  # [x, y, width, height]
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    
    # Create ROI tensor
    roi = torch.tensor([[0, x1, y1, x2, y2]], dtype=torch.float32).to(device)
    print(f"\nROI coordinates: {roi[0].tolist()}")
    
    # Test 1: Full image through UNet
    print("\n=== Testing full image through UNet ===")
    with torch.no_grad():
        full_unet_output = model.pretrained_unet(img_tensor)
        full_fg_prob = torch.sigmoid(full_unet_output)
    
    print(f"Full UNet output shape: {full_unet_output.shape}")
    print(f"Full UNet output range: [{full_unet_output.min():.3f}, {full_unet_output.max():.3f}]")
    print(f"Full foreground prob mean: {full_fg_prob.mean():.3f}")
    
    # Test 2: Extract ROI mask using DynamicRoIAlign
    print("\n=== Testing ROI mask extraction ===")
    roi_align = DynamicRoIAlign(spatial_scale=1.0, sampling_ratio=2, aligned=True)
    roi_mask = roi_align(full_unet_output, roi, 64, 48)
    roi_fg_prob = torch.sigmoid(roi_mask)
    
    print(f"ROI mask shape: {roi_mask.shape}")
    print(f"ROI mask range: [{roi_mask.min():.3f}, {roi_mask.max():.3f}]")
    print(f"ROI foreground prob mean: {roi_fg_prob.mean():.3f}")
    print(f"ROI foreground prob max: {roi_fg_prob.max():.3f}")
    
    # Test 3: Extract RGB ROI
    print("\n=== Testing RGB ROI extraction ===")
    roi_rgb = roi_align(img_tensor, roi, 64, 48)
    print(f"ROI RGB shape: {roi_rgb.shape}")
    print(f"ROI RGB range: [{roi_rgb.min():.3f}, {roi_rgb.max():.3f}]")
    
    # Test 4: Full model forward pass
    print("\n=== Testing full model forward pass ===")
    with torch.no_grad():
        predictions, aux_outputs = model(img_tensor, roi)
        pred_probs = F.softmax(predictions, dim=1)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Prediction probabilities:")
    print(f"  Background: {pred_probs[0, 0].mean():.3f}")
    print(f"  Target: {pred_probs[0, 1].mean():.3f}")
    print(f"  Non-target: {pred_probs[0, 2].mean():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Original image and full UNet output
    axes[0, 0].imshow(img_np)
    # Draw ROI box
    from matplotlib.patches import Rectangle
    rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f'Original Image with ROI\n{img_info["file_name"]}')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(full_unet_output[0, 0].cpu().numpy(), cmap='RdBu_r', vmin=-10, vmax=10)
    axes[0, 1].set_title('Full UNet Logits')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(full_fg_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Full FG Probability\n(mean: {full_fg_prob.mean():.3f})')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: ROI extractions
    axes[1, 0].imshow(roi_rgb[0].cpu().permute(1, 2, 0).numpy())
    axes[1, 0].set_title('Extracted RGB ROI')
    axes[1, 0].axis('off')
    
    im3 = axes[1, 1].imshow(roi_mask[0, 0].cpu().numpy(), cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1, 1].set_title(f'ROI Mask Logits\n(range: [{roi_mask.min():.1f}, {roi_mask.max():.1f}])')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    im4 = axes[1, 2].imshow(roi_fg_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'ROI FG Probability\n(mean: {roi_fg_prob.mean():.3f})')
    axes[1, 2].axis('off')
    plt.colorbar(im4, ax=axes[1, 2])
    
    # Row 3: Model predictions
    for i in range(3):
        class_name = ['Background', 'Target', 'Non-target'][i]
        im = axes[2, i].imshow(pred_probs[0, i].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        axes[2, i].set_title(f'{class_name} Probability\n(mean: {pred_probs[0, i].mean():.3f})')
        axes[2, i].axis('off')
        plt.colorbar(im, ax=axes[2, i])
    
    plt.tight_layout()
    plt.savefig('roi_extraction_debug.png', dpi=150)
    print("\nVisualization saved as 'roi_extraction_debug.png'")
    
    # Additional test: Check if ROI coordinates are correct
    print("\n=== Checking ROI coordinate system ===")
    print(f"Image dimensions: {img_info['height']} x {img_info['width']}")
    print(f"ROI bbox: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
    print(f"ROI width: {x2-x1:.1f}, height: {y2-y1:.1f}")
    
    # Check if ROI is within image bounds
    if x2 > img_info['width'] or y2 > img_info['height']:
        print("WARNING: ROI extends beyond image boundaries!")
    
    # Create a simple test to verify ROI extraction
    print("\n=== Simple ROI extraction test ===")
    # Create a test image with a white square
    test_img = torch.zeros(1, 3, 256, 256).to(device)
    test_img[:, :, 50:150, 50:150] = 1.0  # White square
    
    # Extract the square region
    test_roi = torch.tensor([[0, 50, 50, 150, 150]], dtype=torch.float32).to(device)
    extracted = roi_align(test_img, test_roi, 64, 48)
    
    print(f"Test extraction - should be all white: min={extracted.min():.3f}, max={extracted.max():.3f}")
    if extracted.min() > 0.9 and extracted.max() > 0.9:
        print("✓ ROI extraction working correctly")
    else:
        print("✗ ROI extraction has issues!")

def main():
    print("=== Debugging ROI Extraction Process ===\n")
    debug_roi_extraction()

if __name__ == "__main__":
    main()