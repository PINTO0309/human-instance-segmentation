"""Analyze the pretrained UNet output on real data."""

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
from src.human_edge_detection.advanced.hierarchical_segmentation_unet import (
    PreTrainedPeopleSegmentationUNet
)

def analyze_pretrained_unet_on_real_data():
    """Analyze how the pretrained UNet performs on real COCO data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained UNet directly
    print("Loading pre-trained UNet...")
    unet = PreTrainedPeopleSegmentationUNet(
        in_channels=3,
        classes=1,
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        freeze_weights=True
    ).to(device)
    unet.eval()
    
    # Load COCO annotations
    print("\nLoading COCO annotations...")
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    
    # Get images with person annotations
    person_cat_id = 1
    img_ids = coco.getImgIds(catIds=[person_cat_id])[:5]  # Test on 5 images
    
    print(f"\nAnalyzing {len(img_ids)} images...")
    
    for idx, img_id in enumerate(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        img_path = f"data/images/val2017/{img_info['file_name']}"
        
        # Load and preprocess image
        image = Image.open(img_path).convert('RGB')
        img_np = np.array(image)
        
        # Convert to tensor and normalize (assuming standard ImageNet normalization)
        img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        # Get UNet prediction
        with torch.no_grad():
            unet_output = unet(img_tensor)  # (1, 1, H, W)
            fg_prob = torch.sigmoid(unet_output)
        
        print(f"\nImage {idx + 1} ({img_info['file_name']}):")
        print(f"  UNet output range: [{unet_output.min():.3f}, {unet_output.max():.3f}]")
        print(f"  UNet output mean: {unet_output.mean():.3f}")
        print(f"  Foreground probability range: [{fg_prob.min():.3f}, {fg_prob.max():.3f}]")
        print(f"  Foreground probability mean: {fg_prob.mean():.3f}")
        print(f"  Pixels > 0.5 probability: {(fg_prob > 0.5).float().mean():.3%}")
        
        # Get person annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)
        
        # Create ground truth mask
        gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.float32)
        for ann in anns:
            mask = coco.annToMask(ann)
            gt_mask = np.maximum(gt_mask, mask)
        
        # Compare with ground truth
        if gt_mask.sum() > 0:
            gt_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0).unsqueeze(0).to(device)
            # Resize to match UNet output size
            if gt_tensor.shape != fg_prob.shape:
                gt_tensor = F.interpolate(gt_tensor, size=fg_prob.shape[2:], mode='nearest')
            
            # Calculate IoU
            intersection = (fg_prob > 0.5).float() * gt_tensor
            union = ((fg_prob > 0.5).float() + gt_tensor).clamp(0, 1)
            iou = intersection.sum() / (union.sum() + 1e-6)
            print(f"  IoU with ground truth: {iou:.3f}")
        
        # Save visualization for first image
        if idx == 0:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            
            # Original image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Ground truth mask
            axes[0, 1].imshow(gt_mask, cmap='hot')
            axes[0, 1].set_title('Ground Truth Person Mask')
            axes[0, 1].axis('off')
            
            # UNet logits
            unet_vis = unet_output[0, 0].cpu().numpy()
            im1 = axes[1, 0].imshow(unet_vis, cmap='RdBu_r', vmin=-10, vmax=10)
            axes[1, 0].set_title(f'UNet Logits (range: [{unet_vis.min():.1f}, {unet_vis.max():.1f}])')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0])
            
            # UNet probability
            prob_vis = fg_prob[0, 0].cpu().numpy()
            im2 = axes[1, 1].imshow(prob_vis, cmap='hot', vmin=0, vmax=1)
            axes[1, 1].set_title(f'UNet Probability (mean: {prob_vis.mean():.3f})')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1])
            
            plt.tight_layout()
            plt.savefig('pretrained_unet_analysis.png', dpi=150)
            print("\n  Visualization saved as 'pretrained_unet_analysis.png'")

def test_on_synthetic_person():
    """Test the UNet on a synthetic image with a clear person shape."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained UNet
    unet = PreTrainedPeopleSegmentationUNet(
        in_channels=3,
        classes=1,
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        freeze_weights=True
    ).to(device)
    unet.eval()
    
    # Create synthetic image with person-like shape
    img_size = 512
    img = np.ones((img_size, img_size, 3), dtype=np.float32) * 0.8  # Light gray background
    
    # Draw a simple person shape (head + body)
    # Head
    center_x, center_y = img_size // 2, img_size // 3
    head_radius = 40
    y, x = np.ogrid[:img_size, :img_size]
    head_mask = (x - center_x)**2 + (y - center_y)**2 <= head_radius**2
    img[head_mask] = [0.2, 0.3, 0.4]  # Dark color for head
    
    # Body (rectangle)
    body_top = center_y + head_radius
    body_width = 80
    body_height = 150
    body_left = center_x - body_width // 2
    body_right = center_x + body_width // 2
    body_bottom = body_top + body_height
    img[body_top:body_bottom, body_left:body_right] = [0.3, 0.4, 0.5]
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Get UNet prediction
    with torch.no_grad():
        unet_output = unet(img_tensor)
        fg_prob = torch.sigmoid(unet_output)
    
    print("\nSynthetic person test:")
    print(f"  UNet output range: [{unet_output.min():.3f}, {unet_output.max():.3f}]")
    print(f"  Foreground probability mean: {fg_prob.mean():.3f}")
    print(f"  Max foreground probability: {fg_prob.max():.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Synthetic Person Image')
    axes[0].axis('off')
    
    logits_vis = unet_output[0, 0].cpu().numpy()
    im1 = axes[1].imshow(logits_vis, cmap='RdBu_r', vmin=-10, vmax=10)
    axes[1].set_title(f'UNet Logits (range: [{logits_vis.min():.1f}, {logits_vis.max():.1f}])')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    prob_vis = fg_prob[0, 0].cpu().numpy()
    im2 = axes[2].imshow(prob_vis, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title(f'Foreground Probability (max: {prob_vis.max():.3f})')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('synthetic_person_test.png', dpi=150)
    print("  Visualization saved as 'synthetic_person_test.png'")

def main():
    print("=== Analyzing Pre-trained UNet Output ===\n")
    
    print("1. Testing on real COCO data:")
    print("-" * 50)
    analyze_pretrained_unet_on_real_data()
    
    print("\n\n2. Testing on synthetic person:")
    print("-" * 50)
    test_on_synthetic_person()

if __name__ == "__main__":
    main()