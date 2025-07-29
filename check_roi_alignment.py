"""Check if ROI alignment is extracting the correct regions."""

import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from torch.utils.data import DataLoader
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign

def visualize_roi_alignment():
    """Visualize what regions are being extracted by ROI alignment."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        roi_padding=0.0
    )
    
    # Get a sample
    sample = dataset[0]
    
    # Create batch
    batch = create_collate_fn()([sample])
    
    # Extract data
    images = batch['image'].to(device)
    rois = batch['roi_boxes'].to(device)
    masks = batch['roi_masks'].to(device)
    
    print(f"Image shape: {images.shape}")
    print(f"ROI: {rois[0].tolist()}")
    
    # Create model
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    model.eval()
    
    with torch.no_grad():
        # Get full UNet output
        full_output = model.pretrained_unet(images)
        full_prob = torch.sigmoid(full_output)
        
        # Extract ROI from UNet output
        roi_mask = model.roi_align_mask(full_output, rois, 64, 48)
        roi_prob = torch.sigmoid(roi_mask)
        
        # Extract ROI from RGB image
        roi_rgb = model.roi_align_rgb(images, rois, 64, 48)
    
    # Convert to numpy for visualization
    img_np = images[0].cpu().permute(1, 2, 0).numpy()
    full_prob_np = full_prob[0, 0].cpu().numpy()
    roi_prob_np = roi_prob[0, 0].cpu().numpy()
    roi_rgb_np = roi_rgb[0].cpu().permute(1, 2, 0).numpy()
    
    # ROI coordinates
    _, x1, y1, x2, y2 = rois[0].tolist()
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Full image analysis
    axes[0, 0].imshow(img_np)
    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
    axes[0, 0].add_patch(rect)
    axes[0, 0].set_title(f'Image with ROI\n[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]')
    axes[0, 0].axis('off')
    
    im1 = axes[0, 1].imshow(full_prob_np, cmap='hot', vmin=0, vmax=1)
    rect2 = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=3, edgecolor='red', facecolor='none')
    axes[0, 1].add_patch(rect2)
    axes[0, 1].set_title(f'Full UNet Output\nmax={full_prob_np.max():.3f}')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Manually extract ROI region from full UNet output
    x1_int, y1_int = int(x1), int(y1)
    x2_int, y2_int = int(x2), int(y2)
    roi_manual = full_prob_np[y1_int:y2_int, x1_int:x2_int]
    
    im2 = axes[0, 2].imshow(roi_manual, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title(f'Manual ROI Extract\nmax={roi_manual.max():.3f}')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: ROI analysis
    axes[1, 0].imshow(roi_rgb_np)
    axes[1, 0].set_title('Extracted RGB ROI')
    axes[1, 0].axis('off')
    
    im3 = axes[1, 1].imshow(roi_prob_np, cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'ROI Align Output\nmax={roi_prob_np.max():.3f}')
    axes[1, 1].axis('off')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Ground truth mask
    mask_np = masks[0].cpu().numpy()
    axes[1, 2].imshow(mask_np, cmap='hot')
    axes[1, 2].set_title(f'Ground Truth Mask\nTarget pixels: {(mask_np==1).sum()}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('roi_alignment_check.png', dpi=150)
    print("\nVisualization saved as 'roi_alignment_check.png'")
    
    # Additional checks
    print(f"\n=== Analysis ===")
    print(f"Full UNet output in ROI region:")
    print(f"  Manual extraction max: {roi_manual.max():.3f}")
    print(f"  ROI align extraction max: {roi_prob_np.max():.3f}")
    
    if roi_manual.max() > 0.5 and roi_prob_np.max() < 0.1:
        print("\n⚠️  ROI alignment is NOT extracting the correct region!")
        print("The manual extraction shows high probability but ROI align shows low probability.")
    else:
        print("\n✓ ROI alignment seems to be working correctly.")
    
    # Check if ROI coordinates are within image bounds
    img_h, img_w = images.shape[2:]
    if x2 > img_w or y2 > img_h:
        print(f"\n⚠️  ROI extends beyond image bounds!")
        print(f"Image size: {img_w}x{img_h}, ROI: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

def check_original_vs_resized():
    """Check if the issue is with image resizing."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n=== Checking Original vs Resized Images ===")
    
    # Load COCO
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    img_ids = coco.getImgIds(catIds=[1])[:1]
    img_info = coco.loadImgs(img_ids)[0]
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[1])
    ann = coco.loadAnns(ann_ids)[0]
    
    # Load original image
    img_path = f"data/images/val2017/{img_info['file_name']}"
    img_orig = Image.open(img_path).convert('RGB')
    
    print(f"Original image size: {img_orig.size}")
    print(f"Original bbox: {ann['bbox']}")
    
    # Resize to 640x640
    img_640 = img_orig.resize((640, 640), Image.BILINEAR)
    
    # Scale bbox
    x, y, w, h = ann['bbox']
    scale_x = 640 / img_info['width']
    scale_y = 640 / img_info['height']
    x_scaled = x * scale_x
    y_scaled = y * scale_y
    w_scaled = w * scale_x
    h_scaled = h * scale_y
    
    print(f"Scaled bbox: [{x_scaled:.1f}, {y_scaled:.1f}, {w_scaled:.1f}, {h_scaled:.1f}]")
    
    # Load model
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    
    # Test both images
    for img, name in [(img_orig, "Original"), (img_640, "Resized 640x640")]:
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.pretrained_unet(img_tensor)
            prob = torch.sigmoid(output)
        
        print(f"\n{name}:")
        print(f"  UNet output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"  Probability max: {prob.max():.3f}")

def main():
    print("=== Checking ROI Alignment ===\n")
    
    visualize_roi_alignment()
    check_original_vs_resized()

if __name__ == "__main__":
    main()