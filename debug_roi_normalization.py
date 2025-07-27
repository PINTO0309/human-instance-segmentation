"""Debug ROI normalization issue in the pretrained model."""

import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)
from src.human_edge_detection.advanced.hierarchical_segmentation_unet import (
    PreTrainedPeopleSegmentationUNet
)
from src.human_edge_detection.dynamic_roi_align import DynamicRoIAlign

def debug_roi_normalization():
    """Debug why UNet works on full image but not on ROI."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pretrained UNet directly
    unet = PreTrainedPeopleSegmentationUNet(
        in_channels=3,
        classes=1,
        pretrained_weights_path="ext_extractor/2020-09-23a.pth"
    ).to(device)
    unet.eval()
    
    # Load a test image
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    img_ids = coco.getImgIds(catIds=[1])[:1]
    img_info = coco.loadImgs(img_ids)[0]
    
    img_path = f"data/images/val2017/{img_info['file_name']}"
    image = Image.open(img_path).convert('RGB')
    img_np = np.array(image)
    
    # Convert to tensor [0, 1]
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    print(f"Image: {img_info['file_name']}")
    print(f"Image shape: {img_tensor.shape}")
    print(f"Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Get annotation
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[1])
    anns = coco.loadAnns(ann_ids)
    ann = anns[0]
    bbox = ann['bbox']
    x1, y1, w, h = bbox
    x2, y2 = x1 + w, y1 + h
    
    print(f"\nROI: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    
    # Test 1: Full image through UNet
    print("\n=== Test 1: Full image ===")
    with torch.no_grad():
        full_output = unet(img_tensor)
        full_prob = torch.sigmoid(full_output)
    
    print(f"Full output: range=[{full_output.min():.3f}, {full_output.max():.3f}]")
    print(f"Full prob: mean={full_prob.mean():.3f}, max={full_prob.max():.3f}")
    
    # Test 2: Manual ROI extraction
    print("\n=== Test 2: Manual ROI extraction ===")
    # Resize image to 640x640 first (as done in training)
    img_resized = Image.fromarray(img_np).resize((640, 640), Image.BILINEAR)
    img_resized_np = np.array(img_resized)
    
    # Scale ROI coordinates
    scale_x = 640 / img_info['width']
    scale_y = 640 / img_info['height']
    x1_scaled = int(x1 * scale_x)
    y1_scaled = int(y1 * scale_y)
    x2_scaled = int(x2 * scale_x)
    y2_scaled = int(y2 * scale_y)
    
    # Extract ROI manually
    roi_np = img_resized_np[y1_scaled:y2_scaled, x1_scaled:x2_scaled]
    roi_resized = Image.fromarray(roi_np).resize((64, 48), Image.BILINEAR)
    roi_resized_np = np.array(roi_resized)
    
    # Convert to tensor
    roi_tensor = torch.from_numpy(roi_resized_np).float().permute(2, 0, 1) / 255.0
    roi_tensor = roi_tensor.unsqueeze(0).to(device)
    
    print(f"Manual ROI shape: {roi_tensor.shape}")
    print(f"Manual ROI range: [{roi_tensor.min():.3f}, {roi_tensor.max():.3f}]")
    
    with torch.no_grad():
        roi_output = unet(roi_tensor)
        roi_prob = torch.sigmoid(roi_output)
    
    print(f"Manual ROI output: range=[{roi_output.min():.3f}, {roi_output.max():.3f}]")
    print(f"Manual ROI prob: mean={roi_prob.mean():.3f}, max={roi_prob.max():.3f}")
    
    # Test 3: DynamicRoIAlign extraction
    print("\n=== Test 3: DynamicRoIAlign extraction ===")
    roi_align = DynamicRoIAlign(spatial_scale=1.0, sampling_ratio=2, aligned=True)
    
    # Create ROI tensor [batch_idx, x1, y1, x2, y2]
    # Use scaled coordinates for 640x640 image
    img_640_tensor = torch.from_numpy(img_resized_np).float().permute(2, 0, 1) / 255.0
    img_640_tensor = img_640_tensor.unsqueeze(0).to(device)
    
    roi_coords = torch.tensor([[0, x1_scaled, y1_scaled, x2_scaled, y2_scaled]], dtype=torch.float32).to(device)
    
    # Extract ROI
    roi_aligned = roi_align(img_640_tensor, roi_coords, 64, 48)
    
    print(f"DynamicRoIAlign ROI shape: {roi_aligned.shape}")
    print(f"DynamicRoIAlign ROI range: [{roi_aligned.min():.3f}, {roi_aligned.max():.3f}]")
    
    with torch.no_grad():
        roi_aligned_output = unet(roi_aligned)
        roi_aligned_prob = torch.sigmoid(roi_aligned_output)
    
    print(f"DynamicRoIAlign output: range=[{roi_aligned_output.min():.3f}, {roi_aligned_output.max():.3f}]")
    print(f"DynamicRoIAlign prob: mean={roi_aligned_prob.mean():.3f}, max={roi_aligned_prob.max():.3f}")
    
    # Test 4: Check what happens inside the model
    print("\n=== Test 4: Full model internals ===")
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    model.eval()
    
    # Process through model
    with torch.no_grad():
        # Get full image UNet output
        full_model_output = model.pretrained_unet(img_640_tensor)
        print(f"Model's full UNet output: range=[{full_model_output.min():.3f}, {full_model_output.max():.3f}]")
        
        # Extract ROI mask
        roi_mask = model.roi_align_mask(full_model_output, roi_coords, 64, 48)
        print(f"Model's ROI mask: range=[{roi_mask.min():.3f}, {roi_mask.max():.3f}]")
        
        # Check RGB features
        roi_rgb = model.roi_align_rgb(img_640_tensor, roi_coords, 64, 48)
        print(f"Model's ROI RGB: range=[{roi_rgb.min():.3f}, {roi_rgb.max():.3f}]")
        
        rgb_features = model.rgb_feature_extractor(roi_rgb)
        print(f"RGB features: range=[{rgb_features.min():.3f}, {rgb_features.max():.3f}]")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Images
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(roi_resized_np)
    axes[0, 1].set_title('Manual ROI')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(roi_aligned[0].cpu().permute(1, 2, 0).numpy())
    axes[0, 2].set_title('DynamicRoIAlign ROI')
    axes[0, 2].axis('off')
    
    # Row 2: UNet outputs
    im1 = axes[1, 0].imshow(full_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Full Image Prob\nmax={full_prob.max():.3f}')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0])
    
    im2 = axes[1, 1].imshow(roi_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Manual ROI Prob\nmax={roi_prob.max():.3f}')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1])
    
    im3 = axes[1, 2].imshow(roi_aligned_prob[0, 0].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
    axes[1, 2].set_title(f'DynamicRoIAlign Prob\nmax={roi_aligned_prob.max():.3f}')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('roi_normalization_debug.png')
    print("\nVisualization saved as 'roi_normalization_debug.png'")

def main():
    print("=== Debugging ROI Normalization ===\n")
    debug_roi_normalization()

if __name__ == "__main__":
    main()