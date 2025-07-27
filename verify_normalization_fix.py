"""Verify that the normalization fix improves pretrained model output."""

import torch
import numpy as np
from pycocotools.coco import COCO
from PIL import Image

from src.human_edge_detection.advanced.hierarchical_segmentation_unet import (
    PreTrainedPeopleSegmentationUNet
)

def test_normalization_fix():
    """Test the pretrained model with fixed normalization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model with fixed normalization
    model = PreTrainedPeopleSegmentationUNet(
        in_channels=3,
        classes=1,
        pretrained_weights_path="ext_extractor/2020-09-23a.pth"
    ).to(device)
    model.eval()
    
    # Load a real COCO image
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    img_ids = coco.getImgIds(catIds=[1])[:1]
    img_info = coco.loadImgs(img_ids)[0]
    
    # Load image
    img_path = f"data/images/val2017/{img_info['file_name']}"
    image = Image.open(img_path).convert('RGB')
    img_np = np.array(image)
    
    # Convert to tensor with [0, 1] normalization (dataset default)
    img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    print(f"Image: {img_info['file_name']}")
    print(f"Image shape: {img_tensor.shape}")
    print(f"Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    
    # Get model output
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output)
    
    print(f"\nWith fixed normalization (simple_norm):")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  Output mean: {output.mean():.3f}")
    print(f"  Prob mean: {prob.mean():.3f}, max: {prob.max():.3f}")
    print(f"  Pixels > 0.5: {(prob > 0.5).float().mean():.3%}")
    
    # Compare with ground truth
    ann_ids = coco.getAnnIds(imgIds=img_ids, catIds=[1])
    anns = coco.loadAnns(ann_ids)
    
    if anns:
        print(f"\nGround truth: {len(anns)} person annotation(s)")
        # Check if model detects people in reasonable areas
        if prob.max() > 0.5:
            print("✓ Model successfully detects foreground regions")
        else:
            print("✗ Model still not detecting foreground properly")
    
    return prob.max() > 0.5

def main():
    print("=== Verifying Normalization Fix ===\n")
    
    success = test_normalization_fix()
    
    if success:
        print("\n✅ Normalization fix successful! Model is now detecting people properly.")
    else:
        print("\n❌ Normalization fix may need further adjustment.")

if __name__ == "__main__":
    main()