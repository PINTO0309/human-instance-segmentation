"""Test training with filtered dataset."""

import torch
from torch.utils.data import DataLoader

from src.human_edge_detection.filtered_dataset import FilteredCOCODataset, create_filtered_dataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet


def test_filtered_dataset():
    """Test the filtered dataset and model training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Testing Filtered Dataset ===\n")
    
    # Create filtered dataset with strict criteria
    print("1. Creating filtered dataset with strict criteria...")
    dataset_strict = create_filtered_dataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        min_bbox_width=50,  # Stricter than default
        min_bbox_height=50,
        min_aspect_ratio=0.33,  # 1:3 ratio
        max_aspect_ratio=3.0,   # 3:1 ratio
        verbose=True
    )
    
    print(f"\nFiltered dataset size: {len(dataset_strict)}")
    
    # Create dataloader
    collate_fn = create_collate_fn()
    dataloader = DataLoader(
        dataset_strict, 
        batch_size=2, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Test model forward pass
    print("\n2. Testing model forward pass...")
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48)
    ).to(device)
    model.eval()
    
    # Get a batch
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    roi_boxes = batch['roi_boxes'].to(device)
    
    print(f"Images shape: {images.shape}")
    print(f"ROI boxes shape: {roi_boxes.shape}")
    print(f"ROI boxes:\n{roi_boxes}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(images, roi_boxes)
    
    print(f"\nModel outputs:")
    if isinstance(outputs, tuple):
        masks_output, aux_outputs = outputs
        print(f"  Masks output shape: {masks_output.shape}")
        print(f"  Aux outputs keys: {aux_outputs.keys() if isinstance(aux_outputs, dict) else 'Not a dict'}")
        
        # Analyze the mask predictions
        final_probs = torch.softmax(masks_output, dim=1)
        
        # Check auxiliary outputs if available
        if isinstance(aux_outputs, dict) and 'bg_fg_logits' in aux_outputs:
            bg_fg_logits = aux_outputs['bg_fg_logits']
            print(f"  bg_fg_logits shape: {bg_fg_logits.shape}")
            print(f"  bg_fg_logits range: [{bg_fg_logits.min():.3f}, {bg_fg_logits.max():.3f}]")
            bg_fg_probs = torch.sigmoid(bg_fg_logits)
            print(f"  bg_fg_probs range: [{bg_fg_probs.min():.3f}, {bg_fg_probs.max():.3f}]")
    else:
        print(f"  Output type: {type(outputs)}")
        print(f"  Output shape: {outputs.shape if hasattr(outputs, 'shape') else 'No shape'}")
        final_probs = torch.softmax(outputs, dim=1)
    print(f"\nFinal mask probabilities:")
    for i in range(3):
        print(f"  Class {i}: mean={final_probs[:, i].mean():.3f}, "
              f"max={final_probs[:, i].max():.3f}")
    
    # Compare with original dataset
    print("\n\n3. Comparing with original dataset...")
    from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
    
    dataset_original = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48)
    )
    
    print(f"Original dataset size: {len(dataset_original)}")
    print(f"Filtered dataset size: {len(dataset_strict)}")
    print(f"Samples removed: {len(dataset_original) - len(dataset_strict)} "
          f"({(1 - len(dataset_strict)/len(dataset_original))*100:.1f}%)")
    
    # Test aspect ratio distribution in filtered dataset
    print("\n4. Analyzing filtered dataset bbox distribution...")
    from pycocotools.coco import COCO
    import numpy as np
    
    coco = dataset_strict.coco
    widths = []
    heights = []
    aspect_ratios = []
    
    for sample in dataset_strict.samples:
        ann = coco.loadAnns([sample['target_ann_id']])[0]
        x, y, w, h = ann['bbox']
        widths.append(w)
        heights.append(h)
        aspect_ratios.append(w/h if h > 0 else 0)
    
    widths = np.array(widths)
    heights = np.array(heights)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"\nFiltered dataset statistics:")
    print(f"  Width: min={widths.min():.1f}, max={widths.max():.1f}, mean={widths.mean():.1f}")
    print(f"  Height: min={heights.min():.1f}, max={heights.max():.1f}, mean={heights.mean():.1f}")
    print(f"  Aspect ratio: min={aspect_ratios.min():.3f}, max={aspect_ratios.max():.3f}, mean={aspect_ratios.mean():.3f}")


if __name__ == "__main__":
    test_filtered_dataset()