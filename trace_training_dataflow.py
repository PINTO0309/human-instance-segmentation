"""Trace the complete data flow during training to identify issues."""

import torch
import torch.nn as nn
import numpy as np
from pycocotools.coco import COCO

from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from torch.utils.data import DataLoader
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)
from src.human_edge_detection.advanced.hierarchical_segmentation import HierarchicalLoss

def trace_dataflow():
    """Trace the complete data flow from dataset to loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== Setting up dataset ===")
    # Create dataset with the same settings as training
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),  # Default training size
        mask_size=(64, 48),     # Non-square mask size
        roi_padding=0.0         # Default padding
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=create_collate_fn()
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    print("\n=== Analyzing batch structure ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
        elif isinstance(value, list):
            print(f"{key}: list of {len(value)} items")
        else:
            print(f"{key}: {type(value)}")
    
    # Extract data
    images = batch['image'].to(device)
    rois = batch['roi_boxes'].to(device)
    masks = batch['roi_masks'].to(device)
    
    print("\n=== Detailed ROI analysis ===")
    print(f"ROIs shape: {rois.shape}")
    for i in range(rois.shape[0]):
        batch_idx, x1, y1, x2, y2 = rois[i].tolist()
        print(f"ROI {i}: batch_idx={int(batch_idx)}, bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  Width: {x2-x1:.1f}, Height: {y2-y1:.1f}")
        print(f"  Normalized: x=[{x1/images.shape[3]:.3f}, {x2/images.shape[3]:.3f}], y=[{y1/images.shape[2]:.3f}, {y2/images.shape[2]:.3f}]")
    
    print("\n=== Creating model ===")
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    model.eval()
    
    print("\n=== Forward pass ===")
    with torch.no_grad():
        # Full forward pass
        predictions, aux_outputs = model(images, rois)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    
    # Convert to probabilities
    probs = torch.softmax(predictions, dim=1)
    for i in range(3):
        class_name = ['Background', 'Target', 'Non-target'][i]
        print(f"{class_name} prob: mean={probs[:, i].mean():.3f}, max={probs[:, i].max():.3f}")
    
    print("\n=== Auxiliary outputs ===")
    for key, value in aux_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
        else:
            print(f"{key}: {type(value)}")
    
    print("\n=== Checking pre-trained UNet output ===")
    # Direct UNet output
    full_unet_output = model.pretrained_unet(images)
    print(f"Full UNet output: shape={full_unet_output.shape}, range=[{full_unet_output.min():.3f}, {full_unet_output.max():.3f}]")
    full_fg_prob = torch.sigmoid(full_unet_output)
    print(f"Full foreground prob: mean={full_fg_prob.mean():.3f}, max={full_fg_prob.max():.3f}")
    
    # Extract ROI regions from UNet output
    roi_masks_from_unet = model.roi_align_mask(full_unet_output, rois, 64, 48)
    print(f"\nROI masks from UNet: shape={roi_masks_from_unet.shape}")
    print(f"ROI masks range: [{roi_masks_from_unet.min():.3f}, {roi_masks_from_unet.max():.3f}]")
    roi_fg_probs = torch.sigmoid(roi_masks_from_unet)
    print(f"ROI foreground probs: mean={roi_fg_probs.mean():.3f}, max={roi_fg_probs.max():.3f}")
    
    print("\n=== Computing loss ===")
    # Create loss function
    loss_fn = HierarchicalLoss(
        bg_weight=1.5,
        fg_weight=1.5,
        target_weight=1.2,
        consistency_weight=0.3
    )
    
    # Compute loss
    try:
        loss, loss_dict = loss_fn(predictions, masks, aux_outputs)
        print(f"Total loss: {loss.item():.4f}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"Error computing loss: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Checking target masks ===")
    print(f"Target masks shape: {masks.shape}")
    print(f"Unique values in masks: {torch.unique(masks)}")
    for i in range(masks.shape[0]):
        unique_vals, counts = torch.unique(masks[i], return_counts=True)
        total_pixels = masks[i].numel()
        print(f"Mask {i} distribution:")
        for val, count in zip(unique_vals, counts):
            print(f"  Class {val}: {count} pixels ({count/total_pixels*100:.1f}%)")

def check_model_components():
    """Check individual model components."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n=== Testing RGB Feature Extractor ===")
    from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import RGBFeatureExtractor
    
    rgb_extractor = RGBFeatureExtractor(
        in_channels=3,
        out_channels=256,
        roi_size=(64, 48),
        num_layers=4
    ).to(device)
    
    # Test input
    test_roi = torch.randn(2, 3, 64, 48).to(device)
    features = rgb_extractor(test_roi)
    print(f"RGB features: shape={features.shape}, range=[{features.min():.3f}, {features.max():.3f}]")
    
    print("\n=== Testing Segmentation Head ===")
    from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import PretrainedUNetGuidedSegmentationHead
    
    seg_head = PretrainedUNetGuidedSegmentationHead(
        in_channels=256,
        mid_channels=256,
        num_classes=3,
        mask_size=(64, 48),
        use_attention_module=True
    ).to(device)
    
    # Test with features and binary mask
    bg_fg_mask = torch.randn(2, 1, 64, 48).to(device)
    predictions, aux = seg_head(features, bg_fg_mask)
    print(f"Segmentation predictions: shape={predictions.shape}, range=[{predictions.min():.3f}, {predictions.max():.3f}]")

def main():
    print("=== Tracing Training Data Flow ===\n")
    
    trace_dataflow()
    
    print("\n" + "="*50 + "\n")
    check_model_components()

if __name__ == "__main__":
    main()