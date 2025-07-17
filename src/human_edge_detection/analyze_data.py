"""Analyze COCO annotations to calculate class distribution for 3-class segmentation."""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
from tqdm import tqdm


def analyze_coco_annotations(annotation_file: str, image_dir: str):
    """Analyze COCO annotations for 3-class segmentation task.
    
    Classes:
    - 0: Background
    - 1: Target mask (primary instance in ROI)
    - 2: Non-target mask (other instances in ROI)
    """
    print(f"Loading annotations from {annotation_file}")
    coco = COCO(annotation_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Total images: {len(img_ids)}")
    
    # Initialize statistics
    stats = {
        'total_images': len(img_ids),
        'total_instances': 0,
        'images_with_multiple_instances': 0,
        'pixel_counts': {
            'background': 0,
            'target': 0,
            'non_target': 0
        },
        'pixel_ratios': {},
        'instance_overlap_stats': {
            'total_overlapping_pairs': 0,
            'avg_iou_between_instances': 0.0,
            'max_iou_between_instances': 0.0
        },
        'roi_stats': {
            'avg_roi_area_ratio': 0.0,  # ROI area / image area
            'avg_instances_per_roi': 0.0
        },
        'per_image_stats': []
    }
    
    overlap_ious = []
    roi_area_ratios = []
    instances_per_roi_list = []
    
    # Process each image
    for img_id in tqdm(img_ids, desc="Analyzing images"):
        img_info = coco.loadImgs(img_id)[0]
        height = img_info['height']
        width = img_info['width']
        
        # Get all annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            continue
            
        stats['total_instances'] += len(anns)
        
        # Track per-image statistics
        image_stats = {
            'image_id': img_id,
            'num_instances': len(anns),
            'has_overlaps': False,
            'max_overlap_iou': 0.0
        }
        
        if len(anns) > 1:
            stats['images_with_multiple_instances'] += 1
            
        # Create full image mask for all instances
        full_mask = np.zeros((height, width), dtype=np.uint8)
        instance_masks = []
        bboxes = []
        
        # Process each annotation
        for i, ann in enumerate(anns):
            # Convert segmentation to binary mask
            if isinstance(ann['segmentation'], list):
                # Polygon format
                mask = coco.annToMask(ann)
            else:
                # RLE format
                mask = maskUtils.decode(ann['segmentation'])
                
            instance_masks.append(mask)
            bboxes.append(ann['bbox'])  # [x, y, w, h]
            
        # Calculate overlap statistics between instances
        if len(instance_masks) > 1:
            for i in range(len(instance_masks)):
                for j in range(i + 1, len(instance_masks)):
                    intersection = np.logical_and(instance_masks[i], instance_masks[j])
                    union = np.logical_or(instance_masks[i], instance_masks[j])
                    iou = np.sum(intersection) / (np.sum(union) + 1e-6)
                    
                    if iou > 0:
                        overlap_ious.append(iou)
                        stats['instance_overlap_stats']['total_overlapping_pairs'] += 1
                        image_stats['has_overlaps'] = True
                        image_stats['max_overlap_iou'] = max(image_stats['max_overlap_iou'], iou)
        
        # Simulate ROI-based processing
        # For each instance, consider it as the target and others as non-targets
        for target_idx in range(len(anns)):
            # Create ROI from target bbox with some padding
            x, y, w, h = bboxes[target_idx]
            padding = 0.1  # 10% padding
            x1 = max(0, int(x - w * padding))
            y1 = max(0, int(y - h * padding))
            x2 = min(width, int(x + w * (1 + padding)))
            y2 = min(height, int(y + h * (1 + padding)))
            
            roi_area = (x2 - x1) * (y2 - y1)
            roi_area_ratio = roi_area / (width * height)
            roi_area_ratios.append(roi_area_ratio)
            
            # Create 3-class mask for this ROI
            roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            
            # Target mask (class 1)
            target_mask_roi = instance_masks[target_idx][y1:y2, x1:x2]
            roi_mask[target_mask_roi > 0] = 1
            
            # Non-target masks (class 2)
            instances_in_roi = 0
            for other_idx in range(len(anns)):
                if other_idx != target_idx:
                    other_mask_roi = instance_masks[other_idx][y1:y2, x1:x2]
                    # Only mark as non-target if not already marked as target
                    roi_mask[(other_mask_roi > 0) & (roi_mask == 0)] = 2
                    if np.any(other_mask_roi > 0):
                        instances_in_roi += 1
            
            instances_per_roi_list.append(instances_in_roi + 1)  # +1 for target
            
            # Count pixels
            unique, counts = np.unique(roi_mask, return_counts=True)
            pixel_count_dict = dict(zip(unique, counts))
            
            stats['pixel_counts']['background'] += pixel_count_dict.get(0, 0)
            stats['pixel_counts']['target'] += pixel_count_dict.get(1, 0)
            stats['pixel_counts']['non_target'] += pixel_count_dict.get(2, 0)
        
        stats['per_image_stats'].append(image_stats)
    
    # Calculate final statistics
    total_pixels = sum(stats['pixel_counts'].values())
    stats['pixel_ratios'] = {
        'background': stats['pixel_counts']['background'] / total_pixels,
        'target': stats['pixel_counts']['target'] / total_pixels,
        'non_target': stats['pixel_counts']['non_target'] / total_pixels
    }
    
    if overlap_ious:
        stats['instance_overlap_stats']['avg_iou_between_instances'] = np.mean(overlap_ious)
        stats['instance_overlap_stats']['max_iou_between_instances'] = np.max(overlap_ious)
    
    if roi_area_ratios:
        stats['roi_stats']['avg_roi_area_ratio'] = np.mean(roi_area_ratios)
    
    if instances_per_roi_list:
        stats['roi_stats']['avg_instances_per_roi'] = np.mean(instances_per_roi_list)
    
    # Add class weight recommendations based on pixel ratios
    # Inverse frequency weighting with smoothing
    eps = 1e-3
    freq_weights = {
        'background': 1.0 / (stats['pixel_ratios']['background'] + eps),
        'target': 1.0 / (stats['pixel_ratios']['target'] + eps),
        'non_target': 1.0 / (stats['pixel_ratios']['non_target'] + eps)
    }
    
    # Normalize weights
    weight_sum = sum(freq_weights.values())
    stats['recommended_class_weights'] = {
        k: v / weight_sum * 3.0 for k, v in freq_weights.items()
    }
    
    # Log-scaled weights (more moderate)
    log_weights = {
        'background': np.log(1.0 / (stats['pixel_ratios']['background'] + eps)),
        'target': np.log(1.0 / (stats['pixel_ratios']['target'] + eps)),
        'non_target': np.log(1.0 / (stats['pixel_ratios']['non_target'] + eps))
    }
    weight_sum = sum(log_weights.values())
    stats['recommended_log_class_weights'] = {
        k: v / weight_sum * 3.0 for k, v in log_weights.items()
    }
    
    return stats


def main():
    """Run analysis on training data."""
    # Start with small dataset for testing
    annotation_file = "data/annotations/instances_train2017_person_only_no_crowd_100.json"
    image_dir = "data/images/train2017"
    output_file = "data_analyze_100.json"
    
    # Check if files exist
    if not Path(annotation_file).exists():
        print(f"Annotation file not found: {annotation_file}")
        return
    
    # Run analysis
    stats = analyze_coco_annotations(annotation_file, image_dir)
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj
    
    stats_native = convert_to_native(stats)
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(stats_native, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print("\nSummary:")
    print(f"Total images: {stats['total_images']}")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Images with multiple instances: {stats['images_with_multiple_instances']}")
    print(f"\nPixel ratios:")
    for class_name, ratio in stats['pixel_ratios'].items():
        print(f"  {class_name}: {ratio:.4f}")
    print(f"\nRecommended class weights:")
    for class_name, weight in stats['recommended_class_weights'].items():
        print(f"  {class_name}: {weight:.4f}")
    print(f"\nRecommended log-scaled class weights:")
    for class_name, weight in stats['recommended_log_class_weights'].items():
        print(f"  {class_name}: {weight:.4f}")


if __name__ == "__main__":
    main()