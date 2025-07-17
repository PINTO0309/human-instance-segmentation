"""Analyze full COCO dataset efficiently with batch processing."""

import json
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm
import multiprocessing as mp
from functools import partial


def process_image_batch(img_ids, coco, batch_idx):
    """Process a batch of images for efficiency."""
    batch_stats = {
        'pixel_counts': {'background': 0, 'target': 0, 'non_target': 0},
        'overlap_ious': [],
        'roi_area_ratios': [],
        'instances_per_roi': [],
        'total_instances': 0,
        'images_with_multiple': 0,
        'overlapping_pairs': 0
    }
    
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        height = img_info['height']
        width = img_info['width']
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        if not anns:
            continue
            
        batch_stats['total_instances'] += len(anns)
        if len(anns) > 1:
            batch_stats['images_with_multiple'] += 1
        
        # Process annotations
        instance_masks = []
        bboxes = []
        
        for ann in anns:
            if isinstance(ann['segmentation'], list):
                mask = coco.annToMask(ann)
            else:
                mask = maskUtils.decode(ann['segmentation'])
            instance_masks.append(mask)
            bboxes.append(ann['bbox'])
        
        # Calculate overlaps efficiently
        if len(instance_masks) > 1:
            for i in range(len(instance_masks)):
                for j in range(i + 1, len(instance_masks)):
                    intersection = np.logical_and(instance_masks[i], instance_masks[j])
                    if np.any(intersection):
                        union = np.logical_or(instance_masks[i], instance_masks[j])
                        iou = np.sum(intersection) / np.sum(union)
                        batch_stats['overlap_ious'].append(iou)
                        batch_stats['overlapping_pairs'] += 1
        
        # ROI-based processing
        for target_idx in range(len(anns)):
            x, y, w, h = bboxes[target_idx]
            padding = 0.1
            x1 = max(0, int(x - w * padding))
            y1 = max(0, int(y - h * padding))
            x2 = min(width, int(x + w * (1 + padding)))
            y2 = min(height, int(y + h * (1 + padding)))
            
            roi_area = (x2 - x1) * (y2 - y1)
            batch_stats['roi_area_ratios'].append(roi_area / (width * height))
            
            # Count pixels in ROI
            roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            
            # Target mask
            target_mask_roi = instance_masks[target_idx][y1:y2, x1:x2]
            roi_mask[target_mask_roi > 0] = 1
            
            # Non-target masks
            instances_in_roi = 1
            for other_idx in range(len(anns)):
                if other_idx != target_idx:
                    other_mask_roi = instance_masks[other_idx][y1:y2, x1:x2]
                    roi_mask[(other_mask_roi > 0) & (roi_mask == 0)] = 2
                    if np.any(other_mask_roi > 0):
                        instances_in_roi += 1
            
            batch_stats['instances_per_roi'].append(instances_in_roi)
            
            # Count pixels
            unique, counts = np.unique(roi_mask, return_counts=True)
            for val, count in zip(unique, counts):
                if val == 0:
                    batch_stats['pixel_counts']['background'] += count
                elif val == 1:
                    batch_stats['pixel_counts']['target'] += count
                elif val == 2:
                    batch_stats['pixel_counts']['non_target'] += count
    
    return batch_stats


def analyze_full_dataset():
    """Analyze full COCO dataset efficiently."""
    annotation_file = "data/annotations/instances_train2017_person_only_no_crowd.json"
    output_file = "data_analyze.json"
    
    print(f"Loading annotations from {annotation_file}")
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    print(f"Total images: {len(img_ids)}")
    
    # Process in batches
    batch_size = 1000
    batches = [img_ids[i:i+batch_size] for i in range(0, len(img_ids), batch_size)]
    
    # Initialize combined stats
    combined_stats = {
        'total_images': len(img_ids),
        'total_instances': 0,
        'images_with_multiple_instances': 0,
        'pixel_counts': {'background': 0, 'target': 0, 'non_target': 0},
        'overlap_ious': [],
        'roi_area_ratios': [],
        'instances_per_roi': [],
        'overlapping_pairs': 0
    }
    
    # Process batches
    for i, batch in enumerate(tqdm(batches, desc="Processing batches")):
        batch_stats = process_image_batch(batch, coco, i)
        
        # Combine stats
        combined_stats['total_instances'] += batch_stats['total_instances']
        combined_stats['images_with_multiple_instances'] += batch_stats['images_with_multiple']
        combined_stats['overlapping_pairs'] += batch_stats['overlapping_pairs']
        
        for key in ['background', 'target', 'non_target']:
            combined_stats['pixel_counts'][key] += batch_stats['pixel_counts'][key]
        
        combined_stats['overlap_ious'].extend(batch_stats['overlap_ious'])
        combined_stats['roi_area_ratios'].extend(batch_stats['roi_area_ratios'])
        combined_stats['instances_per_roi'].extend(batch_stats['instances_per_roi'])
    
    # Calculate final statistics
    total_pixels = sum(combined_stats['pixel_counts'].values())
    pixel_ratios = {k: v/total_pixels for k, v in combined_stats['pixel_counts'].items()}
    
    # Calculate class weights
    eps = 1e-3
    freq_weights = {k: 1.0/(ratio + eps) for k, ratio in pixel_ratios.items()}
    weight_sum = sum(freq_weights.values())
    recommended_weights = {k: v/weight_sum * 3.0 for k, v in freq_weights.items()}
    
    log_weights = {k: np.log(1.0/(ratio + eps)) for k, ratio in pixel_ratios.items()}
    log_weight_sum = sum(log_weights.values())
    recommended_log_weights = {k: v/log_weight_sum * 3.0 for k, v in log_weights.items()}
    
    # Prepare final stats
    final_stats = {
        'total_images': combined_stats['total_images'],
        'total_instances': combined_stats['total_instances'],
        'images_with_multiple_instances': combined_stats['images_with_multiple_instances'],
        'pixel_counts': combined_stats['pixel_counts'],
        'pixel_ratios': pixel_ratios,
        'instance_overlap_stats': {
            'total_overlapping_pairs': combined_stats['overlapping_pairs'],
            'avg_iou_between_instances': float(np.mean(combined_stats['overlap_ious'])) if combined_stats['overlap_ious'] else 0.0,
            'max_iou_between_instances': float(np.max(combined_stats['overlap_ious'])) if combined_stats['overlap_ious'] else 0.0
        },
        'roi_stats': {
            'avg_roi_area_ratio': float(np.mean(combined_stats['roi_area_ratios'])),
            'avg_instances_per_roi': float(np.mean(combined_stats['instances_per_roi']))
        },
        'recommended_class_weights': recommended_weights,
        'recommended_log_class_weights': recommended_log_weights
    }
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print("\nSummary:")
    print(f"Total images: {final_stats['total_images']}")
    print(f"Total instances: {final_stats['total_instances']}")
    print(f"Images with multiple instances: {final_stats['images_with_multiple_instances']}")
    print(f"\nPixel ratios:")
    for class_name, ratio in final_stats['pixel_ratios'].items():
        print(f"  {class_name}: {ratio:.4f}")
    print(f"\nRecommended class weights:")
    for class_name, weight in final_stats['recommended_class_weights'].items():
        print(f"  {class_name}: {weight:.4f}")
    print(f"\nRecommended log-scaled class weights:")
    for class_name, weight in final_stats['recommended_log_class_weights'].items():
        print(f"  {class_name}: {weight:.4f}")


if __name__ == "__main__":
    analyze_full_dataset()