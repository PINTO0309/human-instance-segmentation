#!/usr/bin/env python3
"""Helper script to select test images with large ROIs from val2017."""

import json
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch


def get_test_images_by_person_count_large_roi(val_annotation_path='data/annotations/instances_val2017_person_only_no_crowd.json'):
    """Get specific test images with 1, 2, 3, and 5 people, prioritizing larger ROIs.
    
    Returns:
        Dict mapping person count to selected image IDs
    """
    print("Loading val2017 annotations...")
    coco = COCO(val_annotation_path)
    
    # Target person counts
    target_counts = [1, 2, 3, 5]
    
    # Collect images with ROI size info
    img_by_count_with_roi = {count: [] for count in target_counts}
    
    for img_id in coco.imgs.keys():
        img_info = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Filter non-crowd annotations
        valid_anns = [ann for ann in anns if not ann.get('iscrowd', 0)]
        person_count = len(valid_anns)
        
        if person_count in target_counts:
            # Calculate average ROI area relative to image size
            total_roi_area = 0
            img_area = img_info['height'] * img_info['width']
            
            for ann in valid_anns:
                if 'area' in ann:
                    total_roi_area += ann['area']
                elif 'bbox' in ann:
                    # Calculate area from bbox if area not available
                    bbox = ann['bbox']
                    total_roi_area += bbox[2] * bbox[3]  # width * height
            
            # Average ROI area ratio (0-1)
            avg_roi_ratio = (total_roi_area / person_count) / img_area if person_count > 0 else 0
            
            # Store image info
            img_by_count_with_roi[person_count].append({
                'img_id': img_id,
                'file_name': img_info['file_name'],
                'roi_ratio': avg_roi_ratio,
                'total_roi_area': total_roi_area,
                'img_area': img_area
            })
    
    # Sort by ROI size (larger first) and select images
    selected_images = {}
    
    for count in target_counts:
        if img_by_count_with_roi[count]:
            # Sort by ROI ratio
            img_by_count_with_roi[count].sort(key=lambda x: x['roi_ratio'], reverse=True)
            
            # Show statistics
            roi_sizes = [x['roi_ratio'] for x in img_by_count_with_roi[count]]
            print(f"\n{count} person(s): {len(img_by_count_with_roi[count])} images found")
            print(f"  ROI size range: {min(roi_sizes)*100:.1f}% - {max(roi_sizes)*100:.1f}% of image")
            print(f"  Median ROI size: {np.median(roi_sizes)*100:.1f}% of image")
            
            # Select images with larger ROIs (from top 30% of sorted list)
            top_30_percent_idx = max(1, len(img_by_count_with_roi[count]) // 3)
            
            # Different selection indices for variety
            if count == 1:
                # Select from top 30% - index 5
                idx = min(5, top_30_percent_idx - 1)
            elif count == 2:
                # Select from top 30% - index 8
                idx = min(8, top_30_percent_idx - 1)
            elif count == 3:
                # Select from top 30% - index 10
                idx = min(10, top_30_percent_idx - 1)
            elif count == 5:
                # Select from top 30% - index 3
                idx = min(3, top_30_percent_idx - 1)
            else:
                idx = 0
            
            selected = img_by_count_with_roi[count][idx]
            selected_images[count] = selected
            
            print(f"  Selected: {selected['file_name']}")
            print(f"    Image ID: {selected['img_id']}")
            print(f"    ROI ratio: {selected['roi_ratio']*100:.1f}% of image")
    
    return selected_images


if __name__ == "__main__":
    selected = get_test_images_by_person_count_large_roi()
    
    # Save selection for reference
    selection_info = {}
    for count, img_info in selected.items():
        selection_info[f"{count}_person"] = {
            'img_id': int(img_info['img_id']),
            'file_name': img_info['file_name'],
            'roi_percentage': float(img_info['roi_ratio'] * 100)
        }
    
    with open('selected_test_images_large_roi.json', 'w') as f:
        json.dump(selection_info, f, indent=2)
    
    print("\nSelection saved to selected_test_images_large_roi.json")