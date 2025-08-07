#!/usr/bin/env python3
"""Find images with exactly 5 people and reasonable ROI sizes."""

import json
from pycocotools.coco import COCO
import numpy as np
from pathlib import Path

def analyze_5person_images():
    """Find and analyze all images with exactly 5 people."""
    
    # Load validation annotations
    ann_file = "data/annotations/instances_val2017_person_only_no_crowd.json"
    coco = COCO(ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    
    # Find images with exactly 5 people
    five_person_images = []
    
    for img_id in img_ids:
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=1)  # catId=1 for person
        anns = coco.loadAnns(ann_ids)
        
        # Count non-crowd persons
        person_count = sum(1 for ann in anns if not ann.get('iscrowd', 0))
        
        if person_count == 5:
            # Get image info
            img_info = coco.imgs[img_id]
            img_height = img_info['height']
            img_width = img_info['width']
            img_area = img_height * img_width
            
            # Calculate ROI statistics
            roi_areas = []
            roi_percentages = []
            
            for ann in anns:
                if not ann.get('iscrowd', 0):
                    bbox = ann['bbox']
                    roi_area = bbox[2] * bbox[3]  # width * height
                    roi_areas.append(roi_area)
                    roi_percentages.append(100 * roi_area / img_area)
            
            # Calculate average and minimum ROI percentage
            avg_roi_pct = np.mean(roi_percentages)
            min_roi_pct = np.min(roi_percentages)
            max_roi_pct = np.max(roi_percentages)
            total_roi_pct = np.sum(roi_percentages)
            
            five_person_images.append({
                'img_id': img_id,
                'file_name': img_info['file_name'],
                'img_width': img_width,
                'img_height': img_height,
                'avg_roi_pct': avg_roi_pct,
                'min_roi_pct': min_roi_pct,
                'max_roi_pct': max_roi_pct,
                'total_roi_pct': total_roi_pct,
                'roi_percentages': roi_percentages
            })
    
    # Sort by average ROI percentage (larger is better)
    five_person_images.sort(key=lambda x: x['avg_roi_pct'], reverse=True)
    
    print(f"Found {len(five_person_images)} images with exactly 5 people\n")
    
    # Show top 20 candidates with good ROI sizes
    print("Top 20 images with 5 people (sorted by average ROI size):")
    print("-" * 100)
    
    for i, img_data in enumerate(five_person_images[:20]):
        print(f"{i+1:2d}. Image ID: {img_data['img_id']:6d} | File: {img_data['file_name']}")
        print(f"    Size: {img_data['img_width']}x{img_data['img_height']}")
        print(f"    ROI Coverage - Avg: {img_data['avg_roi_pct']:.1f}% | Min: {img_data['min_roi_pct']:.1f}% | Max: {img_data['max_roi_pct']:.1f}% | Total: {img_data['total_roi_pct']:.1f}%")
        print(f"    Individual ROIs: {[f'{p:.1f}%' for p in img_data['roi_percentages']]}")
        print()
    
    # Also find images where minimum ROI is reasonably large
    print("\nImages with 5 people where ALL ROIs are reasonably large (min ROI > 1%):")
    print("-" * 100)
    
    filtered_images = [img for img in five_person_images if img['min_roi_pct'] > 1.0]
    filtered_images.sort(key=lambda x: x['min_roi_pct'], reverse=True)
    
    for i, img_data in enumerate(filtered_images[:10]):
        print(f"{i+1:2d}. Image ID: {img_data['img_id']:6d} | File: {img_data['file_name']}")
        print(f"    Size: {img_data['img_width']}x{img_data['img_height']}")
        print(f"    ROI Coverage - Avg: {img_data['avg_roi_pct']:.1f}% | Min: {img_data['min_roi_pct']:.1f}% | Max: {img_data['max_roi_pct']:.1f}% | Total: {img_data['total_roi_pct']:.1f}%")
        print(f"    Individual ROIs: {[f'{p:.1f}%' for p in img_data['roi_percentages']]}")
        print()
    
    # Return best candidates
    return filtered_images[:5] if filtered_images else five_person_images[:5]

if __name__ == "__main__":
    best_candidates = analyze_5person_images()
    
    if best_candidates:
        print("\n" + "=" * 100)
        print("RECOMMENDED IMAGE:")
        print("=" * 100)
        best = best_candidates[0]
        print(f"Image ID: {best['img_id']}")
        print(f"File: {best['file_name']}")
        print(f"Minimum ROI: {best['min_roi_pct']:.1f}% (ensures all people are visible)")
        print(f"Average ROI: {best['avg_roi_pct']:.1f}%")