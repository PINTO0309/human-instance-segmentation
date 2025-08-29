#!/usr/bin/env python3
"""Analyze foreground vs background pixel ratio in COCO person dataset."""

import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from tqdm import tqdm
import argparse


def analyze_pixel_ratio(annotation_file):
    """Calculate foreground and background pixel ratios from COCO annotations.
    
    Args:
        annotation_file: Path to COCO annotation JSON file
    
    Returns:
        Dictionary with statistics
    """
    print(f"Loading annotations from {annotation_file}...")
    coco = COCO(annotation_file)
    
    # Get person category ID
    person_cat_id = None
    for cat in coco.cats.values():
        if cat['name'] == 'person':
            person_cat_id = cat['id']
            break
    
    if person_cat_id is None:
        raise ValueError("Person category not found in annotations")
    
    # Get all images with person annotations
    img_ids_with_person = set()
    for ann in coco.anns.values():
        if ann['category_id'] == person_cat_id and not ann.get('iscrowd', 0):
            img_ids_with_person.add(ann['image_id'])
    
    img_ids = list(img_ids_with_person)
    print(f"Found {len(img_ids)} images with person annotations")
    
    # Statistics accumulators
    total_pixels = 0
    total_foreground = 0
    total_background = 0
    
    image_ratios = []
    roi_counts = []
    
    print("Processing images...")
    for img_id in tqdm(img_ids):
        # Get image info
        img_info = coco.imgs[img_id]
        height = img_info['height']
        width = img_info['width']
        img_pixels = height * width
        
        # Create binary mask for all person instances
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get all person annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=person_cat_id, iscrowd=None)
        anns = coco.loadAnns(ann_ids)
        
        # Count non-crowd ROIs
        roi_count = 0
        for ann in anns:
            if not ann.get('iscrowd', 0):
                roi_count += 1
                # Convert segmentation to binary mask
                if 'segmentation' in ann and ann['segmentation']:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        rle = mask_utils.frPyObjects(ann['segmentation'], height, width)
                        if isinstance(rle, list):
                            rle = mask_utils.merge(rle)
                        binary_mask = mask_utils.decode(rle)
                    else:
                        # RLE format
                        binary_mask = mask_utils.decode(ann['segmentation'])
                    mask = np.maximum(mask, binary_mask)
        
        # Count pixels
        foreground_pixels = np.sum(mask > 0)
        background_pixels = img_pixels - foreground_pixels
        
        # Accumulate totals
        total_pixels += img_pixels
        total_foreground += foreground_pixels
        total_background += background_pixels
        
        # Store per-image ratio
        if img_pixels > 0:
            image_ratios.append(foreground_pixels / img_pixels)
            roi_counts.append(roi_count)
    
    # Calculate statistics
    global_fg_ratio = total_foreground / total_pixels if total_pixels > 0 else 0
    global_bg_ratio = total_background / total_pixels if total_pixels > 0 else 0
    
    image_ratios = np.array(image_ratios)
    roi_counts = np.array(roi_counts)
    
    stats = {
        'num_images': len(img_ids),
        'total_pixels': int(total_pixels),
        'total_foreground_pixels': int(total_foreground),
        'total_background_pixels': int(total_background),
        'global_foreground_ratio': float(global_fg_ratio),
        'global_background_ratio': float(global_bg_ratio),
        'per_image_fg_ratio_mean': float(np.mean(image_ratios)),
        'per_image_fg_ratio_std': float(np.std(image_ratios)),
        'per_image_fg_ratio_min': float(np.min(image_ratios)),
        'per_image_fg_ratio_max': float(np.max(image_ratios)),
        'per_image_fg_ratio_median': float(np.median(image_ratios)),
        'roi_count_mean': float(np.mean(roi_counts)),
        'roi_count_std': float(np.std(roi_counts)),
        'roi_count_min': int(np.min(roi_counts)),
        'roi_count_max': int(np.max(roi_counts)),
        'roi_count_median': float(np.median(roi_counts))
    }
    
    # Calculate percentiles
    percentiles = [10, 25, 75, 90]
    for p in percentiles:
        stats[f'per_image_fg_ratio_p{p}'] = float(np.percentile(image_ratios, p))
    
    # ROI count distribution
    unique_counts, count_freq = np.unique(roi_counts, return_counts=True)
    roi_distribution = {}
    for count, freq in zip(unique_counts, count_freq):
        roi_distribution[str(count)] = int(freq)
    stats['roi_count_distribution'] = roi_distribution
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze pixel ratios in COCO person dataset')
    parser.add_argument('--annotation', type=str, 
                       default='data/annotations/instances_train2017_person_only_no_crowd.json',
                       help='Path to COCO annotation file')
    parser.add_argument('--output', type=str, default='pixel_ratio_analysis.json',
                       help='Output JSON file for results')
    args = parser.parse_args()
    
    # Analyze pixel ratios
    stats = analyze_pixel_ratio(args.annotation)
    
    # Print results
    print("\n" + "="*60)
    print("PIXEL RATIO ANALYSIS RESULTS")
    print("="*60)
    print(f"Number of images: {stats['num_images']:,}")
    print(f"Total pixels: {stats['total_pixels']:,}")
    print(f"Total foreground pixels: {stats['total_foreground_pixels']:,}")
    print(f"Total background pixels: {stats['total_background_pixels']:,}")
    print()
    print("GLOBAL RATIOS:")
    print(f"  Foreground ratio: {stats['global_foreground_ratio']:.4f} ({stats['global_foreground_ratio']*100:.2f}%)")
    print(f"  Background ratio: {stats['global_background_ratio']:.4f} ({stats['global_background_ratio']*100:.2f}%)")
    print(f"  FG:BG ratio: 1:{stats['global_background_ratio']/stats['global_foreground_ratio']:.2f}")
    print()
    print("PER-IMAGE FOREGROUND RATIO STATISTICS:")
    print(f"  Mean: {stats['per_image_fg_ratio_mean']:.4f} ({stats['per_image_fg_ratio_mean']*100:.2f}%)")
    print(f"  Std: {stats['per_image_fg_ratio_std']:.4f}")
    print(f"  Min: {stats['per_image_fg_ratio_min']:.4f} ({stats['per_image_fg_ratio_min']*100:.2f}%)")
    print(f"  Max: {stats['per_image_fg_ratio_max']:.4f} ({stats['per_image_fg_ratio_max']*100:.2f}%)")
    print(f"  Median: {stats['per_image_fg_ratio_median']:.4f} ({stats['per_image_fg_ratio_median']*100:.2f}%)")
    print(f"  P10: {stats['per_image_fg_ratio_p10']:.4f} ({stats['per_image_fg_ratio_p10']*100:.2f}%)")
    print(f"  P25: {stats['per_image_fg_ratio_p25']:.4f} ({stats['per_image_fg_ratio_p25']*100:.2f}%)")
    print(f"  P75: {stats['per_image_fg_ratio_p75']:.4f} ({stats['per_image_fg_ratio_p75']*100:.2f}%)")
    print(f"  P90: {stats['per_image_fg_ratio_p90']:.4f} ({stats['per_image_fg_ratio_p90']*100:.2f}%)")
    print()
    print("ROI COUNT STATISTICS:")
    print(f"  Mean: {stats['roi_count_mean']:.2f} persons per image")
    print(f"  Std: {stats['roi_count_std']:.2f}")
    print(f"  Min: {stats['roi_count_min']}")
    print(f"  Max: {stats['roi_count_max']}")
    print(f"  Median: {stats['roi_count_median']:.1f}")
    print()
    print("ROI COUNT DISTRIBUTION:")
    roi_dist = stats['roi_count_distribution']
    for count in sorted(roi_dist.keys(), key=int)[:10]:  # Show first 10
        freq = roi_dist[count]
        pct = freq / stats['num_images'] * 100
        print(f"  {count} person(s): {freq:5d} images ({pct:5.2f}%)")
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()