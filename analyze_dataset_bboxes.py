"""Analyze bounding boxes in the dataset to find issues."""

import numpy as np
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset

def analyze_bbox_distribution():
    """Analyze the distribution of bbox sizes in the dataset."""
    print("=== Analyzing Bbox Distribution ===")
    
    # Load COCO annotations
    coco = COCO('data/annotations/instances_val2017_person_only_no_crowd_100.json')
    
    # Get all person annotations
    person_cat_id = 1
    img_ids = coco.getImgIds(catIds=[person_cat_id])
    
    widths = []
    heights = []
    areas = []
    aspect_ratios = []
    
    for img_id in img_ids:
        img_info = coco.loadImgs([img_id])[0]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=[person_cat_id])
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            x, y, w, h = ann['bbox']
            widths.append(w)
            heights.append(h)
            areas.append(w * h)
            aspect_ratios.append(w / h if h > 0 else 0)
    
    widths = np.array(widths)
    heights = np.array(heights)
    areas = np.array(areas)
    aspect_ratios = np.array(aspect_ratios)
    
    print(f"\nTotal annotations: {len(widths)}")
    print(f"\nWidth statistics:")
    print(f"  Min: {widths.min():.1f}")
    print(f"  Max: {widths.max():.1f}")
    print(f"  Mean: {widths.mean():.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    print(f"  Percentiles (10, 25, 75, 90): {np.percentile(widths, [10, 25, 75, 90])}")
    
    print(f"\nHeight statistics:")
    print(f"  Min: {heights.min():.1f}")
    print(f"  Max: {heights.max():.1f}")
    print(f"  Mean: {heights.mean():.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    
    print(f"\nAspect ratio (w/h) statistics:")
    print(f"  Min: {aspect_ratios.min():.3f}")
    print(f"  Max: {aspect_ratios.max():.3f}")
    print(f"  Mean: {aspect_ratios.mean():.3f}")
    print(f"  Median: {np.median(aspect_ratios):.3f}")
    
    # Find extremely narrow bboxes
    narrow_threshold = 30
    narrow_mask = widths < narrow_threshold
    print(f"\nBboxes with width < {narrow_threshold}: {narrow_mask.sum()} ({narrow_mask.mean()*100:.1f}%)")
    
    # Visualize distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(widths, bins=50)
    axes[0, 0].axvline(narrow_threshold, color='r', linestyle='--', label=f'Threshold={narrow_threshold}')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Bbox Width Distribution')
    axes[0, 0].legend()
    
    axes[0, 1].hist(heights, bins=50)
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Bbox Height Distribution')
    
    axes[1, 0].hist(aspect_ratios, bins=50)
    axes[1, 0].axvline(0.75, color='r', linestyle='--', label='w/h=0.75 (3:4)')
    axes[1, 0].set_xlabel('Aspect Ratio (w/h)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Aspect Ratio Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].scatter(widths, heights, alpha=0.5)
    axes[1, 1].axvline(narrow_threshold, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Width (pixels)')
    axes[1, 1].set_ylabel('Height (pixels)')
    axes[1, 1].set_title('Width vs Height')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bbox_distribution.png')
    print("\nVisualization saved as 'bbox_distribution.png'")

def check_dataset_selection():
    """Check how the dataset selects ROIs."""
    print("\n\n=== Checking Dataset ROI Selection ===")
    
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        roi_padding=0.2
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset samples: {len(dataset.samples)}")
    
    # Check first few samples
    print("\nFirst 10 samples:")
    for i in range(min(10, len(dataset))):
        sample_info = dataset.samples[i]
        img_id = sample_info['image_id']
        target_ann_id = sample_info['target_ann_id']
        all_ann_ids = sample_info['all_ann_ids']
        
        # Get image info
        img_info = dataset.coco.loadImgs([img_id])[0]
        
        # Get target annotation
        target_ann = dataset.coco.loadAnns([target_ann_id])[0]
        x, y, w, h = target_ann['bbox']
        
        print(f"\nSample {i}:")
        print(f"  Image: {img_info['file_name']} ({img_info['width']}x{img_info['height']})")
        print(f"  Target bbox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]")
        print(f"  Aspect ratio: {w/h:.3f}")
        print(f"  Number of annotations: {len(all_ann_ids)}")
        
        if len(all_ann_ids) > 1:
            print(f"  Other annotations:")
            for ann_id in all_ann_ids:
                if ann_id != target_ann_id:
                    ann = dataset.coco.loadAnns([ann_id])[0]
                    x2, y2, w2, h2 = ann['bbox']
                    print(f"    - [{x2:.1f}, {y2:.1f}, {w2:.1f}, {h2:.1f}] (w/h={w2/h2:.3f})")

def find_problematic_samples():
    """Find samples with extremely narrow bboxes."""
    print("\n\n=== Finding Problematic Samples ===")
    
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_val2017_person_only_no_crowd_100.json",
        image_dir="data/images/val2017",
        image_size=(640, 640),
        mask_size=(64, 48),
        roi_padding=0.2
    )
    
    narrow_samples = []
    
    for i, sample_info in enumerate(dataset.samples):
        target_ann = dataset.coco.loadAnns([sample_info['target_ann_id']])[0]
        x, y, w, h = target_ann['bbox']
        
        if w < 30:  # Narrow bbox
            img_info = dataset.coco.loadImgs([sample_info['image_id']])[0]
            narrow_samples.append({
                'index': i,
                'image': img_info['file_name'],
                'bbox': [x, y, w, h],
                'aspect_ratio': w/h
            })
    
    print(f"Found {len(narrow_samples)} samples with width < 30 pixels")
    
    # Show first few
    print("\nFirst 5 narrow samples:")
    for sample in narrow_samples[:5]:
        print(f"\nIndex {sample['index']}:")
        print(f"  Image: {sample['image']}")
        print(f"  Bbox: {sample['bbox']}")
        print(f"  Aspect ratio: {sample['aspect_ratio']:.3f}")
    
    return narrow_samples

def main():
    print("=== Analyzing Dataset Bboxes ===\n")
    
    analyze_bbox_distribution()
    check_dataset_selection()
    problematic = find_problematic_samples()
    
    print(f"\n\n=== Summary ===")
    print(f"Found {len(problematic)} problematic samples with narrow bboxes.")
    print("These samples will cause issues with the pretrained UNet because:")
    print("1. Extreme aspect ratio distortion when resizing to 64x48")
    print("2. Loss of detail in the narrow dimension")
    print("3. Pre-trained model expects more typical human proportions")

if __name__ == "__main__":
    main()