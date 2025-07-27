"""Filtered dataset wrapper that removes problematic bounding boxes."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pycocotools.coco import COCO

from .dataset import COCOInstanceSegmentationDataset


class FilteredCOCODataset(COCOInstanceSegmentationDataset):
    """COCO dataset that filters out problematic bounding boxes.
    
    This wrapper extends the base dataset to filter out annotations with:
    - Extremely narrow or tall aspect ratios
    - Very small absolute dimensions
    """
    
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        image_size: Tuple[int, int] = (640, 640),
        mask_size: Tuple[int, int] = (56, 56),
        augment: bool = False,
        roi_padding: float = 0.2,
        max_instances: int = 1,
        min_bbox_width: int = 30,
        min_bbox_height: int = 30,
        min_aspect_ratio: float = 0.2,  # width/height
        max_aspect_ratio: float = 5.0,  # width/height
        verbose: bool = True
    ):
        """Initialize filtered dataset.
        
        Args:
            annotation_file: Path to COCO format annotations
            image_dir: Directory containing images
            image_size: Target image size (height, width)
            mask_size: Target mask size (height, width)
            augment: Whether to apply data augmentation
            roi_padding: Padding ratio for ROI extraction
            max_instances: Maximum instances per image
            min_bbox_width: Minimum bbox width in pixels (in original image)
            min_bbox_height: Minimum bbox height in pixels (in original image)
            min_aspect_ratio: Minimum aspect ratio (width/height)
            max_aspect_ratio: Maximum aspect ratio (width/height)
            verbose: Whether to print filtering statistics
        """
        self.min_bbox_width = min_bbox_width
        self.min_bbox_height = min_bbox_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.verbose = verbose
        
        # Initialize parent class
        super().__init__(
            annotation_file=annotation_file,
            image_dir=image_dir,
            image_size=image_size,
            mask_size=mask_size,
            transform=None,  # augmentation handled separately
            roi_padding=roi_padding,
            max_instances_per_image=max_instances
        )
        
    def _create_samples(self) -> List[Dict]:
        """Create dataset samples with filtering."""
        samples = []
        
        # Statistics for filtering
        total_annotations = 0
        filtered_width = 0
        filtered_height = 0
        filtered_aspect = 0
        
        # Process each image
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=[self.person_cat_id])
            anns = self.coco.loadAnns(ann_ids)
            
            if len(anns) == 0:
                continue
            
            # Get image info for original dimensions
            img_info = self.coco.loadImgs([img_id])[0]
            
            # Filter annotations
            valid_anns = []
            for ann in anns:
                total_annotations += 1
                
                x, y, w, h = ann['bbox']
                aspect_ratio = w / h if h > 0 else 0
                
                # Apply filters
                if w < self.min_bbox_width:
                    filtered_width += 1
                    continue
                    
                if h < self.min_bbox_height:
                    filtered_height += 1
                    continue
                    
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    filtered_aspect += 1
                    continue
                
                valid_anns.append(ann)
            
            # Only create samples if we have valid annotations
            if len(valid_anns) > 0:
                # Create a sample for each valid annotation as target
                for target_ann in valid_anns:
                    sample = {
                        'image_id': img_id,
                        'target_ann_id': target_ann['id'],
                        'all_ann_ids': [ann['id'] for ann in valid_anns]
                    }
                    samples.append(sample)
        
        # Print statistics if verbose
        if self.verbose and total_annotations > 0:
            print(f"\n=== Dataset Filtering Statistics ===")
            print(f"Total annotations: {total_annotations}")
            print(f"Filtered by width < {self.min_bbox_width}: {filtered_width} ({filtered_width/total_annotations*100:.1f}%)")
            print(f"Filtered by height < {self.min_bbox_height}: {filtered_height} ({filtered_height/total_annotations*100:.1f}%)")
            print(f"Filtered by aspect ratio: {filtered_aspect} ({filtered_aspect/total_annotations*100:.1f}%)")
            print(f"Total filtered: {filtered_width + filtered_height + filtered_aspect} ({(filtered_width + filtered_height + filtered_aspect)/total_annotations*100:.1f}%)")
            print(f"Valid samples: {len(samples)}")
            print(f"===================================\n")
        
        return samples


def create_filtered_dataset(
    annotation_file: str,
    image_dir: str,
    image_size: Tuple[int, int] = (640, 640),
    mask_size: Tuple[int, int] = (64, 48),
    augment: bool = False,
    roi_padding: float = 0.2,
    **filter_kwargs
) -> FilteredCOCODataset:
    """Create a filtered dataset with reasonable defaults.
    
    Args:
        annotation_file: Path to COCO annotations
        image_dir: Directory containing images
        image_size: Target image size
        mask_size: Target mask size
        augment: Whether to apply augmentation
        roi_padding: ROI padding ratio
        **filter_kwargs: Additional filtering parameters
            - min_bbox_width (default: 40)
            - min_bbox_height (default: 40)  
            - min_aspect_ratio (default: 0.25)
            - max_aspect_ratio (default: 4.0)
    """
    # Set defaults
    filter_params = {
        'min_bbox_width': 40,
        'min_bbox_height': 40,
        'min_aspect_ratio': 0.25,  # 1:4 ratio
        'max_aspect_ratio': 4.0,   # 4:1 ratio
        'verbose': True
    }
    filter_params.update(filter_kwargs)
    
    return FilteredCOCODataset(
        annotation_file=annotation_file,
        image_dir=image_dir,
        image_size=image_size,
        mask_size=mask_size,
        augment=augment,
        roi_padding=roi_padding,
        **filter_params
    )