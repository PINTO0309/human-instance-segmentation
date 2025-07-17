"""Dataset and data loader for ROI-based instance segmentation."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random


class COCOInstanceSegmentationDataset(Dataset):
    """Dataset for ROI-based instance segmentation with 3-class masks.
    
    Classes:
    - 0: Background
    - 1: Target mask (primary instance in ROI)
    - 2: Non-target mask (other instances in ROI)
    """
    
    def __init__(
        self,
        annotation_file: str,
        image_dir: str,
        transform=None,
        roi_padding: float = 0.1,
        mask_size: Tuple[int, int] = (56, 56),
        feature_size: Tuple[int, int] = (80, 80),
        image_size: Tuple[int, int] = (640, 640),
        min_roi_size: int = 16,
        max_instances_per_image: int = 10
    ):
        """Initialize dataset.
        
        Args:
            annotation_file: Path to COCO format annotation file
            image_dir: Directory containing images
            transform: Optional transform to apply to images
            roi_padding: Padding to add around ROI (fraction of ROI size)
            mask_size: Output size for ROI masks
            feature_size: Size of feature maps from YOLO
            image_size: Input image size
            min_roi_size: Minimum ROI size in pixels
            max_instances_per_image: Maximum instances to process per image
        """
        self.coco = COCO(annotation_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.roi_padding = roi_padding
        self.mask_size = mask_size
        self.feature_size = feature_size
        self.image_size = image_size
        self.min_roi_size = min_roi_size
        self.max_instances_per_image = max_instances_per_image
        
        # Get all valid samples (image_id, annotation_id pairs)
        self.samples = []
        img_ids = self.coco.getImgIds()
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)
            
            # Filter valid annotations
            valid_anns = []
            for ann in anns:
                bbox = ann['bbox']
                if bbox[2] >= min_roi_size and bbox[3] >= min_roi_size:
                    valid_anns.append(ann)
            
            # Create samples for each valid annotation as target
            for i, target_ann in enumerate(valid_anns[:max_instances_per_image]):
                self.samples.append({
                    'image_id': img_id,
                    'target_ann_id': target_ann['id'],
                    'all_ann_ids': [a['id'] for a in valid_anns]
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_info = self.coco.loadImgs(sample['image_id'])[0]
        img_path = self.image_dir / img_info['file_name']
        image = Image.open(img_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # Resize image to target size
        image = image.resize(self.image_size, Image.BILINEAR)
        image_np = np.array(image)
        
        # Load all annotations for this image
        all_anns = self.coco.loadAnns(sample['all_ann_ids'])
        target_ann = self.coco.loadAnns([sample['target_ann_id']])[0]
        
        # Create instance masks
        instance_masks = []
        bboxes = []
        
        for ann in all_anns:
            # Convert segmentation to binary mask
            if isinstance(ann['segmentation'], list):
                mask = self.coco.annToMask(ann)
            else:
                mask = maskUtils.decode(ann['segmentation'])
            
            # Resize mask to match image size
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            instance_masks.append(mask)
            
            # Scale bbox to new image size
            x, y, w, h = ann['bbox']
            x = x * self.image_size[0] / orig_width
            y = y * self.image_size[1] / orig_height
            w = w * self.image_size[0] / orig_width
            h = h * self.image_size[1] / orig_height
            bboxes.append([x, y, w, h])
        
        # Get target ROI with padding
        target_idx = sample['all_ann_ids'].index(sample['target_ann_id'])
        x, y, w, h = bboxes[target_idx]
        
        # Add padding
        pad_x = w * self.roi_padding
        pad_y = h * self.roi_padding
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(self.image_size[0], int(x + w + pad_x))
        y2 = min(self.image_size[1], int(y + h + pad_y))
        
        # Ensure minimum size
        if x2 - x1 < self.min_roi_size:
            center_x = (x1 + x2) // 2
            x1 = max(0, center_x - self.min_roi_size // 2)
            x2 = min(self.image_size[0], x1 + self.min_roi_size)
        
        if y2 - y1 < self.min_roi_size:
            center_y = (y1 + y2) // 2
            y1 = max(0, center_y - self.min_roi_size // 2)
            y2 = min(self.image_size[1], y1 + self.min_roi_size)
        
        # Create 3-class mask for ROI
        roi_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        
        # Target mask (class 1)
        target_mask_roi = instance_masks[target_idx][y1:y2, x1:x2]
        roi_mask[target_mask_roi > 0] = 1
        
        # Non-target masks (class 2)
        for i, mask in enumerate(instance_masks):
            if i != target_idx:
                other_mask_roi = mask[y1:y2, x1:x2]
                # Only mark as non-target if not already marked as target
                roi_mask[(other_mask_roi > 0) & (roi_mask == 0)] = 2
        
        # Resize ROI mask to fixed size
        roi_mask_resized = cv2.resize(roi_mask, self.mask_size, interpolation=cv2.INTER_NEAREST)
        
        # Normalize ROI coordinates
        roi_norm = np.array([
            x1 / self.image_size[0],
            y1 / self.image_size[1],
            x2 / self.image_size[0],
            y2 / self.image_size[1]
        ], dtype=np.float32)
        
        # Apply transforms to image if provided
        if self.transform:
            image_np = self.transform(image_np)
        else:
            # Default normalization
            image_np = image_np.astype(np.float32) / 255.0
            image_np = torch.from_numpy(image_np).permute(2, 0, 1)
        
        # Convert to tensors
        roi_mask_tensor = torch.from_numpy(roi_mask_resized).long()
        roi_norm_tensor = torch.from_numpy(roi_norm)
        
        # Calculate ROI in feature map coordinates
        stride = self.image_size[0] / self.feature_size[0]
        roi_feature = roi_norm * self.feature_size[0]  # Convert to feature map size
        
        return {
            'image': image_np,
            'roi_mask': roi_mask_tensor,
            'roi_coords': roi_norm_tensor,
            'roi_coords_feature': roi_feature,
            'image_id': sample['image_id'],
            'target_ann_id': sample['target_ann_id']
        }


def create_data_loaders(
    train_annotation_file: str,
    val_annotation_file: str,
    train_image_dir: str,
    val_image_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.
    
    Args:
        train_annotation_file: Path to training annotations
        val_annotation_file: Path to validation annotations
        train_image_dir: Path to training images
        val_image_dir: Path to validation images
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = COCOInstanceSegmentationDataset(
        train_annotation_file,
        train_image_dir,
        **dataset_kwargs
    )
    
    val_dataset = COCOInstanceSegmentationDataset(
        val_annotation_file,
        val_image_dir,
        **dataset_kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_fn_with_rois(batch):
    """Custom collate function to handle variable number of ROIs per image."""
    # For now, we process one ROI per image
    # Can be extended to handle multiple ROIs
    return torch.utils.data.default_collate(batch)


if __name__ == "__main__":
    # Test dataset
    dataset = COCOInstanceSegmentationDataset(
        annotation_file="data/annotations/instances_train2017_person_only_no_crowd_100.json",
        image_dir="data/images/train2017"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"ROI mask shape: {sample['roi_mask'].shape}")
    print(f"ROI coords: {sample['roi_coords']}")
    print(f"Unique classes in mask: {torch.unique(sample['roi_mask'])}")