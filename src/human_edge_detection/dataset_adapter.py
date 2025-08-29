"""Adapter to convert dataset output format for training."""

import torch
from typing import Dict, List


def convert_batch_format(batch: List[Dict]) -> Dict:
    """Convert dataset batch format to training format.
    
    Converts from:
        List of {'image', 'roi_mask', 'roi_coords', 'roi_coords_feature', ...}
    
    To:
        {'image', 'roi_boxes', 'roi_masks', ...}
    """
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack masks
    masks = torch.stack([item['roi_mask'] for item in batch])
    
    # Convert roi_coords to roi_boxes format
    # roi_coords is normalized [x1, y1, x2, y2]
    # roi_boxes needs [batch_idx, x1, y1, x2, y2] in normalized coordinates [0.0-1.0]
    roi_boxes = []
    for i, item in enumerate(batch):
        roi_norm = item['roi_coords']
        # Keep coordinates normalized as [0.0-1.0]
        x1 = roi_norm[0]
        y1 = roi_norm[1]
        x2 = roi_norm[2]
        y2 = roi_norm[3]
        roi_boxes.append([i, x1, y1, x2, y2])
    
    roi_boxes = torch.tensor(roi_boxes, dtype=torch.float32)
    
    # Prepare instance info if distance loss is used
    instance_info = []
    for item in batch:
        info = {
            'image_id': item.get('image_id', 0),
            'instance_masks': [],  # Could be populated if needed
            'instance_distances': torch.zeros_like(item['roi_mask'], dtype=torch.float32)
        }
        instance_info.append(info)
    
    return {
        'image': images,
        'roi_boxes': roi_boxes,
        'roi_masks': masks,
        'instance_info': instance_info
    }


def create_collate_fn():
    """Create collate function that converts batch format."""
    def collate_fn(batch):
        # First use default collate
        default_batch = torch.utils.data.default_collate(batch)
        # Then convert format
        return convert_batch_format(batch)
    
    return collate_fn