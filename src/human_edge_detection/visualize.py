"""Visualization tools for validation results."""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from pycocotools.coco import COCO

from .feature_extractor import YOLOv9FeatureExtractor
from .model import ROISegmentationModel, ROIBatchProcessor


class ValidationVisualizer:
    """Visualizer for validation results."""

    # Default validation images as per requirements
    DEFAULT_VALIDATION_IMAGES = [
        {'filename': '000000020992.jpg', 'num_persons': 1},
        {'filename': '000000413552.jpg', 'num_persons': 2},
        {'filename': '000000109118.jpg', 'num_persons': 3},
        {'filename': '000000162732.jpg', 'num_persons': 5}
    ]

    def __init__(
        self,
        model: ROISegmentationModel,
        feature_extractor: YOLOv9FeatureExtractor,
        coco: COCO,
        image_dir: str,
        output_dir: str = 'validation_results',
        device: str = 'cuda'
    ):
        """Initialize visualizer.

        Args:
            model: Trained segmentation model
            feature_extractor: YOLO feature extractor
            coco: COCO dataset object
            image_dir: Directory containing images
            output_dir: Directory to save visualization results
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.coco = coco
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device

        # Color palette for instances
        self.colors = self._generate_colors(20)

    def _generate_colors(self, n_colors: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for visualization."""
        colors = []
        cmap = cm.get_cmap('tab20')
        for i in range(n_colors):
            color = cmap(i / n_colors)[:3]
            colors.append(tuple(int(c * 255) for c in color))
        return colors

    def _draw_bbox(self, draw: ImageDraw.Draw, bbox: List[float], color: Tuple[int, int, int], width: int = 3):
        """Draw bounding box on image."""
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        draw.rectangle([x, y, x2, y2], outline=color, width=width)

    def _draw_instance_number(self, draw: ImageDraw.Draw, bbox: List[float], instance_id: int, color: Tuple[int, int, int]):
        """Draw instance number near bounding box."""
        x, y, w, h = bbox
        text = f"#{instance_id}"

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Get text size
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw background rectangle for text
        padding = 5
        draw.rectangle([x, y - text_height - 2 * padding, x + text_width + 2 * padding, y], fill=color)

        # Draw text
        draw.text((x + padding, y - text_height - padding), text, fill='white', font=font)

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float = 0.5) -> np.ndarray:
        """Apply colored mask overlay to image."""
        overlay = image.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    def _find_fallback_images(self, target_counts: Dict[int, str]) -> Dict[str, Dict]:
        """Find fallback images when default images are not found.
        
        Args:
            target_counts: Dict mapping person count to missing filename
            
        Returns:
            Dict mapping original filename to replacement info
        """
        fallback_map = {}
        
        # Get all images in the dataset
        img_ids = self.coco.getImgIds()
        
        # Group images by person count
        images_by_count = {count: [] for count in target_counts.keys()}
        
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            num_persons = len(ann_ids)
            
            if num_persons in images_by_count:
                img_info = self.coco.loadImgs(img_id)[0]
                images_by_count[num_persons].append({
                    'image_id': img_id,
                    'filename': img_info['file_name']
                })
        
        # Select fallback for each missing image
        for count, original_filename in target_counts.items():
            if images_by_count[count]:
                # Use the first available image with the same person count
                fallback = images_by_count[count][0]
                fallback_map[original_filename] = {
                    'filename': fallback['filename'],
                    'num_persons': count,
                    'is_fallback': True
                }
                print(f"Using fallback image {fallback['filename']} instead of {original_filename} ({count} person(s))")
            else:
                print(f"Warning: No images found with {count} person(s) for fallback")
        
        return fallback_map
    
    def visualize_validation_images(self, epoch: int):
        """Generate validation visualization for specific images.

        Creates a two-row image:
        - Top row: Ground truth with ROI boxes (red), instance numbers, and masks
        - Bottom row: Model predictions (masks only)
        """
        self.model.eval()

        # Collect all visualizations
        all_gt_images = []
        all_pred_images = []
        
        # Check which default images exist
        missing_images = {}
        validation_images = []
        
        for val_img_info in self.DEFAULT_VALIDATION_IMAGES:
            filename = val_img_info['filename']
            img_ids = self.coco.getImgIds()
            img_id = None
            
            # Try to find the image
            for iid in img_ids:
                img_info = self.coco.loadImgs(iid)[0]
                if img_info['file_name'] == filename:
                    img_id = iid
                    break
            
            if img_id is None:
                # Image not found, mark for fallback
                missing_images[val_img_info['num_persons']] = filename
            else:
                # Image found, use it
                validation_images.append(val_img_info)
        
        # Find fallback images if needed
        if missing_images:
            fallback_map = self._find_fallback_images(missing_images)
            # Add fallback images to validation list
            for fallback_info in fallback_map.values():
                validation_images.append(fallback_info)
        
        # Sort by number of persons for consistent ordering
        validation_images.sort(key=lambda x: x['num_persons'])

        for val_img_info in validation_images:
            filename = val_img_info['filename']

            # Find image ID
            img_ids = self.coco.getImgIds()
            img_id = None
            for iid in img_ids:
                img_info = self.coco.loadImgs(iid)[0]
                if img_info['file_name'] == filename:
                    img_id = iid
                    break

            if img_id is None:
                print(f"Warning: Image {filename} not found in dataset")
                continue

            # Load image
            img_path = self.image_dir / filename
            if not img_path.exists():
                print(f"Warning: Image file {img_path} not found")
                continue

            image = Image.open(img_path).convert('RGB')
            orig_width, orig_height = image.size

            # Resize to model input size
            image_resized = image.resize((640, 640), Image.BILINEAR)
            image_np = np.array(image_resized)

            # Get annotations
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ann_ids)

            # Create GT visualization
            gt_image = image_np.copy()
            gt_pil = Image.fromarray(gt_image)
            draw = ImageDraw.Draw(gt_pil)

            # Process each annotation
            pred_masks_combined = np.zeros((640, 640), dtype=np.uint8)

            for i, ann in enumerate(anns):
                color = self.colors[i % len(self.colors)]

                # Scale bbox to resized image
                bbox = ann['bbox'].copy()
                bbox[0] = bbox[0] * 640 / orig_width
                bbox[1] = bbox[1] * 640 / orig_height
                bbox[2] = bbox[2] * 640 / orig_width
                bbox[3] = bbox[3] * 640 / orig_height

                # Draw bbox on GT (red color as specified)
                self._draw_bbox(draw, bbox, color=(255, 0, 0), width=3)
                self._draw_instance_number(draw, bbox, i + 1, color)

                # Get and draw GT mask
                if isinstance(ann['segmentation'], list):
                    mask = self.coco.annToMask(ann)
                else:
                    from pycocotools import mask as maskUtils
                    mask = maskUtils.decode(ann['segmentation'])

                # Resize mask
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)

                # Apply mask to GT image
                gt_image_np = np.array(gt_pil)
                gt_image_np = self._apply_mask(gt_image_np, mask_resized, color)
                gt_pil = Image.fromarray(gt_image_np)
                draw = ImageDraw.Draw(gt_pil)

            # Get model predictions
            pred_image = image_np.copy()

            with torch.no_grad():
                # Prepare input
                img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Extract features
                features = self.feature_extractor.extract_features(img_tensor)

                # Process each annotation as ROI
                for i, ann in enumerate(anns):
                    color = self.colors[i % len(self.colors)]

                    # Prepare ROI
                    bbox = ann['bbox'].copy()
                    x1 = bbox[0] * 640 / orig_width
                    y1 = bbox[1] * 640 / orig_height
                    x2 = x1 + bbox[2] * 640 / orig_width
                    y2 = y1 + bbox[3] * 640 / orig_height

                    # Add padding
                    padding = 0.1
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, x1 - w * padding)
                    y1 = max(0, y1 - h * padding)
                    x2 = min(640, x2 + w * padding)
                    y2 = min(640, y2 + h * padding)

                    # Normalize ROI
                    roi_norm = torch.tensor([[x1/640, y1/640, x2/640, y2/640]], dtype=torch.float32)
                    rois = ROIBatchProcessor.prepare_rois_for_batch(roi_norm).to(self.device)

                    # Get prediction
                    output = self.model(features=features, rois=rois)
                    mask_logits = output['masks']

                    # Get predicted class
                    mask_pred = torch.argmax(mask_logits[0], dim=0).cpu().numpy()

                    # Resize mask back to ROI size
                    roi_h = int(y2 - y1)
                    roi_w = int(x2 - x1)
                    mask_resized = cv2.resize(mask_pred.astype(np.uint8), (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

                    # Create full image mask
                    full_mask = np.zeros((640, 640), dtype=np.uint8)
                    full_mask[int(y1):int(y2), int(x1):int(x2)] = mask_resized

                    # Only show target class (class 1)
                    target_mask = (full_mask == 1)
                    pred_image = self._apply_mask(pred_image, target_mask, color)

            # Add images to lists
            all_gt_images.append(gt_pil)
            all_pred_images.append(Image.fromarray(pred_image))

        # Create combined visualization
        if all_gt_images:
            self._create_combined_image(all_gt_images, all_pred_images, epoch)

    def _create_combined_image(self, gt_images: List[Image.Image], pred_images: List[Image.Image], epoch: int):
        """Create combined visualization with GT on top and predictions on bottom."""
        n_images = len(gt_images)
        if n_images == 0:
            return

        # Calculate dimensions
        img_width = 640
        img_height = 640
        padding = 20

        # Total dimensions
        total_width = n_images * img_width + (n_images + 1) * padding
        total_height = 2 * img_height + 3 * padding

        # Create blank canvas
        combined = Image.new('RGB', (total_width, total_height), color='white')

        # Place images
        for i in range(n_images):
            x_offset = padding + i * (img_width + padding)

            # GT image (top row)
            combined.paste(gt_images[i], (x_offset, padding))

            # Prediction image (bottom row)
            combined.paste(pred_images[i], (x_offset, img_height + 2 * padding))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        except:
            font = ImageFont.load_default()

        # Label for GT row
        draw.text((10, 5), "Ground Truth", fill='black', font=font)

        # Label for prediction row
        draw.text((10, img_height + padding + 5), "Predictions", fill='black', font=font)

        # Save image
        filename = f"validation_all_images_epoch_{epoch:04d}.png"
        output_path = self.output_dir / filename
        combined.save(output_path)
        print(f"Saved validation visualization: {output_path}")


def visualize_epoch_results(
    model_path: str,
    epoch: int,
    annotation_file: str,
    image_dir: str,
    onnx_model_path: str,
    output_dir: str = 'validation_results',
    device: str = 'cuda'
):
    """Visualize results for a specific epoch checkpoint."""
    from .model import create_model

    # Load model
    model = create_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create feature extractor
    feature_extractor = YOLOv9FeatureExtractor(onnx_model_path, device=device)

    # Load COCO dataset
    coco = COCO(annotation_file)

    # Create visualizer
    visualizer = ValidationVisualizer(
        model=model,
        feature_extractor=feature_extractor,
        coco=coco,
        image_dir=image_dir,
        output_dir=output_dir,
        device=device
    )

    # Generate visualization
    visualizer.visualize_validation_images(epoch)


if __name__ == "__main__":
    # Test visualization with a dummy model
    from .model import create_model

    print("Creating dummy model for visualization test...")
    model = create_model()

    # This would normally be called during training
    # visualize_epoch_results(
    #     model_path='checkpoints/checkpoint_epoch_0001_640x640_0500.pth',
    #     epoch=1,
    #     annotation_file='data/annotations/instances_val2017_person_only_no_crowd_100.json',
    #     image_dir='data/images/val2017',
    #     onnx_model_path='ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'
    # )