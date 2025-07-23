"""Extended visualizer for hierarchical UNet models."""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2

from .visualization_adapter import AdvancedValidationVisualizer


class HierarchicalUNetVisualizer(AdvancedValidationVisualizer):
    """Extended visualizer that adds UNet foreground/background visualization."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with variant detection."""
        super().__init__(*args, **kwargs)
        self._detect_model_variant()
    
    def _detect_model_variant(self):
        """Detect which UNet variant is being used based on model architecture."""
        self.unet_variant = 'v1'  # Default
        
        # Try a dummy forward pass to check aux_outputs structure
        try:
            import torch
            dummy_features = torch.randn(1, 256, 28, 28).to(self.device)
            
            if hasattr(self.model, 'hierarchical_head'):
                with torch.no_grad():
                    _, aux_outputs = self.model.hierarchical_head(dummy_features)
                
                # Check aux_outputs to determine variant
                if 'attended_features' in aux_outputs:
                    self.unet_variant = 'v4'
                elif 'target_logits_low' in aux_outputs:
                    self.unet_variant = 'v3'
                elif 'target_nontarget_logits' in aux_outputs:
                    self.unet_variant = 'v2'
                    
            print(f"Detected UNet variant: {self.unet_variant}")
        except:
            print("Could not detect UNet variant, defaulting to v1")

    def visualize_validation_images(self, epoch: int, validate_all: bool = False):
        """Override to add UNet visualization as third row."""
        print(f"HierarchicalUNetVisualizer.visualize_validation_images called (variant: {self.unet_variant})")

        # First call parent method to generate standard visualization
        super().visualize_validation_images(epoch, validate_all)

        print("Generating custom 3-row visualization...")

        # Now generate our custom 3-row visualization
        # Collect all images
        all_gt_images = []
        all_pred_images = []
        all_unet_images = []

        # Use the same logic as parent class for image selection
        if validate_all:
            # Get all images from the dataset
            img_ids = self.coco.getImgIds()
            validation_images = []

            for img_id in img_ids:
                img_info = self.coco.loadImgs(img_id)[0]
                filename = img_info['file_name']

                # Count number of persons
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                num_persons = len(ann_ids)

                validation_images.append({
                    'filename': filename,
                    'num_persons': num_persons
                })

            print(f"Validating all {len(validation_images)} images from annotation file")

            # Sort by filename for consistent ordering
            validation_images.sort(key=lambda x: x['filename'])
        else:
            # Use default validation images
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

        print(f"Processing {len(validation_images)} test images for 3-row visualization")

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
            img_path = Path(self.image_dir) / filename
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

            # Store bbox and instance info for later drawing
            instance_info = []

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

                # Store instance info for later
                instance_info.append((bbox, i + 1, color))

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
            unet_image = image_np.copy()

            # Initialize combined masks for UNet visualization
            combined_fg_mask = np.zeros((640, 640), dtype=bool)
            combined_bg_mask = np.zeros((640, 640), dtype=bool)

            with torch.no_grad():
                # Prepare input
                img_tensor = torch.from_numpy(image_np).float().permute(2, 0, 1) / 255.0
                img_tensor = img_tensor.unsqueeze(0).to(self.device)

                # Extract features
                if self.feature_extractor is not None:
                    features = self.feature_extractor.extract_features(img_tensor)
                else:
                    features = img_tensor

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
                    padding = self.roi_padding
                    w, h = x2 - x1, y2 - y1
                    x1 = max(0, x1 - w * padding)
                    y1 = max(0, y1 - h * padding)
                    x2 = min(640, x2 + w * padding)
                    y2 = min(640, y2 + h * padding)

                    # Normalize ROI
                    roi_norm = torch.tensor([[x1/640, y1/640, x2/640, y2/640]], dtype=torch.float32)
                    from src.human_edge_detection.model import ROIBatchProcessor
                    rois = ROIBatchProcessor.prepare_rois_for_batch(roi_norm).to(self.device)

                    # Get prediction
                    if self.is_multiscale:
                        output = self.model(features, rois)
                    else:
                        output = self.model(features=features, rois=rois)

                    # Extract predictions and aux outputs
                    if isinstance(output, tuple):
                        mask_logits, aux_outputs = output
                    elif isinstance(output, dict) and 'masks' in output:
                        mask_logits = output['masks']
                        aux_outputs = {}
                    else:
                        mask_logits = output
                        aux_outputs = {}

                    # Get predicted class
                    mask_pred = torch.argmax(mask_logits[0], dim=0).cpu().numpy()

                    # Create full image mask
                    full_mask = np.zeros((640, 640), dtype=np.uint8)

                    # Calculate exact slice dimensions to avoid mismatch
                    y1_int, y2_int = int(y1), int(y2)
                    x1_int, x2_int = int(x1), int(x2)
                    slice_h = y2_int - y1_int
                    slice_w = x2_int - x1_int

                    # Resize mask to exact slice dimensions with better interpolation
                    # First resize with linear interpolation for smoother results
                    mask_prob = mask_logits[0].softmax(dim=0)[1].cpu().numpy()  # Get target class probability
                    mask_resized = cv2.resize(mask_prob, (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)

                    # Apply threshold to get binary mask
                    mask_resized = (mask_resized > 0.5).astype(np.uint8)

                    # Place resized mask in full image
                    full_mask[y1_int:y2_int, x1_int:x2_int] = mask_resized

                    # Only show target class (class 1)
                    target_mask = (full_mask == 1)
                    pred_image = self._apply_mask(pred_image, target_mask, color)

                    # Process UNet output for current ROI - collect masks
                    if aux_outputs and 'bg_fg_logits' in aux_outputs:
                        bg_fg_logits = aux_outputs['bg_fg_logits']
                        bg_fg_probs = torch.softmax(bg_fg_logits, dim=1)
                        fg_prob = bg_fg_probs[0, 1].cpu().numpy()  # Foreground probability

                        # bg_fg_logits is already at mask_size (56x56), same as mask_logits
                        # Resize to ROI size using same dimensions as the mask
                        fg_prob_resized = cv2.resize(fg_prob, (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)

                        # Create masks based on probability threshold
                        fg_mask_roi = fg_prob_resized >= 0.5
                        bg_mask_roi = fg_prob_resized < 0.5

                        # Add to combined masks
                        combined_fg_mask[y1_int:y2_int, x1_int:x2_int] |= fg_mask_roi
                        combined_bg_mask[y1_int:y2_int, x1_int:x2_int] |= bg_mask_roi
                        
                        # For V3 and V4, we could potentially visualize additional features
                        # but for now we keep the same visualization for consistency

                # After processing all ROIs, render combined UNet masks
                # Create overlay
                overlay = unet_image.copy()

                # First render background (gray)
                overlay[combined_bg_mask] = [128, 128, 128]  # Gray for background

                # Then render foreground (red) - this will overwrite any background pixels
                overlay[combined_fg_mask] = [150, 0, 0]  # Slightly darker red for foreground

                # Blend with original
                alpha = 0.7
                unet_image = cv2.addWeighted(overlay, alpha, unet_image, 1 - alpha, 0)

            # Draw all instance numbers on GT image at the end
            draw = ImageDraw.Draw(gt_pil)
            for bbox, instance_num, color in instance_info:
                self._draw_instance_number(draw, bbox, instance_num, color)

            if validate_all:
                # Save individual image for validate_all mode
                self._save_individual_image_3rows(gt_pil, Image.fromarray(pred_image), Image.fromarray(unet_image), epoch, filename)
            else:
                # Add images to lists for combined visualization
                all_gt_images.append(gt_pil)
                all_pred_images.append(Image.fromarray(pred_image))
                all_unet_images.append(Image.fromarray(unet_image))

        # Create combined visualization (only for default mode)
        if not validate_all and all_gt_images:
            self._create_combined_image_3rows(all_gt_images, all_pred_images, all_unet_images, epoch)

    def _create_combined_image_3rows(
        self,
        gt_images: List[Image.Image],
        pred_images: List[Image.Image],
        unet_images: List[Image.Image],
        epoch: int
    ):
        """Create combined image with 3 rows."""
        n_images = len(gt_images)
        if n_images == 0:
            return

        # Calculate dimensions (matching parent class layout)
        scale_factor = 1.0
        img_width = int(640 * scale_factor)
        img_height = int(640 * scale_factor)
        padding = int(20 * scale_factor)

        # Scale all images
        gt_images = [img.resize((img_width, img_height), Image.LANCZOS) for img in gt_images]
        pred_images = [img.resize((img_width, img_height), Image.LANCZOS) for img in pred_images]
        unet_images = [img.resize((img_width, img_height), Image.LANCZOS) for img in unet_images]

        # Total dimensions
        total_width = n_images * img_width + (n_images + 1) * padding
        total_height = 3 * img_height + 4 * padding  # 3 rows

        # Create canvas
        combined = Image.new('RGB', (total_width, total_height), color='white')

        # Place images
        for i in range(n_images):
            x_offset = padding + i * (img_width + padding)

            # GT image (top row)
            combined.paste(gt_images[i], (x_offset, padding))

            # Prediction image (middle row)
            combined.paste(pred_images[i], (x_offset, img_height + 2 * padding))

            # UNet image (bottom row)
            combined.paste(unet_images[i], (x_offset, 2 * img_height + 3 * padding))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            # Match parent class font size
            font_size = int(30 * scale_factor * 1.5)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate label positions
        # Update label based on variant
        unet_label = {
            'v1': "UNet FG/BG",
            'v2': "Enhanced UNet FG/BG", 
            'v3': "Enhanced UNet + Shallow UNet",
            'v4': "Dual Enhanced UNet"
        }.get(self.unet_variant, "UNet FG/BG")
        
        labels = [
            ("Ground Truth", (255, 102, 102)),  # Light red
            ("Predictions", (0, 153, 0)),       # Green
            (unet_label, (102, 102, 255))       # Light blue
        ]

        for idx, (text, color) in enumerate(labels):
            y_pos = idx * (img_height + padding) + 5
            # Draw background rectangle
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.rectangle(
                [10, y_pos, 10 + text_width + 20, y_pos + text_height + 10],
                fill=color
            )

            # Draw text
            draw.text((20, y_pos + 5), text, fill='white', font=font)

        # Resize to 60% before saving (matching parent class)
        resized_width = int(combined.width * 0.6)
        resized_height = int(combined.height * 0.6)
        combined_resized = combined.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        # Save with same filename as 2-row visualization
        # Add 1 to epoch to match parent class behavior (0-indexed to 1-indexed)
        display_epoch = epoch + 1 if epoch >= 0 else epoch
        filename = f"validation_all_images_epoch_{display_epoch:04d}.png"
        output_path = self.output_dir / filename
        combined_resized.save(output_path)
        print(f"Saved 3-row validation visualization: {output_path}")

    def _save_individual_image_3rows(
        self,
        gt_image: Image.Image,
        pred_image: Image.Image,
        unet_image: Image.Image,
        epoch: int,
        filename: str
    ):
        """Save individual validation result with 3 rows for a single image."""
        # Create combined image for this single validation (matching parent class layout)
        scale_factor = 1.0
        img_width = int(640 * scale_factor)
        img_height = int(640 * scale_factor)
        padding = int(20 * scale_factor)

        # Scale images
        gt_scaled = gt_image.resize((img_width, img_height), Image.LANCZOS)
        pred_scaled = pred_image.resize((img_width, img_height), Image.LANCZOS)
        unet_scaled = unet_image.resize((img_width, img_height), Image.LANCZOS)

        # Create canvas for vertical 3-row layout
        total_width = img_width + 2 * padding
        total_height = 3 * img_height + 4 * padding

        combined = Image.new('RGB', (total_width, total_height), color='white')

        # Place images vertically
        combined.paste(gt_scaled, (padding, padding))
        combined.paste(pred_scaled, (padding, img_height + 2 * padding))
        combined.paste(unet_scaled, (padding, 2 * img_height + 3 * padding))

        # Add labels
        draw = ImageDraw.Draw(combined)
        try:
            # Use same font size as in _create_combined_image_3rows
            font_size = int(30 * scale_factor * 1.5)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Calculate label positions
        # Update label based on variant
        unet_label = {
            'v1': "UNet FG/BG",
            'v2': "Enhanced UNet FG/BG", 
            'v3': "Enhanced UNet + Shallow UNet",
            'v4': "Dual Enhanced UNet"
        }.get(self.unet_variant, "UNet FG/BG")
        
        labels = [
            ("Ground Truth", (255, 102, 102)),  # Light red
            ("Predictions", (0, 153, 0)),       # Green
            (unet_label, (102, 102, 255))       # Light blue
        ]

        for idx, (text, color) in enumerate(labels):
            y_pos = idx * (img_height + padding) + 5
            # Draw background rectangle
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.rectangle(
                [10, y_pos, 10 + text_width + 20, y_pos + text_height + 10],
                fill=color
            )

            # Draw text
            draw.text((20, y_pos + 5), text, fill='white', font=font)

        # Resize to 60% before saving (matching parent class)
        resized_width = int(combined.width * 0.6)
        resized_height = int(combined.height * 0.6)
        combined_resized = combined.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
        
        # Create subdirectory for epoch
        epoch_dir = self.output_dir / f'epoch_{epoch:04d}'
        epoch_dir.mkdir(exist_ok=True)

        # Save with original filename (without extension)
        base_name = filename.rsplit('.', 1)[0]
        output_path = epoch_dir / f'{base_name}_val.png'
        combined_resized.save(output_path)

        # Print progress every 10 images
        if hasattr(self, '_save_count'):
            self._save_count += 1
        else:
            self._save_count = 1

        if self._save_count % 10 == 0:
            print(f"Processed {self._save_count} images...")

        # Reset counter at the end
        if self._save_count == len(self.coco.getImgIds()):  # Total images
            print(f"All {self._save_count} validation images saved to {epoch_dir}")
            delattr(self, '_save_count')