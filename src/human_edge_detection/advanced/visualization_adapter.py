"""Adapter for visualization with advanced models."""

import torch
import torch.nn as nn
from typing import Dict, Optional
import numpy as np

from ..visualize import ValidationVisualizer as BaseValidationVisualizer


class AdvancedValidationVisualizer(BaseValidationVisualizer):
    """Extended visualizer that handles both basic and multiscale models."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[object],
        coco,
        image_dir: str,
        output_dir: str = 'validation_results',
        device: str = 'cuda',
        is_multiscale: bool = False
    ):
        """Initialize advanced visualizer.

        Args:
            model: The segmentation model (basic or multiscale)
            feature_extractor: Feature extractor (None for multiscale models)
            coco: COCO dataset object
            image_dir: Directory containing validation images
            output_dir: Directory to save visualizations
            device: Device to run inference on
            is_multiscale: Whether the model is multiscale
        """
        # For multiscale models, we need a dummy feature extractor
        if is_multiscale and feature_extractor is None:
            # The multiscale model has integrated feature extraction
            # Create a dummy that just returns the input
            class DummyExtractor:
                def extract_features(self, x):
                    return x
            feature_extractor = DummyExtractor()

        super().__init__(
            model=model,
            feature_extractor=feature_extractor,
            coco=coco,
            image_dir=image_dir,
            output_dir=output_dir,
            device=device
        )

        self.is_multiscale = is_multiscale

    def visualize_validation_images(self, epoch: int, validate_all: bool = False):
        """Override to handle multiscale models properly."""
        if self.is_multiscale:
            # For multiscale models, we need to handle the different interface
            self._visualize_multiscale_images(epoch, validate_all)
        else:
            # For basic models, use the parent implementation
            super().visualize_validation_images(epoch, validate_all)

    def _visualize_multiscale_images(self, epoch: int, validate_all: bool = False):
        """Visualization specifically for multiscale models."""
        import torch
        import numpy as np
        from PIL import Image, ImageDraw
        from torchvision.transforms import functional as F
        from ..model import ROIBatchProcessor

        self.model.eval()

        all_gt_images = []
        all_pred_images = []

        # Get validation images (same logic as parent)
        if validate_all:
            img_ids = self.coco.getImgIds()
            validation_images = []
            for img_id in img_ids:
                img_info = self.coco.loadImgs(img_id)[0]
                filename = img_info['file_name']
                ann_ids = self.coco.getAnnIds(imgIds=img_id)
                num_persons = len(ann_ids)
                validation_images.append({
                    'filename': filename,
                    'num_persons': num_persons
                })
            validation_images.sort(key=lambda x: x['filename'])
        else:
            # Use default validation images with fallback
            missing_images = {}
            validation_images = []

            for val_img_info in self.DEFAULT_VALIDATION_IMAGES:
                filename = val_img_info['filename']
                img_ids = self.coco.getImgIds()
                img_id = None

                for iid in img_ids:
                    img_info = self.coco.loadImgs(iid)[0]
                    if img_info['file_name'] == filename:
                        img_id = iid
                        break

                if img_id is None:
                    missing_images[val_img_info['num_persons']] = filename
                else:
                    validation_images.append(val_img_info)

            if missing_images:
                fallback_map = self._find_fallback_images(missing_images)
                for fallback_info in fallback_map.values():
                    validation_images.append(fallback_info)

            validation_images.sort(key=lambda x: x['num_persons'])

        # Process each validation image
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

            # Load and process image
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
                    from pycocotools import mask as mask_utils
                    mask = mask_utils.decode(ann['segmentation'])

                # Resize mask
                import cv2
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)

                # Apply mask to GT image
                gt_image_np = np.array(gt_pil)
                gt_image_np = self._apply_mask(gt_image_np, mask_resized, color)
                gt_pil = Image.fromarray(gt_image_np)
                draw = ImageDraw.Draw(gt_pil)

            # Draw instance numbers on GT image
            for bbox, instance_id, color in instance_info:
                self._draw_instance_number(draw, bbox, instance_id, color)

            gt_image = np.array(gt_pil)

            # Convert to tensor for model
            image_tensor = F.to_tensor(image_resized).unsqueeze(0).to(self.device)

            # Get model predictions
            pred_image = image_np.copy()
            pred_pil = Image.fromarray(pred_image)
            pred_draw = ImageDraw.Draw(pred_pil)

            # Get predictions for each instance
            with torch.no_grad():
                # For multiscale models, extract features through the model
                if hasattr(self.model, 'yolo_extractor'):
                    # RGB-enhanced variable ROI model
                    features = self.model.yolo_extractor.extract_features(image_tensor)
                elif hasattr(self.model, 'extractor'):
                    # Standard variable ROI model
                    features = self.model.extractor.extract_features(image_tensor)
                elif hasattr(self.model, 'feature_extractor'):
                    # Standard multiscale model
                    features = self.model.feature_extractor.extract_features(image_tensor)
                else:
                    # For base models, use separate feature extractor
                    features = self.feature_extractor.extract_features(image_tensor)

                for bbox, instance_id, color in instance_info:
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h

                    # Normalize to [0, 1]
                    roi_norm = np.array([[x1/640, y1/640, x2/640, y2/640]], dtype=np.float32)

                    # Convert to tensor and prepare ROI
                    roi_tensor = torch.from_numpy(roi_norm).float()
                    rois = ROIBatchProcessor.prepare_rois_for_batch(roi_tensor).to(self.device)

                    # Get prediction - MULTISCALE MODEL INTERFACE
                    # Check model type and call appropriately
                    if hasattr(self.model, 'hierarchical_head'):
                        # Hierarchical model - returns (logits, aux_outputs)
                        predictions = self.model(features, rois)
                        if isinstance(predictions, tuple):
                            mask_logits, _ = predictions  # Ignore aux outputs for visualization
                        else:
                            mask_logits = predictions
                    elif (hasattr(self.model, 'rgb_encoder') and self.model.rgb_encoder is not None and 
                        hasattr(self.model, 'use_rgb') and self.model.use_rgb):
                        # RGB enhanced model - extract RGB features
                        rgb_features = self.model.rgb_encoder(image_tensor)
                        predictions = self.model.segmentation_head(features, rois, rgb_features)
                        mask_logits = predictions
                    else:
                        # Standard model - call segmentation head directly with pre-extracted features
                        predictions = self.model.segmentation_head(features, rois)
                        if isinstance(predictions, tuple):
                            # Cascade output
                            mask_logits = predictions[-1]
                        else:
                            mask_logits = predictions

                    # Get predicted class
                    mask_pred = torch.argmax(mask_logits[0], dim=0).cpu().numpy()

                    # Create full image mask
                    full_mask = np.zeros((640, 640), dtype=np.uint8)

                    # Calculate exact slice dimensions
                    y1_int, y2_int = int(y1), int(y2)
                    x1_int, x2_int = int(x1), int(x2)
                    slice_h = y2_int - y1_int
                    slice_w = x2_int - x1_int

                    # Resize mask to exact slice dimensions
                    if slice_h > 0 and slice_w > 0:
                        import cv2
                        mask_resized = cv2.resize(mask_pred.astype(np.uint8), (slice_w, slice_h), interpolation=cv2.INTER_LINEAR)
                        full_mask[y1_int:y2_int, x1_int:x2_int] = mask_resized

                    # Apply predicted mask (only target class = 1)
                    target_mask = (full_mask == 1)
                    if target_mask.any():
                        pred_image_np = np.array(pred_pil)
                        pred_image_np = self._apply_mask(pred_image_np, target_mask, color)
                        pred_pil = Image.fromarray(pred_image_np)
                        pred_draw = ImageDraw.Draw(pred_pil)

                    # Draw prediction bbox (green)
                    self._draw_bbox(pred_draw, bbox, color=(0, 255, 0), width=3)

                    # Update combined mask
                    pred_masks_combined = np.maximum(pred_masks_combined, full_mask)

                # Draw instance numbers on prediction image
                for bbox, instance_id, color in instance_info:
                    self._draw_instance_number(pred_draw, bbox, instance_id, color)

                pred_image = np.array(pred_pil)

            all_gt_images.append(Image.fromarray(gt_image))
            all_pred_images.append(Image.fromarray(pred_image))

        # Create combined visualization with same layout as base implementation
        if all_gt_images:
            from PIL import ImageFont

            n_images = len(all_gt_images)

            # Scale factor and dimensions (matching base implementation)
            scale_factor = 1.0
            img_width = int(640 * scale_factor)
            img_height = int(640 * scale_factor)
            padding = int(20 * scale_factor)

            # Scale all images
            gt_images = [img.resize((img_width, img_height), Image.LANCZOS) for img in all_gt_images]
            pred_images = [img.resize((img_width, img_height), Image.LANCZOS) for img in all_pred_images]

            # Total dimensions
            total_width = n_images * img_width + (n_images + 1) * padding
            total_height = 2 * img_height + 3 * padding

            # Create blank canvas with white background
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
                # Increase font size by 1.5x
                font_size = int(30 * scale_factor * 1.5)
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Calculate text sizes for both labels
            padding_text = 10
            gt_text = "Ground Truth"
            pred_text = "Predictions"

            # Get text bounding boxes
            gt_bbox = draw.textbbox((0, 0), gt_text, font=font)
            pred_bbox = draw.textbbox((0, 0), pred_text, font=font)

            gt_text_width = gt_bbox[2] - gt_bbox[0]
            pred_text_width = pred_bbox[2] - pred_bbox[0]

            # Use the maximum width for both rectangles
            rect_width = max(gt_text_width, pred_text_width) + 2 * padding_text

            # Helper function to draw text in colored rectangle with fixed width
            def draw_text_in_rectangle(draw, pos, text, bg_color, text_color, font, rect_width, padding=10):
                x, y = pos

                # Get text bounding box
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Draw background rectangle with fixed width
                rect_x1 = x
                rect_y1 = y
                rect_x2 = x + rect_width
                rect_y2 = y + text_height + 2 * padding
                draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=bg_color)

                # Draw text centered horizontally in rectangle
                text_x = rect_x1 + (rect_width - text_width) // 2
                # Adjust y position to account for text baseline
                text_y = rect_y1 + padding - bbox[1]
                draw.text((text_x, text_y), text, fill=text_color, font=font)

            # Label for GT row (white text in light red rectangle)
            # Light red: RGB(255, 102, 102)
            draw_text_in_rectangle(draw, (10, 5), gt_text, bg_color=(255, 102, 102), text_color='white', font=font, rect_width=rect_width)

            # Label for prediction row (white text in dark green rectangle)
            # Dark green: RGB(0, 153, 0)
            draw_text_in_rectangle(draw, (10, img_height + padding + 5), pred_text, bg_color=(0, 153, 0), text_color='white', font=font, rect_width=rect_width)

            # Resize to 60% before saving
            resized_width = int(combined.width * 0.6)
            resized_height = int(combined.height * 0.6)
            combined_resized = combined.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

            # Save image
            filename = f"validation_all_images_epoch_{epoch+1:04d}.png"
            output_path = self.output_dir / filename
            combined_resized.save(output_path)
            print(f"Saved validation visualization (60% size): {output_path}")