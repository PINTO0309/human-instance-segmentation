"""Visualization tools for validation results with auxiliary task support.

To enable ROI patch visualization in the 5th row:
1. Modify your model's forward method to return ROI patches in aux_outputs:
   
   Example for HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet:
   ```python
   def forward(self, images, rois):
       # ... existing code ...
       
       # Extract ROI patches for visualization
       roi_rgb_patches = self.roi_align_rgb(images, rois, self.roi_size[0], self.roi_size[1])
       
       # ... rest of processing ...
       
       # Add to aux_outputs
       aux_outputs['roi_patches'] = roi_rgb_patches
       
       return predictions, aux_outputs
   ```

2. The visualization will automatically detect and display these patches in the ROI Comparison panel.
"""

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from pycocotools.coco import COCO

from .feature_extractor import YOLOv9FeatureExtractor
from .model import ROISegmentationModel, ROIBatchProcessor


class ValidationVisualizerWithAuxiliary:
    """Visualizer for validation results with auxiliary task support."""

    # Default validation images as per requirements
    DEFAULT_VALIDATION_IMAGES = [
        {'filename': '000000020992.jpg', 'num_persons': 1},
        {'filename': '000000413552.jpg', 'num_persons': 2},
        {'filename': '000000109118.jpg', 'num_persons': 3},
        {'filename': '000000162732.jpg', 'num_persons': 5}
    ]

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: Optional[YOLOv9FeatureExtractor],
        coco: COCO,
        image_dir: str,
        output_dir: str = 'validation_results',
        device: str = 'cuda',
        roi_padding: float = 0.0,
        visualize_auxiliary: bool = True
    ):
        """Initialize visualizer with auxiliary task support.

        Args:
            model: Trained segmentation model (may include auxiliary task)
            feature_extractor: YOLO feature extractor (optional for integrated models)
            coco: COCO dataset object
            image_dir: Directory containing images
            output_dir: Directory to save visualization results
            device: Device to run inference on
            roi_padding: ROI padding ratio (0.0 = no padding, 0.1 = 10% padding)
            visualize_auxiliary: Whether to visualize auxiliary predictions
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.coco = coco
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.device = device
        self.roi_padding = roi_padding
        self.visualize_auxiliary = visualize_auxiliary

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Check if model has auxiliary task
        self.has_auxiliary = hasattr(model, 'aux_head') or (
            hasattr(model, 'model') and hasattr(model.model, 'aux_head')
        )

        # ROI batch processor (static class, no constructor args)
        self.roi_processor = ROIBatchProcessor
        
        # Create custom color palette avoiding red and yellow
        self._create_custom_colors()
    
    def _create_custom_colors(self):
        """Create custom color palette avoiding red and yellow colors."""
        # Define colors manually to avoid red and yellow
        # Using blue, green, purple, cyan, pink, brown, olive, navy, teal, and gray
        self.custom_colors = np.array([
            [0.12, 0.47, 0.71],    # Blue
            [0.17, 0.63, 0.17],    # Green
            [0.44, 0.16, 0.89],    # Purple
            [0.0, 0.75, 0.75],     # Cyan
            [0.89, 0.41, 0.76],    # Pink
            [0.55, 0.27, 0.07],    # Brown
            [0.33, 0.42, 0.18],    # Olive
            [0.0, 0.0, 0.5],       # Navy
            [0.0, 0.5, 0.5],       # Teal
            [0.5, 0.5, 0.5]        # Gray
        ])

    def _find_fallback_images(self, missing_images: Dict[int, str]) -> Dict[int, Dict]:
        """Find fallback images with the same number of persons.

        Args:
            missing_images: Dictionary mapping num_persons to missing filename

        Returns:
            Dictionary mapping num_persons to fallback image info
        """
        fallback_map = {}

        # Get all available images with their person counts
        available_images = []
        img_ids = self.coco.getImgIds()

        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])  # Person category
            num_persons = len(ann_ids)

            if num_persons > 0:  # Only consider images with people
                available_images.append({
                    'filename': img_info['file_name'],
                    'num_persons': num_persons,
                    'img_id': img_id
                })

        # Find fallbacks for each missing image
        for target_num_persons, missing_filename in missing_images.items():

            # First try to find exact match
            candidates = [img for img in available_images if img['num_persons'] == target_num_persons]

            # If no exact match, find closest
            if not candidates:
                candidates = sorted(available_images,
                                  key=lambda x: abs(x['num_persons'] - target_num_persons))[:5]

            if candidates:
                # Pick the first suitable candidate
                fallback = candidates[0]
                fallback_map[target_num_persons] = {
                    'filename': fallback['filename'],
                    'num_persons': fallback['num_persons'],
                    'is_fallback': True,
                    'original_request': missing_filename
                }
                print(f"Looking for fallback for {missing_filename} with {target_num_persons} persons - Using fallback: {fallback['filename']} ({fallback['num_persons']} persons)")

        return fallback_map

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

    def extract_roi_from_bbox(
        self,
        bbox: List[float],
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Extract ROI coordinates from bounding box with padding.

        Args:
            bbox: [x1, y1, x2, y2] format
            image_shape: (height, width) of image

        Returns:
            ROI coordinates (x1, y1, x2, y2) or None if invalid
        """
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        # Add padding
        pad_x = w * self.roi_padding
        pad_y = h * self.roi_padding

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image_shape[1], int(x2 + pad_x))
        y2 = min(image_shape[0], int(y2 + pad_y))

        # Check if ROI is valid
        if x2 <= x1 or y2 <= y1:
            return None

        return (x1, y1, x2, y2)

    def visualize_validation_images(
        self,
        epoch: int,
        specific_images: Optional[List[Dict]] = None
    ):
        """Visualize predictions on validation images with auxiliary results.

        Args:
            epoch: Current epoch number
            specific_images: List of specific images to visualize
        """
        # Check which default images exist and find fallbacks if needed
        if specific_images:
            images_to_visualize = specific_images
        else:
            missing_images = {}
            images_to_visualize = []

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
                    images_to_visualize.append(val_img_info)

            # Find fallback images if needed
            if missing_images:
                fallback_map = self._find_fallback_images(missing_images)
                # Add fallback images to validation list
                for fallback_info in fallback_map.values():
                    images_to_visualize.append(fallback_info)

            # Sort by number of persons for consistent ordering
            images_to_visualize.sort(key=lambda x: x['num_persons'])

        self.model.eval()

        # Collect all panel images for combined visualization
        all_panel1_images = []  # Ground Truth
        all_panel2_images = []  # Binary Mask Heatmap
        all_panel3_images = []  # Full-image UNet Output
        all_panel4_images = []  # Predictions
        all_panel5_images = []  # ROI Comparison

        with torch.no_grad():
            # Synchronize CUDA to ensure clean state
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for img_info in images_to_visualize:
                filename = img_info['filename']
                image_path = self.image_dir / filename

                if not image_path.exists():
                    print(f"Warning: Image {filename} not found")
                    continue

                # Load image and resize to 640x640
                image = Image.open(image_path).convert('RGB')
                orig_width, orig_height = image.size
                image = image.resize((640, 640), Image.BILINEAR)
                image_np = np.array(image)

                # Get annotations
                img_id = None
                for img_data in self.coco.imgs.values():
                    if img_data['file_name'] == filename:
                        img_id = img_data['id']
                        break

                if img_id is None:
                    print(f"Warning: No annotations found for {filename}")
                    continue

                ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=[1])  # Person category
                anns = self.coco.loadAnns(ann_ids)

                # Get predictions with UNet outputs
                predictions, auxiliary_pred, unet_outputs = self._get_predictions_with_unet(
                    image_np, anns, orig_width, orig_height, img_info.get('target_index', 0)
                )

                # Create individual panel images
                # Panel 1: Ground Truth
                panel1_img = self._create_panel_ground_truth(image_np, anns, orig_width, orig_height)

                # Panel 2: Binary Mask Heatmap (from pre-trained UNet)
                # Use full_image_logits for Binary Mask Heatmap to show pre-trained UNet output
                pretrained_unet_heatmap = None
                if 'full_image_logits' in unet_outputs:
                    full_image_logits = unet_outputs['full_image_logits']
                    if isinstance(full_image_logits, torch.Tensor):
                        # Apply sigmoid to convert logits to probabilities
                        pretrained_unet_probs = torch.sigmoid(full_image_logits).detach().cpu().numpy()
                        pretrained_unet_heatmap = pretrained_unet_probs[0, 0]  # Shape: (640, 640)
                panel2_img = self._create_panel_heatmap(image_np, pretrained_unet_heatmap)

                # Panel 3: Full-image UNet Output (moved from panel 5)
                panel3_img = self._create_panel_full_image_unet(image_np, unet_outputs)

                # Panel 4: Predictions
                panel4_img = self._create_panel_predictions(image_np, anns, predictions, orig_width, orig_height)
                
                # Panel 5: ROI Comparison
                panel5_img = self._create_panel_roi_comparison(image_np, anns, unet_outputs)

                # Add to lists
                all_panel1_images.append(panel1_img)
                all_panel2_images.append(panel2_img)
                all_panel3_images.append(panel3_img)
                all_panel4_images.append(panel4_img)
                all_panel5_images.append(panel5_img)

        # Create combined 5x4 grid image
        if all_panel1_images:
            self._create_combined_5x4_image(
                all_panel1_images, all_panel2_images, all_panel3_images, all_panel4_images, all_panel5_images, epoch
            )

    def _get_predictions(
        self,
        image_np: np.ndarray,
        annotations: List[Dict],
        target_index: int = 0
    ) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray]]:
        """Get model predictions with auxiliary outputs.

        Returns:
            predictions: Dictionary mapping annotation index to predicted mask
            auxiliary_pred: Auxiliary foreground/background prediction (if available)
        """
        # Extract features
        if self.feature_extractor is not None:
            # External feature extraction
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hasattr(self.feature_extractor, 'extract_features'):
                    features = self.feature_extractor.extract_features(image_tensor)
                else:
                    features = self.feature_extractor(image_tensor)
        else:
            # Integrated feature extraction
            features = None

        # Extract ROIs
        roi_list = []
        valid_indices = []

        for idx, ann in enumerate(annotations):
            if 'bbox' not in ann:
                continue

            x, y, w, h = ann['bbox']
            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                image_np.shape[:2]
            )

            if roi is not None:
                roi_list.append(roi)
                valid_indices.append(idx)

        if not roi_list:
            return {}, None

        # Batch process ROIs
        predictions = {}
        auxiliary_preds = []

        batch_size = 16
        for i in range(0, len(roi_list), batch_size):
            batch_rois = roi_list[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]

            # Process batch
            if features is not None:
                # External features
                roi_features = self._extract_roi_features(features, batch_rois)

                # Check if this is an auxiliary wrapped model
                if hasattr(self.model, 'aux_head'):
                    # Need to prepare ROIs for auxiliary model
                    roi_tensors = []
                    for j, roi in enumerate(batch_rois):
                        x1, y1, x2, y2 = roi
                        # Use batch index 0 since we're processing one image at a time
                        roi_norm = torch.tensor(
                            [[0, x1/640, y1/640, x2/640, y2/640]],
                            dtype=torch.float32, device=self.device
                        )
                        roi_tensors.append(roi_norm)
                    rois = torch.cat(roi_tensors, dim=0)
                    batch_pred = self.model(features, rois)
                    # Synchronize CUDA after model inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                else:
                    # Check if this is a hierarchical model that needs both features and rois
                    model_name = type(self.model).__name__
                    if 'Hierarchical' in model_name:
                        # Hierarchical models need the original features and ROIs
                        roi_tensors = []
                        for j, roi in enumerate(batch_rois):
                            x1, y1, x2, y2 = roi
                            # Use batch index 0 since we're processing one image at a time
                            roi_norm = torch.tensor(
                                [[0, x1/640, y1/640, x2/640, y2/640]],
                                dtype=torch.float32, device=self.device
                            )
                            roi_tensors.append(roi_norm)
                        rois = torch.cat(roi_tensors, dim=0)
                        batch_pred = self.model(features, rois)
                    else:
                        # Direct model call with ROI features
                        batch_pred = self.model(roi_features)

                    # Synchronize CUDA after model inference
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
            else:
                # Integrated model or RGB hierarchical model
                # Check if it's an RGB hierarchical or multi-scale RGB model
                model_name = self.model.__class__.__name__
                if 'RGB' in model_name and ('Hierarchical' in model_name or 'MultiScale' in model_name):
                    # RGB hierarchical model needs full image and ROIs
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)

                    # Prepare ROIs in the correct format
                    roi_tensors = []
                    for roi in batch_rois:
                        x1, y1, x2, y2 = roi
                        # Normalize ROI coordinates to [0,1] range as expected by the model
                        # The model uses spatial_scale=640 with normalized coordinates
                        roi_tensor = torch.tensor(
                            [[0, x1/640, y1/640, x2/640, y2/640]],
                            dtype=torch.float32, device=self.device
                        )
                        roi_tensors.append(roi_tensor)
                    rois = torch.cat(roi_tensors, dim=0)

                    batch_pred = self.model(image_tensor, rois)
                else:
                    # Standard integrated model
                    batch_tensor = self._prepare_batch_tensor(image_np, batch_rois)
                    batch_pred = self.model(batch_tensor)

                # Synchronize CUDA after model inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            # Handle auxiliary outputs
            if isinstance(batch_pred, tuple):
                main_pred, aux_outputs = batch_pred
                if 'fg_bg_binary' in aux_outputs:
                    auxiliary_preds.append(torch.sigmoid(aux_outputs['fg_bg_binary']))
            else:
                main_pred = batch_pred

            # Get predictions
            try:
                # Move to CPU first to avoid CUDA errors
                if main_pred.is_cuda:
                    main_pred = main_pred.cpu()
                batch_masks = torch.argmax(main_pred, dim=1).numpy()

                # Store predictions
                for j, idx in enumerate(batch_indices):
                    if j < len(batch_masks):
                        predictions[idx] = batch_masks[j]
            except Exception as e:
                print(f"Warning: Error getting predictions: {e}")
                # Continue without storing predictions for this batch
                pass

        # Combine auxiliary predictions into a heatmap
        auxiliary_pred = None
        if auxiliary_preds:
            # Create a single heatmap by placing each prediction in its ROI location
            auxiliary_heatmap = np.zeros((640, 640), dtype=np.float32)

            for i, (roi, aux_pred) in enumerate(zip(roi_list, auxiliary_preds)):
                x1, y1, x2, y2 = roi
                # Resize auxiliary prediction to ROI size
                if isinstance(aux_pred, torch.Tensor):
                    aux_pred = aux_pred.cpu().numpy()

                aux_resized = cv2.resize(
                    aux_pred.squeeze(),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LINEAR
                )

                # Place in heatmap
                auxiliary_heatmap[y1:y2, x1:x2] = np.maximum(
                    auxiliary_heatmap[y1:y2, x1:x2],
                    aux_resized
                )

            auxiliary_pred = auxiliary_heatmap

        return predictions, auxiliary_pred

    def _plot_auxiliary(
        self,
        ax: plt.Axes,
        image_np: np.ndarray,
        annotations: List[Dict],
        auxiliary_pred: np.ndarray
    ):
        """Plot auxiliary foreground/background predictions."""
        ax.imshow(image_np)
        ax.set_title('Auxiliary FG/BG Prediction')
        ax.axis('off')

        # Create overlay for auxiliary predictions
        overlay = np.zeros(image_np.shape[:2])

        for idx, (ann, aux_mask) in enumerate(zip(annotations, auxiliary_pred)):
            if 'bbox' not in ann:
                continue

            x, y, w, h = ann['bbox']
            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                image_np.shape[:2]
            )

            if roi is None:
                continue

            # Resize auxiliary prediction to ROI size
            aux_resized = cv2.resize(
                aux_mask.squeeze(),
                (roi[2] - roi[0], roi[3] - roi[1]),
                interpolation=cv2.INTER_LINEAR
            )

            # Apply to overlay
            overlay[roi[1]:roi[3], roi[0]:roi[2]] = np.maximum(
                overlay[roi[1]:roi[3], roi[0]:roi[2]],
                aux_resized
            )

        # Show as heatmap
        masked_overlay = np.ma.masked_where(overlay < 0.1, overlay)
        im = ax.imshow(masked_overlay, cmap='hot', alpha=0.6, vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_original_with_boxes(self, ax: plt.Axes, image_np: np.ndarray, annotations: List[Dict]):
        """Plot original image with bounding boxes."""
        ax.imshow(image_np)
        ax.set_title('Original Image with Boxes')
        ax.axis('off')

        # Draw bounding boxes
        for idx, ann in enumerate(annotations):
            if 'bbox' not in ann:
                continue

            x, y, w, h = ann['bbox']
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor='red' if idx == 0 else 'yellow',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x, y-5, f'Person {idx+1}', color='white', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='red' if idx == 0 else 'yellow', alpha=0.7))

    def _plot_ground_truth(self, ax: plt.Axes, image_np: np.ndarray, annotations: List[Dict], target_index: int):
        """Plot ground truth masks."""
        ax.imshow(image_np)
        ax.set_title('Ground Truth Masks')
        ax.axis('off')

        # Create mask overlay
        mask_overlay = np.zeros((*image_np.shape[:2], 4))
        colors = self.custom_colors

        for idx, ann in enumerate(annotations):
            if 'segmentation' not in ann:
                continue

            mask = self.coco.annToMask(ann)
            color = colors[idx % 10]
            mask_overlay[mask > 0] = [*color[:3], 0.6]

        ax.imshow(mask_overlay)

    def _plot_predictions(self, ax: plt.Axes, image_np: np.ndarray, annotations: List[Dict], predictions: Dict[int, np.ndarray]):
        """Plot predicted masks."""
        ax.imshow(image_np)
        ax.set_title('Predicted Masks')
        ax.axis('off')

        # Create prediction overlay
        pred_overlay = np.zeros((*image_np.shape[:2], 4))
        colors = self.custom_colors

        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']
            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                image_np.shape[:2]
            )

            if roi is None:
                continue

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (roi[2] - roi[0], roi[3] - roi[1]),
                interpolation=cv2.INTER_NEAREST
            )

            # Apply colors
            color = colors[idx % 10]
            roi_overlay = pred_overlay[roi[1]:roi[3], roi[0]:roi[2]]

            # Class 1: Target (full opacity)
            roi_overlay[pred_resized == 1] = [*color[:3], 0.8]

            # Class 2: Non-target (lighter)
            roi_overlay[pred_resized == 2] = [*color[:3], 0.4]

        ax.imshow(pred_overlay)

    def _plot_overlay(self, ax: plt.Axes, image_np: np.ndarray, annotations: List[Dict],
                     predictions: Dict[int, np.ndarray], target_index: int):
        """Plot overlay comparison of GT and predictions."""
        ax.imshow(image_np)
        ax.set_title('GT (Green) vs Pred (Red) Overlay')
        ax.axis('off')

        # Create overlay
        overlay = np.zeros((*image_np.shape[:2], 4))

        # Ground truth in green
        for idx, ann in enumerate(annotations):
            if 'segmentation' not in ann:
                continue

            if idx == target_index:
                mask = self.coco.annToMask(ann)
                overlay[mask > 0, 1] = 1.0  # Green channel
                overlay[mask > 0, 3] = 0.5  # Alpha

        # Predictions in red
        if target_index in predictions:
            ann = annotations[target_index]
            pred_mask = predictions[target_index]
            x, y, w, h = ann['bbox']
            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                image_np.shape[:2]
            )

            if roi is not None:
                pred_resized = cv2.resize(
                    (pred_mask == 1).astype(np.uint8),
                    (roi[2] - roi[0], roi[3] - roi[1]),
                    interpolation=cv2.INTER_NEAREST
                )

                roi_overlay = overlay[roi[1]:roi[3], roi[0]:roi[2]]
                roi_overlay[pred_resized > 0, 0] = 1.0  # Red channel
                roi_overlay[pred_resized > 0, 3] = 0.5  # Alpha

        ax.imshow(overlay)

    def _create_panel_ground_truth(self, image_np: np.ndarray, annotations: List[Dict],
                                  orig_width: int, orig_height: int) -> Image.Image:
        """Create ground truth panel image."""
        # Create a copy to draw on
        img = Image.fromarray(image_np.copy())

        # Create mask overlay
        mask_overlay = np.zeros((*image_np.shape[:2], 4))
        colors = self.custom_colors

        # First draw masks
        for idx, ann in enumerate(annotations):
            # Draw segmentation mask
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                # Resize mask to 640x640
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                color = colors[idx % 10]
                mask_overlay[mask_resized > 0] = [*color[:3], 0.6]

        # Apply mask overlay
        img_np = np.array(img)
        mask_rgb = (mask_overlay[:, :, :3] * 255).astype(np.uint8)
        mask_alpha = mask_overlay[:, :, 3]

        for c in range(3):
            img_np[:, :, c] = img_np[:, :, c] * (1 - mask_alpha) + mask_rgb[:, :, c] * mask_alpha

        # Convert back to PIL for drawing
        img = Image.fromarray(img_np.astype(np.uint8))
        draw = ImageDraw.Draw(img)

        # Then draw bounding boxes and instance numbers
        for idx, ann in enumerate(annotations):
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Scale bbox to resized image
                x = x * 640 / orig_width
                y = y * 640 / orig_height
                w = w * 640 / orig_width
                h = h * 640 / orig_height

                color = colors[idx % 10]
                color_rgb = tuple(int(c * 255) for c in color[:3])

                # Draw red box
                draw.rectangle([x, y, x+w, y+h], outline=(255, 0, 0), width=3)

                # Draw instance number outside box
                self._draw_instance_number(draw, [x, y, w, h], idx + 1, color_rgb)

        return img

    def _create_panel_heatmap(self, image_np: np.ndarray, auxiliary_pred: Optional[np.ndarray]) -> Image.Image:
        """Create auxiliary heatmap panel with colorbar."""
        img = image_np.copy()

        # Create wider canvas to accommodate colorbar
        colorbar_width = 60
        canvas_width = img.shape[1] + colorbar_width
        canvas = np.ones((img.shape[0], canvas_width, 3), dtype=np.uint8) * 255

        # Always create the colorbar for consistency
        cmap = plt.colormaps['hot']

        if auxiliary_pred is not None and auxiliary_pred.max() > 0:
            # Create heatmap overlay
            heatmap = cmap(auxiliary_pred)[:, :, :3]  # Remove alpha channel
            heatmap = (heatmap * 255).astype(np.uint8)

            # Blend with original image
            alpha = 0.6
            img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)

        # Place the image (with or without heatmap) on canvas
        canvas[:, :img.shape[1]] = img

        # Always create colorbar for consistency
        colorbar_height = int(img.shape[0] * 0.8)  # 80% of image height
        colorbar_y_start = int((img.shape[0] - colorbar_height) / 2)

        # Generate colorbar gradient
        gradient = np.linspace(1, 0, colorbar_height).reshape(-1, 1)
        gradient = np.repeat(gradient, 20, axis=1)  # Width of colorbar
        colorbar_img = cmap(gradient)[:, :, :3]
        colorbar_img = (colorbar_img * 255).astype(np.uint8)

        # Place colorbar on canvas
        colorbar_x_start = img.shape[1] + 15
        canvas[colorbar_y_start:colorbar_y_start + colorbar_height,
               colorbar_x_start:colorbar_x_start + 20] = colorbar_img

        # Add colorbar labels
        pil_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil_canvas)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Add text labels
        draw.text((colorbar_x_start + 25, colorbar_y_start - 5), "1.0", fill='black', font=font)
        draw.text((colorbar_x_start + 25, colorbar_y_start + colorbar_height - 15), "0.0", fill='black', font=font)
        draw.text((colorbar_x_start + 25, colorbar_y_start + colorbar_height // 2 - 8), "0.5", fill='black', font=font)

        # Add border around colorbar
        draw.rectangle([colorbar_x_start-1, colorbar_y_start-1,
                       colorbar_x_start + 20, colorbar_y_start + colorbar_height],
                      outline='black', width=1)

        return pil_canvas

    def _create_panel_unet_fg_bg(self, image_np: np.ndarray, unet_outputs: Dict[str, np.ndarray]) -> Image.Image:
        """Create UNet FG/BG panel."""
        img = image_np.copy()

        # Get masks from unet_outputs
        combined_fg_mask = unet_outputs.get('combined_fg_mask', np.zeros((640, 640), dtype=bool))
        combined_bg_mask = unet_outputs.get('combined_bg_mask', np.zeros((640, 640), dtype=bool))

        if combined_fg_mask.any() or combined_bg_mask.any():
            # Create overlay
            overlay = img.copy()

            # Background in gray
            overlay[combined_bg_mask] = [128, 128, 128]

            # Foreground in red (overwrites background)
            overlay[combined_fg_mask] = [255, 0, 0]

            # Blend with original
            alpha = 0.7
            img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return Image.fromarray(img)

    def _create_panel_predictions(self, image_np: np.ndarray, annotations: List[Dict],
                                 predictions: Dict[int, np.ndarray], orig_width: int, orig_height: int) -> Image.Image:
        """Create predictions panel."""
        img = image_np.copy()

        # Create separate overlays for different visualization layers
        pred_overlay = np.zeros((*image_np.shape[:2], 4))
        error_overlay = np.zeros((*image_np.shape[:2], 4))
        overlap_overlay = np.zeros((*image_np.shape[:2], 4))

        colors = self.custom_colors

        # Track target masks count for overlap detection
        target_mask_count = np.zeros(image_np.shape[:2], dtype=int)

        # First pass: Render individual predictions
        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is None:
                continue

            x1, y1, x2, y2 = roi

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )

            # Apply colors
            color = colors[idx % 10]

            # Create masks for target
            target_mask = pred_resized == 1

            # Count target overlaps
            target_mask_count[y1:y2, x1:x2] += target_mask.astype(int)

            # Class 1: Target (with increased transparency)
            # Store the target mask to apply later (excluding overlap areas)
            roi_pred_overlay = pred_overlay[y1:y2, x1:x2]
            roi_pred_overlay[target_mask] = [*color[:3], 0.6]

        # Second pass: Find overlaps and errors
        # Find areas where multiple targets overlap (count > 1)
        target_overlap_mask = target_mask_count > 1

        # Reconstruct full prediction and GT target masks
        full_pred_target_mask = np.zeros(image_np.shape[:2], dtype=bool)
        full_gt_target_mask = np.zeros(image_np.shape[:2], dtype=bool)

        # Collect all GT target masks
        for ann in annotations:
            if 'segmentation' in ann:
                gt_mask = self.coco.annToMask(ann)
                gt_mask_resized = cv2.resize(gt_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                full_gt_target_mask[gt_mask_resized > 0] = True

        # Collect all target predictions
        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is None:
                continue

            x1, y1, x2, y2 = roi

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )

            # Mark target predictions
            full_pred_target_mask[y1:y2, x1:x2] |= (pred_resized == 1)

        # Find errors
        missed_target_mask = full_gt_target_mask & (~full_pred_target_mask)
        false_positive_mask = full_pred_target_mask & (~full_gt_target_mask)

        # Also find areas where we predicted target but GT says it should be non-target
        # This requires checking each ROI individually
        should_be_nontarget_mask = np.zeros(image_np.shape[:2], dtype=bool)

        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is None:
                continue

            x1, y1, x2, y2 = roi

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )

            # Create GT mask for this ROI
            roi_gt_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)

            # Mark current instance as target (class 1)
            if 'segmentation' in ann:
                instance_mask = self.coco.annToMask(ann)
                instance_mask_resized = cv2.resize(instance_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                roi_instance_mask = instance_mask_resized[y1:y2, x1:x2]
                roi_gt_mask[roi_instance_mask > 0] = 1

            # Mark other instances as non-target (class 2)
            for other_idx, other_ann in enumerate(annotations):
                if other_idx != idx and 'segmentation' in other_ann:
                    other_mask = self.coco.annToMask(other_ann)
                    other_mask_resized = cv2.resize(other_mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                    # Check if this other instance overlaps with our ROI
                    roi_other_mask = other_mask_resized[y1:y2, x1:x2]
                    roi_gt_mask[roi_other_mask > 0] = 2

            # Find areas where we predicted target (1) but GT says non-target (2)
            target_but_should_be_nontarget = (pred_resized == 1) & (roi_gt_mask == 2)
            # Only mark as error if this area is NOT part of multiple target overlap
            # (i.e., only this single instance is predicting target there)
            for j in range(y2 - y1):
                for i in range(x2 - x1):
                    if target_but_should_be_nontarget[j, i]:
                        global_y, global_x = y1 + j, x1 + i
                        # Check if this pixel has multiple target predictions
                        if target_mask_count[global_y, global_x] <= 1:
                            should_be_nontarget_mask[global_y, global_x] = True

        # Combine error types (but exclude areas with multiple target overlaps)
        error_mask = missed_target_mask | false_positive_mask | should_be_nontarget_mask

        # Clear individual target colors from overlap areas first
        pred_overlay[target_overlap_mask] = [0.0, 0.0, 0.0, 0.0]

        # Render overlaps and errors in separate overlays
        # First mark errors in red
        error_overlay[error_mask] = [1.0, 0.0, 0.0, 0.55]  # Red

        # Then overwrite with yellow for overlaps (higher priority)
        # This ensures overlap areas are not shown as errors
        overlap_overlay[target_overlap_mask] = [1.0, 1.0, 0.0, 0.35]  # Yellow

        # Remove error marking from overlap areas
        error_overlay[target_overlap_mask] = [0.0, 0.0, 0.0, 0.0]  # Clear error in overlap areas

        # Apply overlays in order: first predictions, then overlaps, then errors
        # This ensures errors are always visible on top
        mask_rgb = (pred_overlay[:, :, :3] * 255).astype(np.uint8)
        mask_alpha = pred_overlay[:, :, 3]

        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - mask_alpha) + mask_rgb[:, :, c] * mask_alpha

        # Apply overlap overlay
        overlap_rgb = (overlap_overlay[:, :, :3] * 255).astype(np.uint8)
        overlap_alpha = overlap_overlay[:, :, 3]

        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - overlap_alpha) + overlap_rgb[:, :, c] * overlap_alpha

        # Apply error overlay last (highest priority)
        error_rgb = (error_overlay[:, :, :3] * 255).astype(np.uint8)
        error_alpha = error_overlay[:, :, 3]

        for c in range(3):
            img[:, :, c] = img[:, :, c] * (1 - error_alpha) + error_rgb[:, :, c] * error_alpha

        # Add legend within the image (bottom-right corner)
        # Convert to PIL for drawing legend
        pil_img = Image.fromarray(img.astype(np.uint8))
        draw = ImageDraw.Draw(pil_img)

        # Try to load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_bold = font

        # Legend dimensions and position
        legend_width = 130
        legend_height = 80  # Reduced height since no title
        legend_margin = 10
        legend_x = img.shape[1] - legend_width - legend_margin
        legend_y = img.shape[0] - legend_height - legend_margin

        # Draw semi-transparent background for legend
        legend_bg = Image.new('RGBA', (legend_width, legend_height), (240, 240, 240, 230))
        legend_draw = ImageDraw.Draw(legend_bg)

        # Draw border
        legend_draw.rectangle(
            [0, 0, legend_width-1, legend_height-1],
            fill=None,
            outline=(100, 100, 100, 255),
            width=2
        )

        # Draw legend items (no title)
        # Calculate vertical centering
        item_height = 22
        num_items = 3
        # Total height needed: (num_items - 1) * item_height + box height
        total_items_height = (num_items - 1) * item_height + 12
        vertical_padding = (legend_height - total_items_height) // 2
        item_y = vertical_padding  # Start with centered vertical padding

        # Error (Red)
        legend_draw.rectangle([10, item_y, 20, item_y + 12],
                            fill=(255, 0, 0, 255), outline=(0, 0, 0, 255))
        legend_draw.text((25, item_y), "Error", fill=(0, 0, 0, 255), font=font)

        # Overlap (Yellow)
        item_y += item_height
        legend_draw.rectangle([10, item_y, 20, item_y + 12],
                            fill=(255, 255, 0, 255), outline=(0, 0, 0, 255))
        legend_draw.text((25, item_y), "Overlap", fill=(0, 0, 0, 255), font=font)

        # Person prediction (sample color)
        item_y += item_height
        sample_color = tuple(int(c * 255) for c in colors[0][:3]) + (255,)
        legend_draw.rectangle([10, item_y, 20, item_y + 12],
                            fill=sample_color, outline=(0, 0, 0, 255))
        legend_draw.text((25, item_y), "Person pred.", fill=(0, 0, 0, 255), font=font)

        # Paste legend onto main image
        pil_img_rgba = pil_img.convert('RGBA')
        pil_img_rgba.paste(legend_bg, (legend_x, legend_y), legend_bg)

        # Convert back to RGB
        final_img = pil_img_rgba.convert('RGB')

        return final_img

    def _create_panel_full_image_unet(self, image_np: np.ndarray, unet_outputs: Dict) -> Image.Image:
        """Create panel showing full-image UNet output.

        Args:
            image_np: Original image
            unet_outputs: Dictionary containing UNet outputs

        Returns:
            Panel image with full-image UNet output visualization
        """
        img = image_np.copy()

        # Check if full_image_logits is available in unet_outputs
        if 'full_image_logits' in unet_outputs:
            full_image_logits = unet_outputs['full_image_logits']

            # Convert to numpy if tensor
            if isinstance(full_image_logits, torch.Tensor):
                segment = full_image_logits.detach().cpu().numpy()
            else:
                segment = full_image_logits

            # Apply the exact rendering requested by user
            image_height, image_width = img.shape[:2]
            mask = np.zeros((image_height, image_width), dtype=np.uint8)
            resized_segment = cv2.resize(segment[0, 0], (image_width, image_height))
            mask[resized_segment > 0] = 255

            # Create green mask colored
            mask_colored = np.zeros_like(img)
            mask_colored[:, :, 1] = mask  # Green component

            # Overlay with 0.3 opacity as requested
            img = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)

        return Image.fromarray(img.astype(np.uint8))
    
    def _create_panel_roi_comparison(self, image_np: np.ndarray, annotations: List[Dict], unet_outputs: Dict) -> Image.Image:
        """Create panel showing ROI clipping comparison.
        
        Args:
            image_np: Original image (640x640)
            annotations: List of COCO annotations
            unet_outputs: Dictionary containing ROI information
            
        Returns:
            Panel image showing ROI comparison
        """
        panel_width = 700  # Wider to accommodate side-by-side ROIs
        panel_height = 640
        panel_img = Image.new('RGB', (panel_width, panel_height), color=(230, 230, 230))
        draw = ImageDraw.Draw(panel_img)
        
        # Check if we have ROI coordinates
        if 'roi_coordinates' not in unet_outputs or len(unet_outputs['roi_coordinates']) == 0:
            # No ROIs to display
            draw.text((panel_width//2 - 50, panel_height//2), "No ROIs", fill='black')
            return panel_img
        
        roi_coordinates = unet_outputs['roi_coordinates']
        num_rois = min(len(roi_coordinates), 4)  # Display up to 4 ROIs
        
        # Layout configuration
        if num_rois == 1:
            rows, cols = 1, 2
        elif num_rois == 2:
            rows, cols = 1, 4  # 2 ROIs side by side, each with original and aligned
        elif num_rois <= 4:
            rows, cols = 2, 4  # 2x2 grid, each cell split in half
        else:
            rows, cols = 2, 4
        
        # Calculate cell dimensions
        cell_width = (panel_width - 20) // cols
        cell_height = (panel_height - 40) // rows
        
        # Try to get font
        try:
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 10)
        except:
            font_small = ImageFont.load_default()
        
        # Process each ROI
        for idx in range(num_rois):
            x1, y1, x2, y2 = roi_coordinates[idx]
            
            # Extract original ROI from image
            roi_original = image_np[y1:y2, x1:x2]
            roi_h, roi_w = roi_original.shape[:2]
            
            # Calculate position in grid
            if num_rois <= 2:
                row = 0
                col_base = idx * 2
            else:
                row = idx // 2
                col_base = (idx % 2) * 2
            
            # Position for this ROI pair
            x_pos_orig = 10 + col_base * cell_width
            x_pos_align = x_pos_orig + cell_width
            y_pos = 30 + row * cell_height
            
            # Resize ROI to fit in cell (maintaining aspect ratio)
            max_w = cell_width - 10
            max_h = cell_height - 20
            
            if roi_w > 0 and roi_h > 0:
                scale = min(max_w / roi_w, max_h / roi_h, 1.0)
                new_w = int(roi_w * scale)
                new_h = int(roi_h * scale)
                
                roi_resized = cv2.resize(roi_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Convert to PIL Image
                roi_pil = Image.fromarray(roi_resized)
                
                # Paste original ROI
                panel_img.paste(roi_pil, (x_pos_orig, y_pos))
                draw.text((x_pos_orig, y_pos - 15), f"ROI {idx+1} Original", fill='black', font=font_small)
                draw.text((x_pos_orig, y_pos + new_h + 2), f"({roi_w}x{roi_h})", fill='gray', font=font_small)
                
                # For RoIAlign output, we need the model to return it
                # For now, show the same ROI with a note
                if 'roi_patches' in unet_outputs and idx < len(unet_outputs['roi_patches']):
                    # If we have actual RoIAlign outputs
                    roi_patch = unet_outputs['roi_patches'][idx]
                    if isinstance(roi_patch, torch.Tensor):
                        # Convert tensor to numpy
                        roi_patch_np = roi_patch.cpu().numpy()
                        if roi_patch_np.shape[0] == 3:  # CHW format
                            roi_patch_np = np.transpose(roi_patch_np, (1, 2, 0))
                        roi_patch_np = (roi_patch_np * 255).astype(np.uint8)
                        
                        # Resize to match display size
                        roi_patch_resized = cv2.resize(roi_patch_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        roi_patch_pil = Image.fromarray(roi_patch_resized)
                        
                        panel_img.paste(roi_patch_pil, (x_pos_align, y_pos))
                        draw.text((x_pos_align, y_pos - 15), f"RoIAlign Output", fill='blue', font=font_small)
                        draw.text((x_pos_align, y_pos + new_h + 2), f"({roi_patch_np.shape[1]}x{roi_patch_np.shape[0]})", fill='gray', font=font_small)
                    else:
                        # Just show the same ROI with different label
                        panel_img.paste(roi_pil, (x_pos_align, y_pos))
                        draw.text((x_pos_align, y_pos - 15), f"RoIAlign (simulated)", fill='gray', font=font_small)
                else:
                    # No RoIAlign output available, show placeholder
                    panel_img.paste(roi_pil, (x_pos_align, y_pos))
                    draw.text((x_pos_align, y_pos - 15), f"RoIAlign (N/A)", fill='gray', font=font_small)
                    
                # Draw bounding box info
                draw.rectangle([x_pos_orig-1, y_pos-1, x_pos_orig+new_w+1, y_pos+new_h+1], outline='red', width=1)
                draw.rectangle([x_pos_align-1, y_pos-1, x_pos_align+new_w+1, y_pos+new_h+1], outline='blue', width=1)
        
        # Add title
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 16)
        except:
            font_title = font_small
            
        draw.text((panel_width//2 - 80, 5), "ROI Clipping Comparison", fill='black', font=font_title)
        
        return panel_img

    def _create_combined_5x4_image(self, panel1_images: List[Image.Image], panel2_images: List[Image.Image],
                                  panel3_images: List[Image.Image], panel4_images: List[Image.Image],
                                  panel5_images: List[Image.Image], epoch: int):
        """Create combined 5x4 grid visualization."""
        n_images = len(panel1_images)
        if n_images == 0:
            return

        # Fixed dimensions
        img_width = 640
        img_height = 640
        heatmap_width = 700  # Heatmap panel is wider due to colorbar
        padding = 20

        # Add extra space for title at the top
        title_height = 80

        # Create combined image (5 rows x n_images columns)
        # Set margins
        label_padding = 30  # Changed to 30px as requested
        right_margin = 30   # Right margin
        bottom_margin = 30  # New bottom margin
        # Account for wider heatmap panels
        combined_width = label_padding + n_images * heatmap_width + (n_images - 1) * padding + right_margin
        combined_height = title_height + 5 * img_height + 4 * padding + bottom_margin
        combined_img = Image.new('RGB', (combined_width, combined_height), color=(255, 255, 255))

        # Add title text
        draw = ImageDraw.Draw(combined_img)

        # Try to load fonts
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 45)
            label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 45)  # Increased from 30 to 45 (1.5x)
        except:
            title_font = ImageFont.load_default()
            label_font = ImageFont.load_default()

        # Paste images in grid first
        for col in range(n_images):
            # Use wider spacing to accommodate heatmap panels
            x_offset = label_padding + col * (heatmap_width + padding)

            # Row 1: Ground Truth (standard width)
            combined_img.paste(panel1_images[col], (x_offset, title_height))

            # Row 2: Binary Mask Heatmap (wider due to colorbar)
            combined_img.paste(panel2_images[col], (x_offset, title_height + img_height + padding))

            # Row 3: Full-image UNet Output (standard width)
            combined_img.paste(panel3_images[col], (x_offset, title_height + 2 * (img_height + padding)))

            # Row 4: Predictions (standard width)
            combined_img.paste(panel4_images[col], (x_offset, title_height + 3 * (img_height + padding)))
            
            # Row 5: ROI Comparison (wider due to side-by-side display)
            combined_img.paste(panel5_images[col], (x_offset, title_height + 4 * (img_height + padding)))

        # Draw main title
        title_text = f"Validation Results with Auxiliary Task - Epoch {epoch+1:04d}"
        title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (combined_width - title_width) // 2
        draw.text((title_x, 20), title_text, fill='black', font=title_font)

        # Row labels with colors matching hierarchical UNet visualizer
        labels = [
            ("Ground Truth", (255, 102, 102)),  # Light red
            ("Binary Mask Heatmap", (102, 178, 255)),  # Light blue
            ("Full UNet Output", (0, 204, 102)),  # Light green
            ("Predictions", (0, 153, 0)),  # Green
            ("ROI Comparison", (255, 153, 51))  # Orange
        ]

        # Draw row labels on top of images
        for idx, (text, color) in enumerate(labels):
            y_pos = title_height + idx * (img_height + padding) + 5
            # Draw background rectangle
            bbox = draw.textbbox((0, 0), text, font=label_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Increase padding for better visual balance (1.5x increase to match font)
            vertical_padding = 23  # Increased from 15 to 23 (1.5x)
            horizontal_padding = 30  # Increased from 20 to 30 (1.5x)

            # Calculate box dimensions
            box_height = text_height + 2 * vertical_padding

            draw.rectangle(
                [10, y_pos, 10 + text_width + 2 * horizontal_padding, y_pos + box_height],
                fill=color
            )

            # Center text vertically within the box
            # Account for text baseline by using bbox offset
            text_y = y_pos + vertical_padding - bbox[1]
            draw.text((10 + horizontal_padding, text_y), text, fill='white', font=label_font)

        # Resize to 45% of original size
        scale_factor = 0.45
        new_width = int(combined_img.width * scale_factor)
        new_height = int(combined_img.height * scale_factor)
        combined_img_resized = combined_img.resize((new_width, new_height), Image.LANCZOS)

        # Save combined image
        output_path = self.output_dir / f'validation_all_images_epoch_{epoch+1:04d}.png'
        combined_img_resized.save(output_path)
        print(f"  Saved combined visualization: {output_path}")

    def _extract_roi_features(self, features: Union[torch.Tensor, Dict[str, torch.Tensor]], rois: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """Extract ROI features from feature map."""
        # Handle dictionary features (multi-scale models)
        if isinstance(features, dict):
            # For multi-scale models, we need to handle this differently
            # Just return a dummy tensor for now since the model will handle ROI extraction internally
            return torch.zeros(len(rois), 1024, 7, 7).to(self.device)

        # This is a simplified version - in practice, you'd use ROIAlign
        roi_features = []

        for roi in rois:
            x1, y1, x2, y2 = roi
            # Map ROI to feature coordinates
            # Assuming 8x downsampling from 640x640 to 80x80
            fx1 = int(x1 * 80 / 640)
            fy1 = int(y1 * 80 / 640)
            fx2 = int(x2 * 80 / 640)
            fy2 = int(y2 * 80 / 640)

            # Extract and resize
            roi_feat = features[:, :, fy1:fy2, fx1:fx2]
            roi_feat = torch.nn.functional.interpolate(
                roi_feat, size=(7, 7), mode='bilinear', align_corners=False
            )
            roi_features.append(roi_feat)

        return torch.cat(roi_features, dim=0)

    def _prepare_batch_tensor(self, image_np: np.ndarray, rois: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """Prepare batch tensor for integrated models."""
        batch_images = []

        for roi in rois:
            x1, y1, x2, y2 = roi
            roi_image = image_np[y1:y2, x1:x2]

            # Resize to model input size
            roi_image = cv2.resize(roi_image, (640, 640))

            # Convert to tensor
            roi_tensor = torch.from_numpy(roi_image).permute(2, 0, 1).float() / 255.0
            batch_images.append(roi_tensor)

        return torch.stack(batch_images).to(self.device)

    def _get_predictions_with_unet(
        self,
        image_np: np.ndarray,
        annotations: List[Dict],
        orig_width: int,
        orig_height: int,
        target_index: int = 0
    ) -> Tuple[Dict[int, np.ndarray], Optional[np.ndarray], Dict[str, np.ndarray]]:
        """Get model predictions with auxiliary outputs and UNet-style outputs.

        Returns:
            predictions: Dictionary mapping annotation index to predicted mask
            auxiliary_pred: Auxiliary foreground/background prediction (if available)
            unet_outputs: Dictionary containing UNet-style FG/BG masks
        """
        # Initialize UNet outputs
        unet_outputs = {
            'combined_fg_mask': np.zeros((640, 640), dtype=bool),
            'combined_bg_mask': np.zeros((640, 640), dtype=bool),
            'all_auxiliary_preds': [],  # Store all auxiliary predictions
            'roi_coordinates': [],  # Store ROI coordinates for visualization
            'roi_patches': []  # Store ROI patches if available
        }

        # Extract features once for all ROIs
        if self.feature_extractor is not None:
            # External feature extraction
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                if hasattr(self.feature_extractor, 'extract_features'):
                    features = self.feature_extractor.extract_features(image_tensor)
                else:
                    features = self.feature_extractor(image_tensor)
        else:
            # Integrated feature extraction
            features = None

        # Process all ROIs at once for better batching
        roi_list = []
        valid_indices = []
        roi_coordinates = []  # Store ROI coordinates for later use

        for idx, ann in enumerate(annotations):
            if 'bbox' not in ann:
                continue

            x, y, w, h = ann['bbox']
            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is not None:
                roi_list.append(roi)
                valid_indices.append(idx)
                roi_coordinates.append(roi)

        if not roi_list:
            return {}, None, unet_outputs
        
        # Store ROI coordinates for visualization
        unet_outputs['roi_coordinates'] = roi_coordinates

        # Batch process ROIs
        predictions = {}
        all_auxiliary_preds = []

        batch_size = 16
        for i in range(0, len(roi_list), batch_size):
            batch_rois = roi_list[i:i+batch_size]
            batch_indices = valid_indices[i:i+batch_size]
            batch_coords = roi_coordinates[i:i+batch_size]

            # Process batch
            if features is not None:
                # External features - need to handle based on model type
                if hasattr(self.model, 'main_head') and hasattr(self.model.main_head, 'base_model'):
                    # Multi-scale model with auxiliary task
                    # Prepare ROIs for the model
                    roi_tensors = []
                    for roi in batch_rois:
                        x1, y1, x2, y2 = roi
                        roi_norm = torch.tensor(
                            [[0, x1/640, y1/640, x2/640, y2/640]],
                            dtype=torch.float32, device=self.device
                        )
                        roi_tensors.append(roi_norm)
                    rois = torch.cat(roi_tensors, dim=0)

                    batch_pred = self.model(features, rois)
                else:
                    # Check if features is a dict (multi-scale)
                    if isinstance(features, dict):
                        # Multi-scale model without auxiliary wrapper
                        # Prepare ROIs for the model
                        roi_tensors = []
                        for roi in batch_rois:
                            x1, y1, x2, y2 = roi
                            roi_norm = torch.tensor(
                                [[0, x1/640, y1/640, x2/640, y2/640]],
                                dtype=torch.float32, device=self.device
                            )
                            roi_tensors.append(roi_norm)
                        rois = torch.cat(roi_tensors, dim=0)
                        batch_pred = self.model(features, rois)
                    else:
                        # Standard model with tensor features
                        roi_features = self._extract_roi_features(features, batch_rois)
                        # For auxiliary wrapped models, we need to pass both features and rois
                        if hasattr(self.model, 'aux_head'):
                            # Prepare dummy ROIs for the auxiliary model
                            roi_tensors = []
                            for i in range(len(batch_rois)):
                                # Create dummy ROI since we already have ROI features
                                roi_norm = torch.tensor([[i, 0, 0, 1, 1]], dtype=torch.float32, device=self.device)
                                roi_tensors.append(roi_norm)
                            rois = torch.cat(roi_tensors, dim=0)
                            batch_pred = self.model(roi_features, rois)
                        else:
                            # Check if this is a full-image UNet model
                            model_name = self.model.__class__.__name__
                            if 'FullImagePretrainedUNet' in model_name:
                                # Full-image UNet model needs full image and ROIs
                                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                                image_tensor = image_tensor.unsqueeze(0).to(self.device)

                                # Prepare ROIs in the correct format
                                roi_tensors = []
                                for roi in batch_rois:
                                    x1, y1, x2, y2 = roi
                                    # Normalize ROI coordinates to [0,1] range as expected by the model
                                    roi_tensor = torch.tensor(
                                        [[0, x1/640, y1/640, x2/640, y2/640]],
                                        dtype=torch.float32, device=self.device
                                    )
                                    roi_tensors.append(roi_tensor)
                                rois = torch.cat(roi_tensors, dim=0)

                                batch_pred = self.model(image_tensor, rois)
                            else:
                                batch_pred = self.model(roi_features)
            else:
                # Integrated model or RGB hierarchical model
                # Check if it's an RGB hierarchical or multi-scale RGB model
                model_name = self.model.__class__.__name__
                if 'RGB' in model_name and ('Hierarchical' in model_name or 'MultiScale' in model_name):
                    # RGB hierarchical model needs full image and ROIs
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0).to(self.device)

                    # Prepare ROIs in the correct format
                    roi_tensors = []
                    for roi in batch_rois:
                        x1, y1, x2, y2 = roi
                        # Normalize ROI coordinates to [0,1] range as expected by the model
                        # The model uses spatial_scale=640 with normalized coordinates
                        roi_tensor = torch.tensor(
                            [[0, x1/640, y1/640, x2/640, y2/640]],
                            dtype=torch.float32, device=self.device
                        )
                        roi_tensors.append(roi_tensor)
                    rois = torch.cat(roi_tensors, dim=0)

                    batch_pred = self.model(image_tensor, rois)
                else:
                    # Standard integrated model
                    batch_tensor = self._prepare_batch_tensor(image_np, batch_rois)
                    batch_pred = self.model(batch_tensor)

            # Handle auxiliary outputs
            if isinstance(batch_pred, tuple):
                main_pred, aux_outputs = batch_pred
                aux_probs = None

                if 'fg_bg_binary' in aux_outputs:
                    aux_probs = torch.sigmoid(aux_outputs['fg_bg_binary']).cpu().numpy()
                elif 'bg_fg_logits' in aux_outputs:
                    # Handle hierarchical models that output bg_fg_logits
                    bg_fg_logits = aux_outputs['bg_fg_logits']
                    # Convert to foreground probability
                    fg_probs = torch.softmax(bg_fg_logits, dim=1)[:, 1:2, :, :]  # Take foreground channel
                    aux_probs = fg_probs.cpu().numpy()

                # Check for full-image UNet output
                if 'full_image_logits' in aux_outputs:
                    unet_outputs['full_image_logits'] = aux_outputs['full_image_logits']
                
                # Check for ROI patches (if model returns them)
                if 'roi_patches' in aux_outputs:
                    unet_outputs['roi_patches'].extend(aux_outputs['roi_patches'])

                # Process each auxiliary prediction if we have any
                if aux_probs is not None:
                    for j, (roi_coord, aux_pred) in enumerate(zip(batch_coords, aux_probs)):
                        x1, y1, x2, y2 = roi_coord

                        # Resize auxiliary prediction to ROI size
                        aux_resized = cv2.resize(
                            aux_pred.squeeze(),
                            (x2 - x1, y2 - y1),
                            interpolation=cv2.INTER_LINEAR
                        )

                        # Store for heatmap visualization
                        unet_outputs['all_auxiliary_preds'].append({
                            'roi': roi_coord,
                            'pred': aux_resized
                        })

                        # Update combined masks
                        fg_mask_roi = aux_resized >= 0.5
                        bg_mask_roi = aux_resized < 0.5

                        unet_outputs['combined_fg_mask'][y1:y2, x1:x2] |= fg_mask_roi
                        # Only update background where there's no foreground
                        current_bg = unet_outputs['combined_bg_mask'][y1:y2, x1:x2]
                        current_fg = unet_outputs['combined_fg_mask'][y1:y2, x1:x2]
                        unet_outputs['combined_bg_mask'][y1:y2, x1:x2] = current_bg | (bg_mask_roi & ~current_fg)
            else:
                main_pred = batch_pred

            # Get predictions
            try:
                # Move to CPU first to avoid CUDA errors
                if main_pred.is_cuda:
                    main_pred = main_pred.cpu()
                batch_masks = torch.argmax(main_pred, dim=1).numpy()

                # Store predictions
                for j, idx in enumerate(batch_indices):
                    if j < len(batch_masks):
                        predictions[idx] = batch_masks[j]
            except Exception as e:
                print(f"Warning: Error getting predictions: {e}")
                # Continue without storing predictions for this batch
                pass

        # Create a single auxiliary heatmap from all predictions
        auxiliary_pred = None
        if unet_outputs['all_auxiliary_preds']:
            # Create a single heatmap by taking max over all ROIs
            auxiliary_heatmap = np.zeros((640, 640), dtype=np.float32)
            for aux_info in unet_outputs['all_auxiliary_preds']:
                x1, y1, x2, y2 = aux_info['roi']
                auxiliary_heatmap[y1:y2, x1:x2] = np.maximum(
                    auxiliary_heatmap[y1:y2, x1:x2],
                    aux_info['pred']
                )
            auxiliary_pred = auxiliary_heatmap

        return predictions, auxiliary_pred, unet_outputs

    def _plot_ground_truth_combined(
        self,
        ax: plt.Axes,
        image_np: np.ndarray,
        annotations: List[Dict],
        orig_width: int,
        orig_height: int
    ):
        """Plot ground truth with boxes and masks combined."""
        ax.imshow(image_np)
        ax.set_title('Ground Truth')
        ax.axis('off')

        # Create mask overlay
        mask_overlay = np.zeros((*image_np.shape[:2], 4))
        colors = self.custom_colors

        for idx, ann in enumerate(annotations):
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                # Scale bbox to resized image
                x = x * 640 / orig_width
                y = y * 640 / orig_height
                w = w * 640 / orig_width
                h = h * 640 / orig_height

                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2,
                    edgecolor='red',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(x, y-5, f'Person {idx+1}', color='white', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))

            # Draw segmentation mask
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                # Resize mask to 640x640
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                color = colors[idx % 10]
                mask_overlay[mask_resized > 0] = [*color[:3], 0.6]

        ax.imshow(mask_overlay)

    def _plot_unet_fg_bg(
        self,
        ax: plt.Axes,
        image_np: np.ndarray,
        annotations: List[Dict],
        unet_outputs: Dict[str, np.ndarray]
    ):
        """Plot UNet-style foreground/background masks."""
        # Create overlay
        overlay = image_np.copy()

        # Get masks from unet_outputs
        combined_fg_mask = unet_outputs.get('combined_fg_mask', np.zeros((640, 640), dtype=bool))
        combined_bg_mask = unet_outputs.get('combined_bg_mask', np.zeros((640, 640), dtype=bool))

        # Create visualization
        mask_overlay = np.zeros((*image_np.shape[:2], 4))

        # If we have masks, visualize them
        if combined_fg_mask.any() or combined_bg_mask.any():
            # Foreground in red (overwrites background)
            mask_overlay[combined_fg_mask] = [1.0, 0.0, 0.0, 0.55]  # Red with alpha

            ax.imshow(image_np)
            ax.imshow(mask_overlay)
        else:
            # No auxiliary predictions available - show placeholder
            ax.imshow(image_np)
            ax.text(0.5, 0.5, 'No FG/BG predictions available',
                   transform=ax.transAxes,
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

        ax.set_title('Enhanced UNet FG/BG')
        ax.axis('off')

    def _plot_auxiliary_heatmap(
        self,
        ax: plt.Axes,
        image_np: np.ndarray,
        auxiliary_pred: np.ndarray
    ):
        """Plot auxiliary foreground/background prediction as heatmap."""
        ax.imshow(image_np)
        ax.set_title('Binary Mask Heatmap')
        ax.axis('off')

        # auxiliary_pred is already a 640x640 heatmap
        if auxiliary_pred.ndim == 2:
            overlay = auxiliary_pred
        else:
            # If it's still in the old format, handle it
            overlay = np.zeros(image_np.shape[:2])
            for aux_mask in auxiliary_pred:
                overlay = np.maximum(overlay, cv2.resize(
                    aux_mask.squeeze(),
                    (640, 640),
                    interpolation=cv2.INTER_LINEAR
                ))

        # Show as heatmap
        masked_overlay = np.ma.masked_where(overlay < 0.1, overlay)
        im = ax.imshow(masked_overlay, cmap='hot', alpha=0.6, vmin=0, vmax=1)

        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def _plot_final_predictions(
        self,
        ax: plt.Axes,
        image_np: np.ndarray,
        annotations: List[Dict],
        predictions: Dict[int, np.ndarray],
        orig_width: int,
        orig_height: int
    ):
        """Plot final predicted masks."""
        ax.imshow(image_np)
        ax.set_title('Predictions')
        ax.axis('off')

        # Create prediction overlay
        pred_overlay = np.zeros((*image_np.shape[:2], 4))
        colors = self.custom_colors

        # Track target masks count for overlap detection
        target_mask_count = np.zeros(image_np.shape[:2], dtype=int)

        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is None:
                continue

            x1, y1, x2, y2 = roi

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )

            # Apply colors
            color = colors[idx % 10]
            roi_overlay = pred_overlay[y1:y2, x1:x2]

            # Create masks for target and non-target
            target_mask_roi = pred_resized == 1
            # nontarget_mask_roi = pred_resized == 2  # Not needed since we're not rendering non-target

            # Count target overlaps
            target_mask_count[y1:y2, x1:x2] += target_mask_roi.astype(int)

            # Class 1: Target (with increased transparency)
            roi_overlay[target_mask_roi] = [*color[:3], 0.6]  # Reduced from 0.8 to 0.6

            # Class 2: Non-target - skip rendering
            # roi_overlay[nontarget_mask_roi] = [*color[:3], 0.3]  # Commented out - not rendering non-target

        # Find areas where multiple targets overlap (count > 1)
        target_overlap_mask = target_mask_count > 1

        # Render target overlap areas in yellow with slight transparency
        pred_overlay[target_overlap_mask] = [1.0, 1.0, 0.0, 0.35]  # Yellow with slight transparency (alpha = 0.55)

        # Reconstruct full prediction and GT target masks
        full_pred_target_mask = np.zeros(image_np.shape[:2], dtype=bool)
        full_gt_target_mask = np.zeros(image_np.shape[:2], dtype=bool)

        # Collect all GT target masks
        for ann in annotations:
            if 'segmentation' in ann:
                mask = self.coco.annToMask(ann)
                # Resize mask to 640x640
                mask_resized = cv2.resize(mask, (640, 640), interpolation=cv2.INTER_NEAREST)
                full_gt_target_mask[mask_resized > 0] = True  # Mark as target in GT

        # Collect all target predictions
        for idx, ann in enumerate(annotations):
            if idx not in predictions:
                continue

            pred_mask = predictions[idx]
            x, y, w, h = ann['bbox']

            # Scale bbox to resized image
            x = x * 640 / orig_width
            y = y * 640 / orig_height
            w = w * 640 / orig_width
            h = h * 640 / orig_height

            roi = self.extract_roi_from_bbox(
                [x, y, x + w, y + h],
                (640, 640)
            )

            if roi is None:
                continue

            x1, y1, x2, y2 = roi

            # Resize prediction to ROI size
            pred_resized = cv2.resize(
                pred_mask.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_NEAREST
            )

            # Mark target predictions
            full_pred_target_mask[y1:y2, x1:x2] |= (pred_resized == 1)

        # Find where GT is target but prediction is not (missed targets)
        missed_target_mask = full_gt_target_mask & (~full_pred_target_mask)

        # Find where prediction is target but GT is background (false positives)
        false_positive_mask = full_pred_target_mask & (~full_gt_target_mask)

        # Combine both error types (missed targets and false positives)
        error_mask = missed_target_mask | false_positive_mask

        # Render all error areas in red with slight transparency
        pred_overlay[error_mask] = [1.0, 0.0, 0.0, 0.55]  # Red with slight transparency (alpha = 0.55)

        ax.imshow(pred_overlay)