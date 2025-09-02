#!/usr/bin/env python3
"""Test ONNX instance segmentation model with visualization."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional
import onnxruntime as ort
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch
import torch.nn.functional as F


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test ONNX instance segmentation model')
    parser.add_argument(
        '--onnx', '-o',
        type=str,
        required=True,
        help='Path to ONNX model file'
    )
    parser.add_argument(
        '--annotations', '-a',
        type=str,
        required=True,
        help='Path to COCO annotation file'
    )
    parser.add_argument(
        '--images_dir', '-i',
        type=str,
        default='data/images/val2017',
        help='Path to images directory (default: data/images/val2017)'
    )
    parser.add_argument(
        '--num_images', '-n',
        type=int,
        default=5,
        help='Number of images to process (default: 5)'
    )
    parser.add_argument(
        '--output_dir', '-d',
        type=str,
        default='test_outputs',
        help='Output directory for visualizations (default: test_outputs)'
    )
    parser.add_argument(
        '--provider', '-p',
        type=str,
        choices=['cpu', 'cuda'],
        default='cuda',
        help='Execution provider (default: cuda)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Alpha value for mask overlay (default: 0.5)'
    )
    parser.add_argument(
        '--score_threshold',
        type=float,
        default=0.01,
        help='Score threshold for mask visualization (default: 0.01)'
    )
    parser.add_argument(
        '--save_masks',
        action='store_true',
        help='Save individual mask predictions'
    )
    parser.add_argument(
        '--binary_mode',
        action='store_true',
        help='Use binary_masks output for green mask overlay instead of instance segmentation'
    )
    parser.add_argument(
        '--draw_rois',
        action='store_true',
        help='Debug: always draw all GT ROIs as rectangles'
    )
    return parser.parse_args()


def generate_instance_colors(num_instances: int) -> np.ndarray:
    """Generate distinct colors for each instance."""
    # Use HSV color space for better color distribution
    hues = np.linspace(0, 1, num_instances + 1)[:-1]
    colors = []
    for hue in hues:
        # Convert HSV to RGB
        rgb = plt.cm.hsv(hue)[:3]
        colors.append([int(c * 255) for c in rgb])
    return np.array(colors)


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """Normalize bounding box coordinates to 0-1 range.

    Args:
        bbox: [x, y, width, height] in COCO format
        img_width: Image width
        img_height: Image height

    Returns:
        Normalized [x1, y1, x2, y2] coordinates
    """
    x, y, w, h = bbox
    x1 = x / img_width
    y1 = y / img_height
    x2 = (x + w) / img_width
    y2 = (y + h) / img_height

    # Clip to [0, 1]
    x1 = max(0, min(1, x1))
    y1 = max(0, min(1, y1))
    x2 = max(0, min(1, x2))
    y2 = max(0, min(1, y2))

    return [x1, y1, x2, y2]


def denormalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[int]:
    """Convert normalized bbox back to pixel coordinates.

    Args:
        bbox: Normalized [x1, y1, x2, y2]
        img_width: Image width
        img_height: Image height

    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    x1, y1, x2, y2 = bbox
    return [
        int(x1 * img_width),
        int(y1 * img_height),
        int(x2 * img_width),
        int(y2 * img_height)
    ]


def scale_bbox(bbox: List[float], sx: float, sy: float) -> List[float]:
    """Scale COCO-format bbox [x, y, w, h] by factors (sx, sy)."""
    x, y, w, h = bbox
    return [x * sx, y * sy, w * sx, h * sy]


def prepare_image(image_path: str, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """Load and preprocess image for model input.

    Args:
        image_path: Path to image file
        target_size: Target size (width, height)

    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize to target size
    image_resized = cv2.resize(image, target_size)

    # Convert to float32 and normalize to [0, 1] to match validation pipeline
    image_float = image_resized.astype(np.float32) / 255.0

    # Transpose to CHW format and add batch dimension
    image_chw = np.transpose(image_float, (2, 0, 1))
    image_batch = np.expand_dims(image_chw, axis=0)

    return image_batch


def _to_int(v) -> Optional[int]:
    """Convert ONNX dim value to int if possible, else None."""
    try:
        if isinstance(v, (int, np.integer)):
            return int(v)
        # Some ONNX shapes may be strings/symbolic; return None
        return None
    except Exception:
        return None


def get_images_input_size(session: ort.InferenceSession, input_name: str = 'images') -> Optional[Tuple[int, int]]:
    """Return (width, height) for the specified input if statically known.

    Looks up the input named `images` (by default) and extracts width/height from
    the 4D shape [N, C, H, W]. Returns None if dynamic/unknown.
    """
    try:
        for inp in session.get_inputs():
            if inp.name == input_name:
                shape = list(inp.shape)
                if len(shape) == 4:
                    h = _to_int(shape[2])
                    w = _to_int(shape[3])
                    if h is not None and w is not None:
                        return (w, h)
        return None
    except Exception:
        return None


def process_mask_output(masks: np.ndarray, rois: np.ndarray,
                        img_width: int, img_height: int,
                        score_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """Process mask output from model.

    Args:
        masks: Model output masks [N, 3, H, W]
        rois: ROI coordinates [N, 5] with batch_idx and normalized coords
        img_width: Original image width
        img_height: Original image height
        score_threshold: Threshold for mask scores

    Returns:
        List of processed mask dictionaries
    """
    results = []
    num_rois = masks.shape[0]

    for i in range(num_rois):
        # Get mask logits for this ROI
        mask_logits = masks[i]  # [3, H, W]

        # Apply softmax to get probabilities
        mask_probs = torch.from_numpy(mask_logits).float()
        mask_probs = F.softmax(mask_probs, dim=0).numpy()

        # Get predicted class (0=background, 1=target, 2=non-target)
        mask_pred = np.argmax(mask_probs, axis=0)  # [H, W]

        # Get confidence scores
        mask_scores = np.max(mask_probs, axis=0)  # [H, W]

        # Create binary mask for target class (class 1)
        target_mask = (mask_pred == 1) & (mask_scores > score_threshold)

        # Get ROI coordinates in pixel space
        roi_coords = denormalize_bbox(rois[i, 1:5], img_width, img_height)

        # Resize mask to ROI size
        roi_width = roi_coords[2] - roi_coords[0]
        roi_height = roi_coords[3] - roi_coords[1]

        if roi_width > 0 and roi_height > 0:
            # Resize mask to ROI dimensions
            mask_resized = cv2.resize(
                target_mask.astype(np.uint8),
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST
            )

            # Get average confidence for the instance
            avg_score = np.mean(mask_scores[target_mask]) if np.any(target_mask) else 0

            results.append({
                'mask': mask_resized,
                'bbox': roi_coords,
                'score': avg_score,
                'class_probs': mask_probs,
                'raw_mask': mask_pred
            })

    return results


def visualize_binary_mask(image: np.ndarray,
                         binary_mask: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Visualize binary mask as green overlay on image.
    
    Args:
        image: Original image [H, W, 3]
        binary_mask: Binary mask [1, H, W] with values in [0, 1]
        alpha: Alpha value for overlay
        
    Returns:
        Image with green mask overlay
    """
    output_image = image.copy()
    h, w = image.shape[:2]
    
    # Ensure binary_mask is 2D
    if binary_mask.ndim == 3:
        binary_mask = binary_mask.squeeze(0)  # Remove channel dimension
    
    # Resize binary mask to match image size if needed
    if binary_mask.shape != (h, w):
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Debug: check mask value range (optional - can be commented out)
    # print(f"Binary mask stats - min: {binary_mask.min():.3f}, max: {binary_mask.max():.3f}, mean: {binary_mask.mean():.3f}")
    
    # Create green overlay
    overlay = np.zeros_like(image)
    overlay[:, :, 1] = 255  # Green channel
    
    # Apply mask with alpha blending
    # Use the binary mask as alpha channel
    mask_3d = np.stack([binary_mask] * 3, axis=-1)
    
    # Blend: output = image * (1 - alpha * mask) + overlay * (alpha * mask)
    output_image = image * (1 - alpha * mask_3d) + overlay * (alpha * mask_3d)
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    
    return output_image


def visualize_instance_segmentation(image: np.ndarray,
                                   mask_results: List[Dict[str, Any]],
                                   alpha: float = 0.5) -> np.ndarray:
    """Visualize instance segmentation results on image.

    Args:
        image: Original image
        mask_results: List of mask result dictionaries
        alpha: Alpha value for overlay

    Returns:
        Image with overlaid masks
    """
    output_image = image.copy()
    h, w = image.shape[:2]

    # Create overlay for all masks
    overlay = np.zeros_like(image)

    # Generate colors for instances
    num_instances = len(mask_results)
    if num_instances > 0:
        colors = generate_instance_colors(num_instances)

        for idx, result in enumerate(mask_results):
            mask = result['mask']
            bbox = result['bbox']
            score = result['score']

            if score > 0:  # Only visualize if there's a valid score
                # Create full-size mask
                full_mask = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = bbox

                # Place ROI mask in correct position
                if x2 > x1 and y2 > y1:
                    full_mask[y1:y2, x1:x2] = mask

                # Apply color to mask region
                mask_colored = np.zeros_like(overlay)
                mask_colored[full_mask > 0] = colors[idx]

                # Add to overlay
                overlay = np.where(
                    np.stack([full_mask] * 3, axis=-1) > 0,
                    mask_colored,
                    overlay
                )

                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), colors[idx].tolist(), 2)

                # Add score text
                label = f"Instance {idx+1}: {score:.2f}"
                cv2.putText(
                    output_image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx].tolist(), 2
                )

    # Alpha blend overlay with original image
    output_image = cv2.addWeighted(output_image, 1 - alpha, overlay, alpha, 0)

    return output_image


def save_individual_masks(mask_results: List[Dict[str, Any]],
                         output_path: str,
                         img_width: int,
                         img_height: int):
    """Save individual mask predictions for debugging.

    Args:
        mask_results: List of mask result dictionaries
        output_path: Base output path
        img_width: Image width
        img_height: Image height
    """
    base_name = Path(output_path).stem
    output_dir = Path(output_path).parent / f"{base_name}_masks"
    output_dir.mkdir(exist_ok=True)

    for idx, result in enumerate(mask_results):
        # Save raw mask prediction (3 classes)
        raw_mask = result['raw_mask']  # [H, W]
        class_probs = result['class_probs']  # [3, H, W]

        # Create visualization for each class
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        # Class predictions
        axes[0].imshow(raw_mask, cmap='tab10', vmin=0, vmax=2)
        axes[0].set_title(f'Instance {idx+1}: Predicted Classes')
        axes[0].axis('off')

        # Background probability
        axes[1].imshow(class_probs[0], cmap='viridis', vmin=0, vmax=1)
        axes[1].set_title('Background Prob')
        axes[1].axis('off')

        # Target probability
        axes[2].imshow(class_probs[1], cmap='viridis', vmin=0, vmax=1)
        axes[2].set_title('Target Prob')
        axes[2].axis('off')

        # Non-target probability
        axes[3].imshow(class_probs[2], cmap='viridis', vmin=0, vmax=1)
        axes[3].set_title('Non-Target Prob')
        axes[3].axis('off')

        plt.tight_layout()
        plt.savefig(output_dir / f"instance_{idx+1}_probs.png")
        plt.close()

        # Save binary mask
        mask = result['mask']
        cv2.imwrite(str(output_dir / f"instance_{idx+1}_mask.png"), mask * 255)


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model: {args.onnx}")

    # Set up ONNX Runtime
    providers = ['CUDAExecutionProvider'] if args.provider == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx, providers=providers)

    # Get model input/output info
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")
    
    # Determine model input spatial size for `images`
    target_size = get_images_input_size(session, input_name='images')
    if target_size is not None:
        print(f"Detected model input size for 'images': {target_size[0]}x{target_size[1]}")
    else:
        target_size = (640, 640)
        print(f"Could not determine static input size; defaulting to {target_size[0]}x{target_size[1]}")
    
    # Check if binary_mode is compatible with model
    if args.binary_mode:
        if len(output_names) < 2 or 'binary_masks' not in output_names[1]:
            print("Warning: Binary mode requested but model doesn't have binary_masks output.")
            print("         Falling back to instance segmentation mode.")
            args.binary_mode = False
        else:
            print("Binary mode enabled - will use binary_masks output for green overlay")

    # Load COCO annotations
    print(f"Loading annotations: {args.annotations}")
    coco = COCO(args.annotations)

    # Get person category ID
    cat_ids = coco.getCatIds(catNms=['person'])
    if not cat_ids:
        print("Error: No 'person' category found in annotations")
        return 1

    # Get images with person annotations
    img_ids = coco.getImgIds(catIds=cat_ids)
    if len(img_ids) == 0:
        print("No images with person annotations found")
        return 1

    # Limit number of images
    img_ids = img_ids[:args.num_images]
    print(f"Processing {len(img_ids)} images")

    # Process each image
    for img_id in tqdm(img_ids, desc="Processing images"):
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        img_path = Path(args.images_dir) / img_info['file_name']

        if not img_path.exists():
            print(f"Warning: Image not found: {img_path}")
            continue

        # Load annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)

        if len(anns) == 0:
            print(f"No person annotations for image {img_info['file_name']}")
            continue

        # Load original image for visualization and size reference
        image_orig = cv2.imread(str(img_path))
        image_orig = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
        img_height, img_width = image_orig.shape[:2]

        # Load and preprocess image to model input resolution
        image_input = prepare_image(str(img_path), target_size=target_size)

        # Prepare ROIs from annotations
        rois_list = []
        # Normalize bboxes using ORIGINAL dimensions in annotations, not the resized image size.
        # This avoids degenerate boxes when images_dir contains resized images (e.g. 160x120).
        ann_img_width = int(img_info.get('width', img_width))
        ann_img_height = int(img_info.get('height', img_height))
        for ann in anns:
            if 'bbox' in ann:
                norm_bbox = normalize_bbox(ann['bbox'], ann_img_width, ann_img_height)
                # Add batch index (0 for single image)
                roi = [0] + norm_bbox
                rois_list.append(roi)

        if len(rois_list) == 0:
            print(f"No valid bboxes for image {img_info['file_name']}")
            continue

        # Convert ROIs to numpy array
        rois = np.array(rois_list, dtype=np.float32)

        # Run inference
        outputs = session.run(output_names, {
            'images': image_input,
            'rois': rois, #rois_scaled
        })

        # Check if binary_mode is enabled and binary_masks output exists
        if args.binary_mode and len(outputs) > 1:
            # Use binary_masks output (second output)
            binary_masks = outputs[1]  # [B, 1, H, W]
            
            # Get the first batch's binary mask
            binary_mask = binary_masks[0]  # [1, H, W]
            
            # Visualize binary mask
            output_image = visualize_binary_mask(
                image_orig, binary_mask, alpha=args.alpha
            )
        else:
            # Process mask outputs for instance segmentation
            masks = outputs[0]  # [N, 3, H, W]

            # Process masks
            mask_results = process_mask_output(
                masks, rois, img_width, img_height,
                score_threshold=args.score_threshold
            )

            # Visualize results
            output_image = visualize_instance_segmentation(
                image_orig, mask_results, alpha=args.alpha
            )

        # Optionally overlay GT ROIs for debugging
        if args.draw_rois:
            # Draw all ROIs on top of current output_image
            dbg_img = output_image.copy()
            for roi in rois:
                x1, y1, x2, y2 = denormalize_bbox(roi[1:5].tolist(), img_width, img_height)
                cv2.rectangle(dbg_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            output_image = dbg_img

        # Save output image
        output_suffix = "_binary" if args.binary_mode else "_segmented"
        output_path = output_dir / f"{Path(img_info['file_name']).stem}{output_suffix}.png"
        cv2.imwrite(str(output_path), cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

        # Save individual masks if requested (only for instance segmentation mode)
        if args.save_masks and not args.binary_mode:
            save_individual_masks(mask_results, str(output_path), img_width, img_height)
            print(f"  Saved individual masks to {output_path.stem}_masks/")

    print(f"\n✅ Processing complete! Results saved to {output_dir}")
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
