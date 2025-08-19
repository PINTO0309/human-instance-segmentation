"""Training script for UNet decoder-only knowledge distillation."""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import warnings
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._trace")
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean might cause the trace to be incorrect")

# Import base components
from src.human_edge_detection.experiments.config_manager import (
    ExperimentConfig, ConfigManager, create_experiment_dirs
)
from src.human_edge_detection.text_logger import TextLogger
from src.human_edge_detection.advanced.unet_decoder_distillation import (
    DistillationUNetWrapper, UNetDistillationLoss, create_unet_distillation_model
)

# Setup logger
logger = logging.getLogger(__name__)


def create_dataloader(config: ExperimentConfig, is_training: bool = True, use_heavy_augmentation: bool = True) -> DataLoader:
    """Create dataloader for full image processing (no ROI).

    Args:
        config: Experiment configuration
        is_training: Whether this is for training (True) or validation (False)
        use_heavy_augmentation: Whether to use heavy augmentations for training
    """
    from torchvision import transforms
    from torchvision.datasets import CocoDetection
    import numpy as np
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    class COCOPersonSegmentation(CocoDetection):
        """COCO dataset for person segmentation."""

        def __init__(self, root, annFile, transform=None):
            super().__init__(root, annFile)
            self.transform = transform

            # Filter for person-only images
            person_cat_id = None
            for cat in self.coco.cats.values():
                if cat['name'] == 'person':
                    person_cat_id = cat['id']
                    break

            # Keep only images with person annotations
            valid_img_ids = set()
            for ann in self.coco.anns.values():
                if ann['category_id'] == person_cat_id and not ann.get('iscrowd', 0):
                    valid_img_ids.add(ann['image_id'])

            self.ids = list(valid_img_ids)

        def __getitem__(self, index):
            """Get image and binary mask for person segmentation."""
            img, anns = super().__getitem__(index)

            # Create binary mask for all person instances
            img_info = self.coco.imgs[self.ids[index]]
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

            for ann in anns:
                if not ann.get('iscrowd', 0):
                    # Convert segmentation to binary mask
                    from pycocotools import mask as mask_utils
                    if 'segmentation' in ann and ann['segmentation']:
                        if isinstance(ann['segmentation'], list):
                            # Polygon format
                            rle = mask_utils.frPyObjects(ann['segmentation'],
                                                         img_info['height'],
                                                         img_info['width'])
                            if isinstance(rle, list):
                                rle = mask_utils.merge(rle)
                            binary_mask = mask_utils.decode(rle)
                        else:
                            # RLE format
                            binary_mask = mask_utils.decode(ann['segmentation'])
                        mask = np.maximum(mask, binary_mask)

            # Convert mask to uint8 for Albumentations
            mask = mask.astype(np.uint8)

            if self.transform:
                # Check if it's an Albumentations transform
                if hasattr(self.transform, '__module__') and 'albumentations' in self.transform.__module__:
                    # Convert PIL to numpy for Albumentations
                    img = np.array(img)
                    # Apply augmentations to both image and mask
                    transformed = self.transform(image=img, mask=mask)
                    img = transformed['image']
                    mask = transformed['mask']
                    # Ensure mask is binary and has channel dimension
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    mask = torch.unsqueeze(mask, 0) if len(mask.shape) == 2 else mask
                    mask = (mask > 0.5).float()
                else:
                    # Fall back to torchvision transforms
                    mask = Image.fromarray(mask * 255)
                    img = self.transform(img)
                    # Apply only resize and ToTensor to mask (no normalization)
                    mask_transform = transforms.Compose([
                        transforms.Resize((640, 640)),
                        transforms.ToTensor()
                    ])
                    mask = mask_transform(mask)
                    mask = (mask > 0.5).float()  # Binarize after transform

            return img, mask

    # Create transforms based on training/validation mode
    if is_training and use_heavy_augmentation:
        # Heavy augmentation pipeline similar to get_roi_safe_heavy_transforms
        transform = A.Compose([
            # Resize first
            A.Resize(640, 640),

            # Horizontal flip
            A.HorizontalFlip(p=0.5),

            # Color augmentations
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=1.0
                ),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
            ], p=0.8),

            # Lighting conditions
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.5),

            # Weather/environmental effects
            A.OneOf([
                A.RandomRain(drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                           blur_value=5, brightness_coefficient=0.7, rain_type='drizzle', p=1.0),
                A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=1.0),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_range=(0, 1),
                    num_flare_circles_range=(3, 6),
                    src_radius=100,
                    src_color=(255, 255, 255),
                    p=1.0
                ),
            ], p=0.1),

            # Blur effects
            A.OneOf([
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ], p=0.05),

            # Noise
            A.OneOf([
                A.GaussNoise(noise_scale_factor=1.0, p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.05),

            # Image quality degradation
            A.OneOf([
                A.ImageCompression(quality_range=(70, 95), p=1.0),
                A.Downscale(scale_range=(0.5, 0.9), p=1.0),
            ], p=0.1),

            # ImageNet normalization
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    elif is_training:
        # Light augmentation for training (if heavy is disabled)
        transform = A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
            ], p=0.8),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    else:
        # Validation transforms (no augmentation)
        transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    # Create dataset
    if is_training:
        ann_file = config.data.train_annotation
        img_dir = "data/images/train2017"
    else:
        ann_file = config.data.val_annotation
        img_dir = "data/images/val2017"

    dataset = COCOPersonSegmentation(
        root=img_dir,
        annFile=ann_file,
        transform=transform
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=is_training,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=is_training
    )

    return dataloader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    model.student.train()  # Ensure student is in training mode
    if model.teacher is not None:
        model.teacher.eval()   # Teacher always in eval mode (if exists)

    total_loss = 0
    kl_loss = 0
    mse_loss = 0
    bce_loss = 0
    dice_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}", dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(progress_bar):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                # Forward pass through both student and teacher
                student_output, teacher_output = model(images)

                # Compute loss
                loss, loss_dict = loss_fn(student_output, teacher_output, masks)

            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            # Unscale the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Clip gradients only for parameters being optimized (decoder only)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.student.get_decoder_parameters(), max_norm=1.0)

            # Step optimizer with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            student_output, teacher_output = model(images)
            loss, loss_dict = loss_fn(student_output, teacher_output, masks)
            loss.backward()

            # Gradient clipping for stability (only decoder parameters)
            torch.nn.utils.clip_grad_norm_(model.student.get_decoder_parameters(), max_norm=1.0)

            optimizer.step()

        # Update metrics
        total_loss += loss.item()
        kl_loss += loss_dict.get('kl_loss', 0)
        mse_loss += loss_dict.get('mse_loss', 0)
        bce_loss += loss_dict.get('bce_loss', 0)
        dice_loss += loss_dict.get('dice_loss', 0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'kl': f"{loss_dict.get('kl_loss', 0):.4f}",
            'mse': f"{loss_dict.get('mse_loss', 0):.4f}",
            'dice': f"{loss_dict.get('dice_loss', 0):.4f}"
        })

        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/KL_Loss', loss_dict.get('kl_loss', 0), global_step)
            writer.add_scalar('Train/MSE_Loss', loss_dict.get('mse_loss', 0), global_step)
            writer.add_scalar('Train/BCE_Loss', loss_dict.get('bce_loss', 0), global_step)
            writer.add_scalar('Train/Dice_Loss', loss_dict.get('dice_loss', 0), global_step)

    # Average metrics
    num_batches = len(dataloader)
    metrics = {
        'total_loss': total_loss / num_batches,
        'kl_loss': kl_loss / num_batches,
        'mse_loss': mse_loss / num_batches,
        'bce_loss': bce_loss / num_batches,
        'dice_loss': dice_loss / num_batches,
    }

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    visualize_dir: Optional[Path] = None,
    teacher_miou_cache: Optional[float] = None,
    student_name: str = 'Student',
    teacher_name: str = 'Teacher'
) -> Dict[str, float]:
    """Evaluate model.

    Args:
        model: Model to evaluate
        dataloader: Validation dataloader
        loss_fn: Loss function
        device: Device to use
        epoch: Current epoch
        writer: Tensorboard writer
        visualize_dir: Directory for saving visualizations
        teacher_miou_cache: Cached teacher mIoU value (computed once on first eval)
    """
    model.eval()

    total_loss = 0
    kl_loss = 0
    mse_loss = 0
    bce_loss = 0
    dice_loss = 0
    total_student_iou = 0
    total_teacher_iou = 0
    total_agreement = 0

    # Use cached teacher mIoU if available
    use_cached_teacher_miou = teacher_miou_cache is not None
    if use_cached_teacher_miou:
        print(f"  Using cached Teacher {teacher_name} mIoU: {teacher_miou_cache:.4f}")

    # For visualization - get selected test images first
    test_images = None
    test_masks = None
    student_output_test = None
    teacher_output_test = None

    if visualize_dir is not None:
        # Get specific test images with 1, 2, 3, 5 people
        test_images, test_masks = get_test_images_by_person_count(dataloader, device)

    with torch.no_grad():
        # Get predictions for test images (we'll save visualization after computing mIoU)
        if test_images is not None:
            student_output_test, teacher_output_test = model(test_images)

        # Then run normal validation
        progress_bar = tqdm(dataloader, desc="Validation", dynamic_ncols=True)
        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            student_output, teacher_output = model(images)

            # Compute loss
            loss, loss_dict = loss_fn(student_output, teacher_output, masks)

            # Update metrics
            total_loss += loss.item()
            kl_loss += loss_dict.get('kl_loss', 0)
            mse_loss += loss_dict.get('mse_loss', 0)
            bce_loss += loss_dict.get('bce_loss', 0)
            dice_loss += loss_dict.get('dice_loss', 0)  # This was missing!

            # Compute mIoU for both student and teacher predictions
            student_preds = (torch.sigmoid(student_output) > 0.5).float()
            if model.teacher is not None:
                teacher_preds = (torch.sigmoid(teacher_output) > 0.5).float()
            else:
                teacher_preds = student_preds  # Use student predictions for compatibility
            masks_binary = (masks > 0.5).float()

            # Student mIoU (per class: background and foreground)
            # Class 0: Background
            student_bg_tp = ((1 - student_preds) * (1 - masks_binary)).sum(dim=(1, 2, 3))
            student_bg_fp = ((1 - student_preds) * masks_binary).sum(dim=(1, 2, 3))
            student_bg_fn = (student_preds * (1 - masks_binary)).sum(dim=(1, 2, 3))
            student_bg_iou = student_bg_tp / (student_bg_tp + student_bg_fp + student_bg_fn + 1e-6)

            # Class 1: Foreground
            student_fg_tp = (student_preds * masks_binary).sum(dim=(1, 2, 3))
            student_fg_fp = (student_preds * (1 - masks_binary)).sum(dim=(1, 2, 3))
            student_fg_fn = ((1 - student_preds) * masks_binary).sum(dim=(1, 2, 3))
            student_fg_iou = student_fg_tp / (student_fg_tp + student_fg_fp + student_fg_fn + 1e-6)

            # Student mIoU
            student_miou = (student_bg_iou + student_fg_iou) / 2.0
            total_student_iou += student_miou.mean().item()

            # Only calculate Teacher mIoU if not cached and teacher exists
            if not use_cached_teacher_miou and model.teacher is not None:
                # Teacher mIoU (per class: background and foreground)
                # Class 0: Background
                teacher_bg_tp = ((1 - teacher_preds) * (1 - masks_binary)).sum(dim=(1, 2, 3))
                teacher_bg_fp = ((1 - teacher_preds) * masks_binary).sum(dim=(1, 2, 3))
                teacher_bg_fn = (teacher_preds * (1 - masks_binary)).sum(dim=(1, 2, 3))
                teacher_bg_iou = teacher_bg_tp / (teacher_bg_tp + teacher_bg_fp + teacher_bg_fn + 1e-6)

                # Class 1: Foreground
                teacher_fg_tp = (teacher_preds * masks_binary).sum(dim=(1, 2, 3))
                teacher_fg_fp = (teacher_preds * (1 - masks_binary)).sum(dim=(1, 2, 3))
                teacher_fg_fn = ((1 - teacher_preds) * masks_binary).sum(dim=(1, 2, 3))
                teacher_fg_iou = teacher_fg_tp / (teacher_fg_tp + teacher_fg_fp + teacher_fg_fn + 1e-6)

                # Teacher mIoU
                teacher_miou = (teacher_bg_iou + teacher_fg_iou) / 2.0
                total_teacher_iou += teacher_miou.mean().item()
            elif not use_cached_teacher_miou and model.teacher is None:
                # No teacher - use student metrics
                total_teacher_iou += student_miou.mean().item()

            # Teacher-Student Agreement (pixel-wise accuracy)
            agreement = (student_preds == teacher_preds).float().mean(dim=(1, 2, 3))
            total_agreement += agreement.mean().item()

            # Update progress bar with current batch metrics
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                f'mIoU_{student_name}': f"{student_miou.mean().item():.4f}",
                'agree': f"{agreement.mean().item():.4f}"
            })

    # Average metrics
    num_batches = len(dataloader)

    # Use cached teacher mIoU if available, otherwise use calculated value
    if use_cached_teacher_miou:
        final_teacher_miou = teacher_miou_cache
    elif model.teacher is None:
        # No teacher model - use student mIoU
        final_teacher_miou = total_student_iou / num_batches
    else:
        final_teacher_miou = total_teacher_iou / num_batches

    metrics = {
        'total_loss': total_loss / num_batches,
        'kl_loss': kl_loss / num_batches,
        'mse_loss': mse_loss / num_batches,
        'bce_loss': bce_loss / num_batches,
        'dice_loss': dice_loss / num_batches,
        'student_miou': total_student_iou / num_batches,
        'teacher_miou': final_teacher_miou,
        'agreement': total_agreement / num_batches,
        'miou_diff': (total_student_iou / num_batches) - final_teacher_miou,
        # Keep 'iou' for backward compatibility (using student mIoU)
        'iou': total_student_iou / num_batches,
    }

    # Print validation summary
    print(f"\n[Validation Summary]")
    print(f"  Total Loss: {metrics['total_loss']:.4f}")
    print(f"  KL Loss: {metrics['kl_loss']:.4f}")
    # Check if distillation is enabled for appropriate output
    if model.teacher is not None:
        # Distillation mode - show full metrics
        print(f"  MSE Loss: {metrics['mse_loss']:.4f}")
        print(f"  BCE Loss: {metrics['bce_loss']:.4f}")
        print(f"  Dice Loss: {metrics['dice_loss']:.4f}")
        print(f"  Student {student_name} mIoU: {metrics['student_miou']:.4f}")
        print(f"  Teacher {teacher_name} mIoU: {metrics['teacher_miou']:.4f}")
        print(f"  {teacher_name}-{student_name} Agreement: {metrics['agreement']:.4f} ({metrics['agreement']*100:.2f}%)")
        print(f"  mIoU Difference ({student_name}-{teacher_name}): {metrics['miou_diff']:.4f}")
    else:
        # Fine-tuning mode - simplified metrics
        print(f"  BCE Loss: {metrics['bce_loss']:.4f}")
        print(f"  Dice Loss: {metrics['dice_loss']:.4f}")
        print(f"  Model mIoU: {metrics['student_miou']:.4f}")

    # Save visualization with mIoU values
    if visualize_dir is not None and test_images is not None:
        # Check if distillation is enabled (teacher model exists)
        if model.teacher is not None:
            # Normal visualization with teacher
            save_visualization(
                test_images, test_masks, student_output_test, teacher_output_test,
                visualize_dir, epoch, 0, num_samples=4,
                student_miou=metrics['student_miou'],
                teacher_miou=metrics['teacher_miou'],
                student_name=student_name,
                teacher_name=teacher_name
            )
        else:
            # Pure fine-tuning: save simplified visualization without teacher
            save_finetuning_visualization(
                test_images, test_masks, student_output_test,
                visualize_dir, epoch, 0, num_samples=4,
                model_miou=metrics['student_miou'],
                model_name='Model'
            )

    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', metrics['total_loss'], epoch)
        writer.add_scalar('Val/KL_Loss', metrics['kl_loss'], epoch)
        writer.add_scalar('Val/MSE_Loss', metrics['mse_loss'], epoch)
        writer.add_scalar('Val/BCE_Loss', metrics['bce_loss'], epoch)
        writer.add_scalar('Val/Dice_Loss', metrics['dice_loss'], epoch)
        writer.add_scalar('Val/Student_mIoU', metrics['student_miou'], epoch)
        writer.add_scalar('Val/Teacher_mIoU', metrics['teacher_miou'], epoch)
        writer.add_scalar('Val/Agreement', metrics['agreement'], epoch)
        writer.add_scalar('Val/mIoU_Diff', metrics['miou_diff'], epoch)

    return metrics


def get_test_images_by_person_count(dataloader, device='cuda'):
    """Get specific test images with 1, 2, 3, and 5 people.

    Args:
        dataloader: Validation dataloader
        device: Device to use

    Returns:
        Tuple of (images, masks) tensors containing selected samples
    """
    from pycocotools import mask as mask_utils
    import numpy as np

    # Target person counts we want to find (in order)
    target_counts = [1, 2, 3, 5]
    selected_samples = {count: [] for count in target_counts}

    # Get the underlying dataset to access COCO annotations
    dataset = dataloader.dataset

    # Load full val2017 annotations
    from pycocotools.coco import COCO
    import sys
    import io

    # Suppress COCO loading messages
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        coco_full = COCO('data/annotations/instances_val2017_person_only_no_crowd.json')
    finally:
        sys.stdout = old_stdout

    # Use pre-selected images with large ROIs
    # These image IDs were selected based on having large ROI areas
    # Alternative sets available - uncomment to use different images:

    # Set 1: Largest ROIs available
    large_roi_img_ids = {
        1: 500716,  # 83.5% ROI coverage - 000000500716.jpg
        2: 468954,  # 33.2% ROI coverage - 000000468954.jpg
        3: 23899,   # 20.1% ROI coverage - 000000023899.jpg
        5: 162732   # Best 5-person image - all ROIs > 16%, avg 22.7% - 000000162732.jpg
    }

    # Alternative Set 2: Second largest ROIs (uncomment to use)
    # large_roi_img_ids = {
    #     1: 167572,  # Large ROI - single person portrait
    #     2: 324158,  # Large ROI - two people close-up
    #     3: 485802,  # Large ROI - three people medium shot
    #     5: 579635   # Large ROI - five people group
    # }

    # No need to print - using pre-selected images with large ROIs

    # Search through the dataset to find these specific images
    found_images = {count: False for count in target_counts}

    for batch_idx, (images, masks) in enumerate(dataloader):
        batch_size = images.shape[0]

        for i in range(batch_size):
            # Get the actual image ID from dataset
            actual_idx = batch_idx * dataloader.batch_size + i
            if actual_idx >= len(dataset):
                break

            img_id = dataset.ids[actual_idx]

            # Check if this is one of our target images
            for count, target_img_id in large_roi_img_ids.items():
                if img_id == target_img_id and not found_images[count]:
                    selected_samples[count] = [(
                        images[i:i+1].clone().to(device),
                        masks[i:i+1].clone().to(device),
                        img_id
                    )]
                    found_images[count] = True
                    pass  # Found image in dataloader
                    break

        # Stop after checking many batches
        if batch_idx > 100:
            break

    # For images not found in dataloader, load them directly
    from PIL import Image
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    for count, target_img_id in large_roi_img_ids.items():
        if not found_images[count] and target_img_id in coco_full.imgs:
            # Load image directly from file

            # Load image
            img_info = coco_full.imgs[target_img_id]
            img_path = f"data/images/val2017/{img_info['file_name']}"
            img = Image.open(img_path).convert('RGB')

            # Create mask
            ann_ids = coco_full.getAnnIds(imgIds=target_img_id, iscrowd=None)
            anns = coco_full.loadAnns(ann_ids)
            mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)

            for ann in anns:
                if not ann.get('iscrowd', 0) and 'segmentation' in ann and ann['segmentation']:
                    if isinstance(ann['segmentation'], list):
                        # Polygon format
                        from pycocotools import mask as mask_utils
                        rle = mask_utils.frPyObjects(ann['segmentation'],
                                                     img_info['height'],
                                                     img_info['width'])
                        if isinstance(rle, list):
                            rle = mask_utils.merge(rle)
                        binary_mask = mask_utils.decode(rle)
                    else:
                        # RLE format
                        from pycocotools import mask as mask_utils
                        binary_mask = mask_utils.decode(ann['segmentation'])
                    mask = np.maximum(mask, binary_mask)

            # Apply transforms
            img_np = np.array(img)

            # Create transform for preprocessing
            # Use ImageNet normalization to match training
            transform = A.Compose([
                A.Resize(640, 640),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])

            transformed = transform(image=img_np, mask=mask)
            img_tensor = transformed['image'].unsqueeze(0).to(device)
            mask_tensor = transformed['mask'].unsqueeze(0).unsqueeze(0).float().to(device)
            mask_tensor = (mask_tensor > 0.5).float()

            selected_samples[count] = [(img_tensor, mask_tensor, target_img_id)]
            found_images[count] = True
            # Successfully loaded image

    # If we couldn't find exact counts, use whatever we have
    # Fill missing samples with first available image
    fallback_img, fallback_mask = next(iter(dataloader))
    fallback_img = fallback_img.to(device)
    fallback_mask = fallback_mask.to(device)

    result_images = []
    result_masks = []

    # Use the specific large ROI images we found
    for count in target_counts:
        if selected_samples[count]:
            # We should have exactly one image per count (the pre-selected one with large ROI)
            sample = selected_samples[count][0]
            result_images.append(sample[0])
            result_masks.append(sample[1])
            # Show img_id if available
            if len(sample) > 2:
                img_info = coco_full.imgs.get(sample[2], {})
                file_name = img_info.get('file_name', 'unknown')
                pass  # Using large ROI image
            else:
                pass  # Using image
        else:
            # Use fallback if the specific image wasn't found in the batch
            result_images.append(fallback_img[0:1])
            result_masks.append(fallback_mask[0:1])
            pass  # Using fallback image

    # Concatenate results
    images_tensor = torch.cat(result_images, dim=0)
    masks_tensor = torch.cat(result_masks, dim=0)

    return images_tensor, masks_tensor


def save_finetuning_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    model_output: torch.Tensor,
    visualize_dir: Path,
    epoch: int,
    batch_idx: int,  # Keep for API compatibility, though unused
    num_samples: int = 4,
    model_miou: Optional[float] = None,
    model_name: str = 'Model'
):
    """Save visualization for fine-tuning (no teacher model).
    
    Args:
        images: Input images (B, 3, H, W)
        masks: Ground truth masks (B, 1, H, W)
        model_output: Model predictions (B, 1, H, W)
        visualize_dir: Directory to save visualizations
        epoch: Current epoch
        batch_idx: Batch index
        num_samples: Number of samples to visualize
        model_miou: Model's mIoU score
        model_name: Name of the model
    """
    # Convert to numpy arrays
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    model_preds = torch.sigmoid(model_output).cpu().numpy()
    
    # Create figure with 3 columns: Original, Ground Truth, Prediction
    _, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    def create_overlay(image, mask, color=(0, 1, 0), alpha=0.3):
        """Create overlay of mask on image with specified color."""
        # Denormalize image from ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_denorm = image.transpose(1, 2, 0) * std + mean
        image_denorm = np.clip(image_denorm, 0, 1)
        
        # Create colored overlay
        overlay = image_denorm.copy()
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, 
                                       overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                                       overlay[:, :, c])
        return overlay
    
    # Count persons per image for display
    person_counts = []
    for i in range(min(num_samples, images.shape[0])):
        # Simple heuristic: count connected components
        mask_binary = (masks_np[i, 0] > 0.5).astype(np.uint8)
        # Use a simple approximation for person count
        person_count = 1 if mask_binary.sum() > 0 else 0
        # Could be improved with proper connected component analysis
        person_counts.append(person_count)
    
    # Plot each sample
    for i in range(min(num_samples, images.shape[0])):
        # Original image - Column 0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_denorm = images_np[i].transpose(1, 2, 0) * std + mean
        image_denorm = np.clip(image_denorm, 0, 1)
        axes[i, 0].imshow(image_denorm)
        axes[i, 0].set_title(f'Original Image\n{person_counts[i]} Person(s)')
        axes[i, 0].axis('off')
        
        # Ground truth overlay (green) - Column 1
        gt_overlay = create_overlay(images_np[i], masks_np[i, 0], color=(0, 1, 0), alpha=0.4)
        axes[i, 1].imshow(gt_overlay)
        axes[i, 1].set_title('Ground Truth (Green)')
        axes[i, 1].axis('off')
        
        # Model prediction overlay (blue) - Column 2
        model_overlay = create_overlay(images_np[i], model_preds[i, 0], color=(0, 0, 1), alpha=0.4)
        axes[i, 2].imshow(model_overlay)
        model_title = f'{model_name} (Blue)'
        if model_miou is not None:
            model_title += f'\nmIoU: {model_miou:.4f}'
        axes[i, 2].set_title(model_title)
        axes[i, 2].axis('off')
    
    plt.suptitle(f'Fine-tuning Results - Epoch {epoch+1}', fontsize=16)
    plt.tight_layout()
    
    # Save figure with consistent naming
    save_path = visualize_dir / f'epoch_{epoch+1:03d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"  Visualization saved to {save_path}")

def save_visualization(
    images: torch.Tensor,
    masks: torch.Tensor,
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    visualize_dir: Path,
    epoch: int,
    batch_idx: int,  # Keep for API compatibility, though unused
    num_samples: int = 4,
    student_miou: Optional[float] = None,
    teacher_miou: Optional[float] = None,
    student_name: str = 'Student',
    teacher_name: str = 'Teacher'
):
    """Save visualization of predictions with overlay.

    Args:
        images: Input images (B, 3, H, W)
        masks: Ground truth masks (B, 1, H, W)
        student_output: Student predictions (B, 1, H, W)
        teacher_output: Teacher predictions (B, 1, H, W)
        visualize_dir: Directory to save visualizations
        epoch: Current epoch
        batch_idx: Batch index
        num_samples: Number of samples to visualize (should be 4 for 1,2,3,5 people)
        student_miou: Student model's mIoU score
        teacher_miou: Teacher model's mIoU score
    """
    # Convert to numpy arrays
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    student_preds = torch.sigmoid(student_output).cpu().numpy()
    teacher_preds = torch.sigmoid(teacher_output).cpu().numpy()

    # Limit number of samples
    num_samples = min(num_samples, images.shape[0])

    # Create figure (4 columns: GT, Teacher, Student, Comparison)
    _, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    def create_overlay(image, mask, color=(0, 1, 0), alpha=0.3):
        """Create overlay of mask on image with specified color."""
        # Denormalize image from ImageNet normalization
        img = np.transpose(image, (1, 2, 0))
        # Denormalize: x = (x_norm * std) + mean
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std) + mean
        img = np.clip(img, 0, 1)  # Clip to valid range

        # Create colored overlay
        overlay = img.copy()
        mask_bool = mask > 0.5

        # Apply green color to mask area
        for c in range(3):
            overlay[:, :, c] = img[:, :, c] * (1 - alpha * mask_bool) + \
                              color[c] * alpha * mask_bool

        return overlay

    # Person count labels for the 4 selected images
    person_counts = [1, 2, 3, 5]

    for i in range(num_samples):
        # Denormalize image for overlays
        img = np.transpose(images_np[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img * std) + mean
        img = np.clip(img, 0, 1)

        # Ground truth overlay (green) - Column 0
        gt_overlay = create_overlay(images_np[i], masks_np[i, 0], color=(0, 1, 0), alpha=0.4)
        axes[i, 0].imshow(gt_overlay)
        axes[i, 0].set_title(f'Ground Truth (Green)\n{person_counts[i]} Person(s)')
        axes[i, 0].axis('off')

        # Teacher prediction overlay (cyan) - Column 1
        teacher_overlay = create_overlay(images_np[i], teacher_preds[i, 0], color=(0, 1, 1), alpha=0.4)
        axes[i, 1].imshow(teacher_overlay)
        teacher_title = f'Teacher {teacher_name} (Cyan)'
        if teacher_miou is not None:
            teacher_title += f'\nmIoU: {teacher_miou:.4f}'
        axes[i, 1].set_title(teacher_title)
        axes[i, 1].axis('off')

        # Student prediction overlay (yellow) - Column 2
        student_overlay = create_overlay(images_np[i], student_preds[i, 0], color=(1, 1, 0), alpha=0.4)
        axes[i, 2].imshow(student_overlay)
        student_title = f'Student {student_name} (Yellow)'
        if student_miou is not None:
            student_title += f'\nmIoU: {student_miou:.4f}'
        axes[i, 2].set_title(student_title)
        axes[i, 2].axis('off')

        # Comparison overlay (Ground Truth vs Student) - Column 3
        comparison = img.copy()
        gt_mask = masks_np[i, 0] > 0.5  # Ground truth mask
        student_mask = student_preds[i, 0] > 0.5

        # Identify different pixel categories
        # Foreground: both have person (True)
        foreground_both = gt_mask & student_mask
        # Mismatch: predictions differ from ground truth
        mismatch = (gt_mask != student_mask)

        # Alpha values - use same as Ground Truth for green
        alpha_green = 0.4  # Same as Ground Truth
        alpha_red = 0.7    # Strong red for mismatches

        # Apply colors with priority: Red (mismatch) > Green (match) > Original
        # Start with original image
        comparison = img.copy()

        # First apply green for foreground matches (same as GT)
        # Only apply if NOT a mismatch pixel
        green_mask = foreground_both & (~mismatch)  # This should be same as foreground_both
        comparison[:, :, 0] = img[:, :, 0] * (1 - alpha_green * green_mask)  # Reduce red
        comparison[:, :, 1] = img[:, :, 1] * (1 - alpha_green * green_mask) + \
                              1.0 * alpha_green * green_mask  # Add green
        comparison[:, :, 2] = img[:, :, 2] * (1 - alpha_green * green_mask)  # Reduce blue

        # Then apply red for mismatches (overwrites green if any)
        comparison[:, :, 0] = comparison[:, :, 0] * (1 - alpha_red * mismatch) + \
                              1.0 * alpha_red * mismatch  # Add red
        comparison[:, :, 1] = comparison[:, :, 1] * (1 - alpha_red * mismatch)  # Reduce green
        comparison[:, :, 2] = comparison[:, :, 2] * (1 - alpha_red * mismatch)  # Reduce blue

        axes[i, 3].imshow(comparison)
        axes[i, 3].set_title('Ground Truth vs Student\n(Green=Correct, Red=Error)')
        axes[i, 3].axis('off')

    plt.suptitle(f'Epoch {epoch+1:03d}', fontsize=16)
    plt.tight_layout()

    # Save figure
    save_path = visualize_dir / f'epoch_{epoch+1:03d}.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Visualization saved to {save_path}")


def export_onnx_model(model: nn.Module, save_dir: Path, device: str):
    """Export model to ONNX format.

    Args:
        model: Model to export
        save_dir: Directory to save ONNX file
        device: Device to use for export
    """
    try:
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, 3, 640, 640).to(device)

        # Export to ONNX
        onnx_path = save_dir / 'model.onnx'
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        print(f"  Model exported to ONNX: {onnx_path}")

        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print(f"  ONNX model verified successfully")

            # Simplify ONNX model with onnxsim
            try:
                import onnxsim
                print(f"  Simplifying ONNX model with onnxsim...")
                model_sim, check = onnxsim.simplify(
                    onnx_model,
                    overwrite_input_shapes={'input': [1, 3, 640, 640]}
                )

                if check:
                    # Save simplified model
                    onnx_sim_path = save_dir / 'model_simplified.onnx'
                    onnx.save(model_sim, str(onnx_sim_path))
                    print(f"  Simplified model saved to: {onnx_sim_path}")

                    # Also overwrite original with simplified version
                    onnx.save(model_sim, str(onnx_path))
                    print(f"  Original model replaced with simplified version")

                    # Report size reduction
                    import os
                    original_size = os.path.getsize(str(onnx_path))
                    print(f"  Model size: {original_size / (1024*1024):.2f} MB")
                else:
                    print("  Warning: Simplified model failed validation check")

            except ImportError:
                print("  Warning: onnxsim not installed. Install with: pip install onnx-simplifier")
            except Exception as e:
                print(f"  Warning: ONNX simplification failed: {e}")

        except ImportError:
            print("  Warning: onnx package not installed, skipping verification")
        except Exception as e:
            print(f"  Warning: ONNX verification failed: {e}")

    except Exception as e:
        print(f"  Warning: Failed to export ONNX model: {e}")


def get_encoder_name(encoder_type: str) -> str:
    """Extract encoder name from encoder type string.

    Args:
        encoder_type: Encoder type string (e.g., 'timm-efficientnet-b0')

    Returns:
        Short name for logging (e.g., 'B0')
    """
    if 'efficientnet-b0' in encoder_type.lower():
        return 'B0'
    elif 'efficientnet-b1' in encoder_type.lower():
        return 'B1'
    elif 'efficientnet-b2' in encoder_type.lower():
        return 'B2'
    elif 'efficientnet-b3' in encoder_type.lower():
        return 'B3'
    elif 'efficientnet-b4' in encoder_type.lower():
        return 'B4'
    elif 'efficientnet-b5' in encoder_type.lower():
        return 'B5'
    else:
        # Default fallback - extract last part
        parts = encoder_type.split('-')
        if len(parts) > 1:
            return parts[-1].upper()
        return encoder_type.upper()


def main():
    parser = argparse.ArgumentParser(description='UNet Decoder Distillation Training')
    parser.add_argument('--config', type=str, default='rgb_hierarchical_unet_v2_distillation_b0_from_b3',
                       help='Configuration name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--resume', type=str, default='',
                       help='Resume from checkpoint')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--use_heavy_augmentation', action='store_true', default=True,
                       help='Use heavy augmentations during training (default: True)')
    parser.add_argument('--no_heavy_augmentation', dest='use_heavy_augmentation', action='store_false',
                       help='Disable heavy augmentations during training')
    parser.add_argument('--test_only', action='store_true',
                       help='Only run validation without training (requires --resume)')
    parser.add_argument('--teacher_encoder', type=str, default=None,
                       help='Teacher model encoder name (e.g., timm-efficientnet-b7)')
    args = parser.parse_args()

    # Load configuration
    config = ConfigManager.get_config(args.config)

    # Override config with command line arguments if provided
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.teacher_encoder is not None:
        config.distillation.teacher_encoder = args.teacher_encoder

    # Get encoder names for logging
    student_name = get_encoder_name(config.distillation.student_encoder)
    teacher_name = get_encoder_name(config.distillation.teacher_encoder)

    # Create experiment directories
    exp_dirs = create_experiment_dirs(config)

    # Setup logging (pass logs directory, not a file path)
    text_logger = TextLogger(exp_dirs['logs'])
    if args.test_only:
        text_logger.log(f"Running validation only with config: {config.name}")
        if not args.resume:
            raise ValueError("--test_only requires --resume to specify checkpoint")
    else:
        text_logger.log(f"Starting UNet decoder distillation training with config: {config.name}")
    text_logger.log(f"Batch size: {config.training.batch_size}")
    text_logger.log(f"Epochs: {config.training.num_epochs}")
    text_logger.log(f"Heavy augmentation: {'Enabled' if args.use_heavy_augmentation else 'Disabled'}")
    text_logger.log(f"Configuration:\n{json.dumps(config.__dict__, default=str, indent=2)}")

    # Setup tensorboard
    writer = SummaryWriter(exp_dirs['logs'])

    # Parse temperature scheduling parameters from feature_match_layers
    enable_temp_scheduling = False
    initial_temperature = config.distillation.temperature
    final_temperature = 1.0
    schedule_type = "linear"

    # Parse progressive unfreezing parameters
    enable_progressive_unfreeze = False
    unfreeze_start_epoch = 10
    unfreeze_rate = 5
    encoder_lr_scale = 0.1

    # Check if temperature scheduling is configured
    if (config.distillation.feature_match_layers and
        len(config.distillation.feature_match_layers) >= 5 and
        config.distillation.feature_match_layers[0] == "temp_scheduling"):
        enable_temp_scheduling = config.distillation.feature_match_layers[1].lower() == "true"
        initial_temperature = float(config.distillation.feature_match_layers[2])
        final_temperature = float(config.distillation.feature_match_layers[3])
        schedule_type = config.distillation.feature_match_layers[4]

        # Check for progressive unfreezing parameters (if present)
        if len(config.distillation.feature_match_layers) >= 10:
            if config.distillation.feature_match_layers[5] == "progressive_unfreeze":
                enable_progressive_unfreeze = config.distillation.feature_match_layers[6].lower() == "true"
                unfreeze_start_epoch = int(config.distillation.feature_match_layers[7])
                unfreeze_rate = int(config.distillation.feature_match_layers[8])
                encoder_lr_scale = float(config.distillation.feature_match_layers[9])

    # Log temperature scheduling configuration
    if enable_temp_scheduling:
        text_logger.log(f"Temperature scheduling: Enabled")
        text_logger.log(f"  Initial temperature: {initial_temperature:.2f}")
        text_logger.log(f"  Final temperature: {final_temperature:.2f}")
        text_logger.log(f"  Schedule type: {schedule_type}")
    else:
        text_logger.log(f"Temperature: Fixed at {initial_temperature:.2f}")

    # Log progressive unfreezing configuration
    if enable_progressive_unfreeze:
        text_logger.log(f"Progressive unfreezing: Enabled")
        text_logger.log(f"  Start epoch: {unfreeze_start_epoch}")
        text_logger.log(f"  Unfreeze rate: 1 block every {unfreeze_rate} epochs")
        text_logger.log(f"  Encoder LR scale: {encoder_lr_scale}")
    else:
        text_logger.log(f"Progressive unfreezing: Disabled")

    # Check if pretrained student weights should be used
    student_pretrained_path = None
    if config.model.use_pretrained_unet and config.model.pretrained_weights_path:
        student_pretrained_path = config.model.pretrained_weights_path
        text_logger.log(f"Will load pretrained student weights from: {student_pretrained_path}")
    
    # Check if distillation should be disabled (for pure fine-tuning)
    if not config.distillation.enabled:
        text_logger.log("Distillation DISABLED - Pure fine-tuning mode")
        # Override distillation parameters for pure fine-tuning
        config.distillation.alpha = 0.0
        config.distillation.task_weight = 1.0
        # Don't need teacher model for pure fine-tuning
        config.distillation.teacher_checkpoint = None
    
    # Create model and loss
    model, loss_fn = create_unet_distillation_model(
        student_encoder=config.distillation.student_encoder,
        teacher_encoder=config.distillation.teacher_encoder,
        teacher_checkpoint=config.distillation.teacher_checkpoint,
        device=args.device,
        progressive_unfreeze=enable_progressive_unfreeze,  # Pass progressive unfreeze flag
        adaptive_distillation=config.distillation.adaptive_distillation,
        amplification_factor=config.distillation.amplification_factor,
        min_alpha=config.distillation.min_alpha,
        student_pretrained_path=student_pretrained_path
    )

    # Set initial temperature
    loss_fn.temperature = initial_temperature
    loss_fn.initial_temperature = initial_temperature

    # Create optimizer
    if enable_progressive_unfreeze:
        # Progressive unfreeze: start with decoder only
        optimizer = torch.optim.AdamW(
            model.student.get_decoder_parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        # No progressive unfreeze: train entire model from start
        optimizer = torch.optim.AdamW(
            model.student.parameters(),  # All parameters (encoder + decoder)
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        text_logger.log("Training entire model (encoder + decoder) from the start")

    # Create scheduler
    if config.training.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.min_lr
        )
    else:
        scheduler = None

    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision else None

    # Create dataloaders
    train_loader = create_dataloader(config, is_training=True, use_heavy_augmentation=args.use_heavy_augmentation)
    val_loader = create_dataloader(config, is_training=False, use_heavy_augmentation=False)

    # Export initial model to ONNX
    export_onnx_model(model.student, exp_dirs['checkpoints'], args.device)

    # Test only mode - skip training and run validation
    if args.test_only:
        text_logger.log("\n" + "="*50)
        text_logger.log("Running validation only...")

        # Run validation
        val_metrics = evaluate(
            model,
            val_loader,
            loss_fn,
            args.device,
            0,  # Use epoch 0 for test_only mode
            writer,
            visualize_dir=exp_dirs['visualizations'],
            teacher_miou_cache=None,
            student_name=student_name,
            teacher_name=teacher_name
        )

        # Log results
        text_logger.log(f"\nValidation Results:")
        text_logger.log(f"  Total Loss: {val_metrics['total_loss']:.4f}")
        text_logger.log(f"  Student {student_name} mIoU: {val_metrics['student_miou']:.4f}")
        text_logger.log(f"  Teacher {teacher_name} mIoU: {val_metrics['teacher_miou']:.4f}")
        text_logger.log(f"  Agreement Rate: {val_metrics['agreement']:.4f}")
        text_logger.log(f"  KL Loss: {val_metrics['kl_loss']:.4f}")
        text_logger.log(f"  MSE Loss: {val_metrics['mse_loss']:.4f}")
        text_logger.log(f"  BCE Loss: {val_metrics['bce_loss']:.4f}")
        text_logger.log(f"  Dice Loss: {val_metrics['dice_loss']:.4f}")

        text_logger.log("\nValidation completed!")
        writer.close()
        return

    # Training loop
    best_iou = 0
    start_epoch = 0
    teacher_miou_cache = None  # Cache for teacher mIoU

    # Get progressive unfreezing schedule if enabled
    unfreeze_schedule = None
    prev_blocks_unfrozen = 0

    if enable_progressive_unfreeze:
        unfreeze_schedule = model.get_progressive_unfreeze_schedule(
            total_epochs=config.training.num_epochs,
            unfreeze_start_epoch=unfreeze_start_epoch,
            unfreeze_rate=unfreeze_rate
        )
        text_logger.log(f"Progressive unfreeze schedule created")

    # Resume from checkpoint if specified
    if args.resume or args.test_only:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.student.load_state_dict(checkpoint['model_state_dict'])

        # Get the resume epoch first
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0)

        # Restore loss function state (adaptive distillation)
        if 'loss_fn_state' in checkpoint:
            loss_fn_state = checkpoint['loss_fn_state']
            if hasattr(loss_fn, 'performance_ratio'):
                loss_fn.performance_ratio = loss_fn_state.get('performance_ratio', 1.0)
            if hasattr(loss_fn, 'alpha'):
                loss_fn.alpha = loss_fn_state.get('alpha', config.distillation.alpha)
            if hasattr(loss_fn, 'task_weight'):
                loss_fn.task_weight = loss_fn_state.get('task_weight', 0.7)
            if hasattr(loss_fn, 'temperature'):
                loss_fn.temperature = loss_fn_state.get('temperature', 1.0)
            if hasattr(loss_fn, 'distillation_eliminated'):
                loss_fn.distillation_eliminated = loss_fn_state.get('distillation_eliminated', False)

            text_logger.log(f"Restored loss function state: ratio={loss_fn.performance_ratio:.4f}, "
                           f"α={loss_fn.alpha:.3f}, task_weight={loss_fn.task_weight:.3f}, "
                           f"eliminated={loss_fn.distillation_eliminated}")

            # Print special message if distillation was eliminated
            if loss_fn.distillation_eliminated:
                text_logger.log("  🎯 Distillation was PERMANENTLY ELIMINATED - continuing with 100% ground truth")
                print("  🎯 Distillation was PERMANENTLY ELIMINATED - continuing with 100% ground truth")

        # Restore cached teacher mIoU
        if 'teacher_miou_cache' in checkpoint:
            teacher_miou_cache = checkpoint['teacher_miou_cache']
            text_logger.log(f"Restored cached teacher mIoU: {teacher_miou_cache:.4f}")

        # Restore progressive unfreeze state from checkpoint
        if enable_progressive_unfreeze and 'progressive_unfreeze_state' in checkpoint and checkpoint['progressive_unfreeze_state']:
            saved_blocks_unfrozen = checkpoint['progressive_unfreeze_state'].get('blocks_unfrozen', 0)
            text_logger.log(f"Checkpoint has {saved_blocks_unfrozen} blocks unfrozen")

            # If blocks were unfrozen in the checkpoint, restore that exact state
            if saved_blocks_unfrozen > 0:
                text_logger.log(f"Restoring progressive unfreeze state: unfreezing {saved_blocks_unfrozen} encoder blocks")

                # Unfreeze the same number of blocks as in checkpoint
                unfrozen_params = model.unfreeze_encoder_blocks(
                    num_blocks=saved_blocks_unfrozen,
                    learning_rate_scale=encoder_lr_scale
                )

                if unfrozen_params:
                    # Get decoder parameters
                    decoder_params = model.student.get_decoder_parameters()

                    # Convert to lists to get counts
                    decoder_params_list = list(decoder_params)
                    unfrozen_params_list = list(unfrozen_params)

                    # Recreate optimizer with the same structure as checkpoint
                    optimizer = torch.optim.AdamW([
                        {'params': decoder_params_list, 'lr': config.training.learning_rate},
                        {'params': unfrozen_params_list, 'lr': config.training.learning_rate * encoder_lr_scale}
                    ], weight_decay=config.training.weight_decay)

                    text_logger.log(f"Optimizer recreated with {len(decoder_params_list)} decoder params and {len(unfrozen_params_list)} encoder params")
                    prev_blocks_unfrozen = saved_blocks_unfrozen

                    # Set initial_lr for scheduler
                    for group in optimizer.param_groups:
                        if 'initial_lr' not in group:
                            group['initial_lr'] = group['lr']

                    # Now load the optimizer state - it should match since we recreated the same structure
                    try:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        text_logger.log("Optimizer state restored successfully")
                    except Exception as e:
                        text_logger.log(f"Warning: Could not fully restore optimizer state: {e}")
                        text_logger.log("Newly unfrozen parameters will use fresh optimizer state")

                    # Recreate scheduler at the correct position
                    if config.training.scheduler == 'cosine':
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=config.training.num_epochs,
                            eta_min=config.training.min_lr,
                            last_epoch=start_epoch-1
                        )
                        text_logger.log(f"Scheduler recreated at epoch {start_epoch}")
            else:
                # No blocks were unfrozen yet, just load optimizer normally
                prev_blocks_unfrozen = 0
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    text_logger.log("Optimizer state loaded successfully")
                except Exception as e:
                    text_logger.log(f"Warning: Could not load optimizer state: {e}")
        else:
            # No progressive unfreezing or not enabled, load optimizer normally
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                text_logger.log("Optimizer state loaded successfully")
            except Exception as e:
                text_logger.log(f"Warning: Could not load optimizer state: {e}")

        if scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except:
                text_logger.log("Could not restore scheduler state, maintaining recreated scheduler")

        text_logger.log(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    for epoch in range(start_epoch, config.training.num_epochs):
        # Progressive unfreezing check
        if enable_progressive_unfreeze and unfreeze_schedule:
            blocks_to_unfreeze = unfreeze_schedule[epoch]

            # Check if we need to unfreeze more blocks
            if blocks_to_unfreeze > prev_blocks_unfrozen:
                text_logger.log(f"Epoch {epoch+1:03d} - Unfreezing {blocks_to_unfreeze} encoder blocks")

                # Unfreeze encoder blocks
                unfrozen_params = model.unfreeze_encoder_blocks(
                    num_blocks=blocks_to_unfreeze,
                    learning_rate_scale=encoder_lr_scale
                )

                # Update optimizer with new parameter groups
                if unfrozen_params:
                    # Get decoder parameters and convert to lists
                    decoder_params = list(model.student.get_decoder_parameters())
                    unfrozen_params = list(unfrozen_params)

                    # Create new optimizer with discriminative learning rates
                    new_optimizer = torch.optim.AdamW([
                        {'params': decoder_params, 'lr': config.training.learning_rate},
                        {'params': unfrozen_params, 'lr': config.training.learning_rate * encoder_lr_scale}
                    ], weight_decay=config.training.weight_decay)

                    # Transfer optimizer state from old to new optimizer
                    # Only transfer state for decoder parameters that existed before
                    if hasattr(optimizer, 'state') and optimizer.state:
                        transferred_count = 0
                        # The first param group in new optimizer is decoder params
                        decoder_group = new_optimizer.param_groups[0]

                        # The old optimizer also had decoder params as first group
                        old_decoder_group = optimizer.param_groups[0]

                        # Map decoder parameters (they should be the same objects)
                        for new_p in decoder_group['params']:
                            for old_p in old_decoder_group['params']:
                                if new_p is old_p and old_p in optimizer.state:
                                    new_optimizer.state[new_p] = optimizer.state[old_p]
                                    transferred_count += 1
                                    break

                        text_logger.log(f"  Transferred optimizer state for {transferred_count} decoder parameters")

                    # Replace old optimizer with new one
                    optimizer = new_optimizer

                    # Set initial_lr for scheduler restoration
                    for group in optimizer.param_groups:
                        if 'initial_lr' not in group:
                            group['initial_lr'] = group['lr']

                    # Recreate scheduler with new optimizer
                    if config.training.scheduler == 'cosine':
                        # Create a new scheduler but maintain the current learning rate schedule
                        # Use last_epoch parameter to set the current position without calling step()
                        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer,
                            T_max=config.training.num_epochs,  # Total epochs
                            eta_min=config.training.min_lr,
                            last_epoch=epoch-1  # Set current position without calling step()
                        )

                    text_logger.log(f"  Optimizer updated with {len(unfrozen_params)} unfrozen encoder parameters")
                    text_logger.log(f"  Total parameters in optimizer: decoder={len(decoder_params)}, encoder={len(unfrozen_params)}")
                    # Log current learning rates after scheduler update
                    if scheduler:
                        current_lrs = scheduler.get_last_lr()
                        if len(current_lrs) > 1:
                            text_logger.log(f"  Current LR - Decoder: {current_lrs[0]:.6f}, Encoder: {current_lrs[1]:.6f}")
                        else:
                            text_logger.log(f"  Current LR: {current_lrs[0]:.6f}")
                    else:
                        text_logger.log(f"  Decoder LR: {config.training.learning_rate:.6f}, Encoder LR: {config.training.learning_rate * encoder_lr_scale:.6f}")

                prev_blocks_unfrozen = blocks_to_unfreeze

        # Log current learning rates and scheduler status at the start of epoch
        if scheduler:
            current_lrs = scheduler.get_last_lr()
            if len(current_lrs) > 1:
                text_logger.log(f"Epoch {epoch+1:03d} - Learning Rates - Decoder: {current_lrs[0]:.6f}, Encoder: {current_lrs[1]:.6f}")
            else:
                text_logger.log(f"Epoch {epoch+1:03d} - Learning Rate: {current_lrs[0]:.6f}")

        # Update temperature if scheduling is enabled
        if enable_temp_scheduling:
            current_temp = loss_fn.update_temperature(
                current_epoch=epoch,
                total_epochs=config.training.num_epochs,
                final_temperature=final_temperature,
                schedule_type=schedule_type
            )
            text_logger.log(f"Epoch {epoch+1:03d} - Temperature updated to: {current_temp:.3f}")
            writer.add_scalar('Train/Temperature', current_temp, epoch)

        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer,
            args.device, epoch, scaler, writer
        )
        # Log training results
        if config.distillation.enabled:
            text_logger.log(f"Epoch {epoch+1:03d} - Train - Loss: {train_metrics['total_loss']:.4f}, "
                           f"KL: {train_metrics['kl_loss']:.4f}, MSE: {train_metrics['mse_loss']:.4f}, "
                           f"Dice: {train_metrics['dice_loss']:.4f}")
        else:
            # Simplified log for pure fine-tuning
            text_logger.log(f"Epoch {epoch+1:03d} - Train - Loss: {train_metrics['total_loss']:.4f}, "
                           f"BCE: {train_metrics['bce_loss']:.4f}, Dice: {train_metrics['dice_loss']:.4f}")

        # Validate
        val_metrics = evaluate(
            model, val_loader, loss_fn,
            args.device, epoch, writer,
            visualize_dir=exp_dirs['visualizations'],
            teacher_miou_cache=teacher_miou_cache,
            student_name=student_name,
            teacher_name=teacher_name
        )

        # Cache teacher mIoU after first evaluation (only if distillation is enabled)
        if config.distillation.enabled and teacher_miou_cache is None:
            teacher_miou_cache = val_metrics['teacher_miou']
            text_logger.log(f"Cached Teacher {teacher_name} mIoU: {teacher_miou_cache:.4f}")

        # Update distillation weight based on student vs teacher performance (only if distillation is enabled)
        if config.distillation.enabled and hasattr(loss_fn, 'update_distillation_weight'):
            old_alpha = loss_fn.alpha
            old_task_weight = loss_fn.task_weight
            new_alpha = loss_fn.update_distillation_weight(
                student_iou=val_metrics['student_miou'],
                teacher_iou=teacher_miou_cache,
                min_alpha=config.distillation.min_alpha,
                amplification_factor=config.distillation.amplification_factor,
                zero_distillation_threshold=config.distillation.zero_distillation_threshold
            )
            if old_alpha != new_alpha or old_task_weight != loss_fn.task_weight:
                amplified_diff = (loss_fn.performance_ratio - 1.0) * config.distillation.amplification_factor
                if loss_fn.distillation_eliminated and old_alpha != 0.0:
                    # Distillation just got permanently eliminated
                    adaptive_msg = (f"  🎯 Distillation PERMANENTLY ELIMINATED: Student/Teacher ratio = {loss_fn.performance_ratio:.4f}, "
                                   f"Student surpassed teacher by {(loss_fn.performance_ratio - 1.0)*100:.1f}%, "
                                   f"Now using 100% ground truth labels FOREVER!")
                elif loss_fn.distillation_eliminated:
                    # Distillation was already eliminated
                    adaptive_msg = (f"  🎯 Distillation remains ELIMINATED: Student/Teacher ratio = {loss_fn.performance_ratio:.4f}, "
                                   f"Continuing with 100% ground truth labels")
                else:
                    adaptive_msg = (f"  Adaptive distillation: Student/Teacher ratio = {loss_fn.performance_ratio:.4f}, "
                                   f"amplified_diff = {amplified_diff:.3f}, "
                                   f"α: {old_alpha:.3f} → {new_alpha:.3f}, "
                                   f"task_weight: {old_task_weight:.3f} → {loss_fn.task_weight:.3f}")
                text_logger.log(adaptive_msg)
                print(adaptive_msg)  # Also print to console

        # Log validation results
        if config.distillation.enabled:
            text_logger.log(f"Epoch {epoch+1:03d} - Val - Loss: {val_metrics['total_loss']:.4f}, "
                           f"Dice: {val_metrics['dice_loss']:.4f}, {student_name}_mIoU: {val_metrics['student_miou']:.4f}, "
                           f"{teacher_name}_mIoU: {val_metrics['teacher_miou']:.4f}, Agreement: {val_metrics['agreement']:.4f}")
        else:
            # Simplified log for pure fine-tuning (no teacher metrics)
            text_logger.log(f"Epoch {epoch+1:03d} - Val - Loss: {val_metrics['total_loss']:.4f}, "
                           f"Dice: {val_metrics['dice_loss']:.4f}, mIoU: {val_metrics['student_miou']:.4f}")

        # Update scheduler
        if scheduler:
            scheduler.step()
            # Log learning rates for all parameter groups
            current_lrs = scheduler.get_last_lr()
            if len(current_lrs) == 1:
                writer.add_scalar('Train/LR', current_lrs[0], epoch)
            else:
                # Multiple parameter groups (decoder and encoder)
                writer.add_scalar('Train/LR_Decoder', current_lrs[0], epoch)
                writer.add_scalar('Train/LR_Encoder', current_lrs[1], epoch)
                # Also log the main LR (decoder) for backward compatibility
                writer.add_scalar('Train/LR', current_lrs[0], epoch)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'best_iou': best_iou,
            'config': config.__dict__,
            # Save progressive unfreezing state
            'progressive_unfreeze_state': {
                'enabled': enable_progressive_unfreeze,
                'blocks_unfrozen': prev_blocks_unfrozen if enable_progressive_unfreeze else 0,
                'unfreeze_start_epoch': unfreeze_start_epoch,
                'unfreeze_rate': unfreeze_rate,
                'encoder_lr_scale': encoder_lr_scale,
            } if enable_progressive_unfreeze else None,
            # Save loss function state (adaptive distillation)
            'loss_fn_state': {
                'performance_ratio': loss_fn.performance_ratio if hasattr(loss_fn, 'performance_ratio') else 1.0,
                'alpha': loss_fn.alpha if hasattr(loss_fn, 'alpha') else config.distillation.alpha,
                'task_weight': loss_fn.task_weight if hasattr(loss_fn, 'task_weight') else 0.7,
                'temperature': loss_fn.temperature if hasattr(loss_fn, 'temperature') else 1.0,
                'distillation_eliminated': loss_fn.distillation_eliminated if hasattr(loss_fn, 'distillation_eliminated') else False,
            },
            'teacher_miou_cache': teacher_miou_cache,  # Save cached teacher mIoU
        }

        # Save regular checkpoint
        checkpoint_path = exp_dirs['checkpoints'] / f'checkpoint_epoch_{epoch+1:04d}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            best_path = exp_dirs['checkpoints'] / 'best_model.pth'
            torch.save(checkpoint, best_path)
            text_logger.log(f"New best model saved with IoU: {best_iou:.4f}")

            # Export best model to ONNX
            export_onnx_model(model.student, exp_dirs['checkpoints'], args.device)

    text_logger.log("Training completed!")
    writer.close()


if __name__ == "__main__":
    main()