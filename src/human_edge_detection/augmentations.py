"""Data augmentation transforms for training.

This module provides augmentation transforms based on albumentations library.
Supports both light and heavy augmentation modes.

Note: For ROI-based tasks, geometric transformations (flip, affine) are disabled
by default to avoid ROI coordinate misalignment issues.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple


def get_light_train_transforms(input_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Lightweight augmentation for faster training.

    Args:
        input_size: Target image size (height, width)

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Basic geometric transforms
        A.HorizontalFlip(p=0.5),
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.85, 1.15),
            rotate=0,  # No rotation for ROI-based tasks
            p=0.5
        ),

        # Color augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.8),

        # Slight blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

        # Normalization and conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_heavy_train_transforms(input_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Heavy augmentation for better generalization.

    Args:
        input_size: Target image size (height, width)

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),

        # Spatial transforms (shift and scale only, no rotation for ROI tasks)
        A.Affine(
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            scale=(0.8, 1.2),
            rotate=0,  # No rotation
            p=0.5
        ),

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
            A.RandomFog(alpha_coef=0.1, p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
                src_radius=100,
                src_color=(255, 255, 255),
                p=1.0
            ),
        ], p=0.1),

        # Blur effects (motion blur, defocus)
        A.OneOf([
            A.MotionBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.05),

        # Noise
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.05),

        # Image quality degradation
        A.OneOf([
            A.ImageCompression(quality_range=(70, 95), p=1.0),
            A.Downscale(scale_range=(0.5, 0.9), p=1.0),
        ], p=0.1),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_train_transforms(input_size: Tuple[int, int] = (640, 640),
                        use_heavy_augmentation: bool = False) -> A.Compose:
    """Get training transforms based on augmentation setting.

    Args:
        input_size: Target image size (height, width)
        use_heavy_augmentation: If True, use heavy augmentation; if False, use light

    Returns:
        Albumentations compose transform
    """
    if use_heavy_augmentation:
        return get_heavy_train_transforms(input_size)
    else:
        return get_light_train_transforms(input_size)


def get_val_transforms(input_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """Get validation transforms (no augmentation, only normalization).

    Args:
        input_size: Target image size (height, width)

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_roi_safe_light_transforms(input_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """ROI-safe lightweight augmentation (with horizontal flip only).

    Includes horizontal flip and color/brightness augmentations.
    ROI coordinates must be adjusted for horizontal flip in the dataset.

    Args:
        input_size: Target image size (height, width)

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Horizontal flip only (ROI coordinates need to be adjusted)
        A.HorizontalFlip(p=0.5),
        
        # Color augmentations only
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=1.0),
        ], p=0.8),

        # Slight blur
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),

        # Normalization and conversion
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))


def get_roi_safe_heavy_transforms(input_size: Tuple[int, int] = (640, 640)) -> A.Compose:
    """ROI-safe heavy augmentation (with horizontal flip only).

    Includes horizontal flip and various color, lighting, and effect augmentations.
    ROI coordinates must be adjusted for horizontal flip in the dataset.

    Args:
        input_size: Target image size (height, width)

    Returns:
        Albumentations compose transform
    """
    return A.Compose([
        # Horizontal flip only (ROI coordinates need to be adjusted)
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

        # Weather/environmental effects (no geometric distortion)
        A.OneOf([
            A.RandomRain(drop_length=20, drop_width=1, drop_color=(200, 200, 200),
                        blur_value=5, brightness_coefficient=0.7, rain_type='drizzle', p=1.0),
            A.RandomFog(alpha_coef=0.1, p=1.0),
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.5),
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
            A.GaussNoise(p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.05),

        # Image quality degradation
        A.OneOf([
            A.ImageCompression(quality_range=(70, 95), p=1.0),
            A.Downscale(scale_range=(0.5, 0.9), p=1.0),
        ], p=0.1),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))


def get_roi_safe_transforms(input_size: Tuple[int, int] = (640, 640),
                           use_heavy_augmentation: bool = False) -> A.Compose:
    """Get ROI-safe training transforms (with horizontal flip only).

    Args:
        input_size: Target image size (height, width)
        use_heavy_augmentation: If True, use heavy augmentation; if False, use light

    Returns:
        Albumentations compose transform
    """
    if use_heavy_augmentation:
        return get_roi_safe_heavy_transforms(input_size)
    else:
        return get_roi_safe_light_transforms(input_size)