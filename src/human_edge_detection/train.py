"""Training pipeline with full parameter control for ROI-based instance segmentation.

This module provides comprehensive training functionality with support for:
- Multiple optimizers (Adam, AdamW, SGD)
- Various learning rate schedulers (Cosine, Step, Exponential)
- Configurable data loading parameters
- Gradient clipping
- And more...

For a simplified tutorial version, see train_tutorial.py.
"""

import torch
import torch.optim as optim
from .train_tutorial import Trainer, create_data_loaders
from .model import create_model
from .losses import create_loss_function
from .feature_extractor import YOLOv9FeatureExtractor
from .dataset import COCOInstanceSegmentationDataset
from torch.utils.data import DataLoader
from typing import Optional


def create_enhanced_data_loaders(
    train_annotation_file: str,
    val_annotation_file: str,
    train_image_dir: str,
    val_image_dir: str,
    batch_size: int,
    num_workers: int,
    image_size: tuple = (640, 640),
    min_roi_size: int = 16
) -> tuple:
    """Create data loaders with configurable parameters."""
    # Create datasets
    train_dataset = COCOInstanceSegmentationDataset(
        annotation_file=train_annotation_file,
        image_dir=train_image_dir,
        image_size=image_size,
        mask_size=(56, 56),
        min_roi_size=min_roi_size
    )
    
    val_dataset = COCOInstanceSegmentationDataset(
        annotation_file=val_annotation_file,
        image_dir=val_image_dir,
        image_size=image_size,
        mask_size=(56, 56),
        min_roi_size=min_roi_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_optimizer(model, config: dict):
    """Create optimizer based on config."""
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
    elif config['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
    elif config['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config.get('momentum', 0.9),
            weight_decay=config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler based on config."""
    if config['scheduler'] == 'none':
        return None
    elif config['scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config.get('min_lr', 1e-6)
        )
    elif config['scheduler'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.get('lr_decay_epochs', [30, 60, 90]),
            gamma=config.get('lr_decay_rate', 0.1)
        )
    elif config['scheduler'] == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=config.get('lr_decay_rate', 0.95)
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['scheduler']}")
    
    return scheduler


def create_enhanced_trainer(
    config: dict,
    train_annotation_file: str,
    val_annotation_file: str,
    train_image_dir: str,
    val_image_dir: str,
    onnx_model_path: str,
    pixel_ratios: dict,
    device: str = 'cuda',
    separation_aware_weights: Optional[dict] = None
) -> Trainer:
    """Create trainer with enhanced configuration support."""
    # Create data loaders
    train_loader, val_loader = create_enhanced_data_loaders(
        train_annotation_file=train_annotation_file,
        val_annotation_file=val_annotation_file,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        image_size=config.get('image_size', (640, 640)),
        min_roi_size=config.get('min_roi_size', 16)
    )
    
    # Create feature extractor
    feature_extractor = YOLOv9FeatureExtractor(
        onnx_path=onnx_model_path,
        device=device
    )
    
    # Create model
    model = create_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        mid_channels=config['mid_channels'],
        mask_size=config['mask_size'],
        roi_size=config.get('roi_size', 28)  # Default to improved ROI size
    )
    
    # Create loss function
    loss_fn = create_loss_function(
        pixel_ratios=pixel_ratios,
        use_log_weights=config.get('use_log_weights', True),
        ce_weight=config.get('ce_weight', 1.0),
        dice_weight=config.get('dice_weight', 1.0),
        dice_classes=config.get('dice_classes', [1]),
        device=device,
        separation_aware_weights=separation_aware_weights
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        feature_extractor=feature_extractor,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        log_dir=config.get('log_dir', 'logs'),
        save_every=config.get('save_every', 1),
        validate_every=config.get('validate_every', 1)
    )
    
    # Set gradient clipping if specified
    if config.get('gradient_clip', 0) > 0:
        trainer.gradient_clip_val = config['gradient_clip']
    
    # Create and set validation visualizer
    from pycocotools.coco import COCO
    from .visualize import ValidationVisualizer
    
    val_coco = COCO(val_annotation_file)
    visualizer = ValidationVisualizer(
        model=model,
        feature_extractor=feature_extractor,
        coco=val_coco,
        image_dir=val_image_dir,
        output_dir=config.get('validation_output_dir', 'validation_results'),
        device=device
    )
    trainer.set_visualizer(visualizer)
    
    return trainer


# Alias for backward compatibility
create_trainer = create_enhanced_trainer