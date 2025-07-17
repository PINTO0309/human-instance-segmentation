"""Tutorial training pipeline for ROI-based instance segmentation.

This is a simplified version of the training pipeline with fixed settings.
For production use with full parameter control, see train.py.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import create_data_loaders
from .feature_extractor import YOLOv9FeatureExtractor
from .model import create_model, ROIBatchProcessor
from .losses import create_loss_function
from .visualize import ValidationVisualizer


class Trainer:
    """Trainer for ROI-based instance segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        feature_extractor: YOLOv9FeatureExtractor,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        save_every: int = 1,
        validate_every: int = 1
    ):
        """Initialize trainer.

        Args:
            model: Segmentation model
            feature_extractor: YOLO feature extractor
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
            save_every: Save checkpoint every N epochs
            validate_every: Run validation every N epochs
        """
        self.model = model.to(device)
        self.feature_extractor = feature_extractor
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_every = save_every
        self.validate_every = validate_every

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.epoch = 0
        self.best_miou = 0.0
        self.global_step = 0

        # Validation visualizer (will be set later)
        self.visualizer = None
        
        # Gradient clipping
        self.gradient_clip_val = 0.0

    def set_visualizer(self, visualizer: ValidationVisualizer):
        """Set validation visualizer."""
        self.visualizer = visualizer

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        epoch_losses = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'dice_loss': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} - Training', dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            roi_masks = batch['roi_mask'].to(self.device)
            roi_coords = batch['roi_coords'].to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.feature_extractor.extract_features(images)

            # Prepare ROIs for model
            rois = ROIBatchProcessor.prepare_rois_for_batch(roi_coords)
            rois = rois.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(features=features, rois=rois)
            masks_pred = output['masks']

            # Compute loss
            loss, loss_dict = self.loss_fn(masks_pred, roi_masks)

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.optimizer.step()

            # Update metrics
            for key, value in loss_dict.items():
                epoch_losses[key] += value.item()

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce': f"{loss_dict['ce_loss'].item():.4f}",
                'dice': f"{loss_dict['dice_loss'].item():.4f}",
                'lr': f"{current_lr:.6f}"
            })

            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/ce_loss', loss_dict['ce_loss'].item(), self.global_step)
                self.writer.add_scalar('train/dice_loss', loss_dict['dice_loss'].item(), self.global_step)

            self.global_step += 1

        # Average losses
        num_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self) -> Tuple[Dict[str, float], float]:
        """Run validation."""
        self.model.eval()

        val_losses = {
            'total_loss': 0.0,
            'ce_loss': 0.0,
            'dice_loss': 0.0
        }

        # IoU calculation
        intersection = torch.zeros(3).to(self.device)
        union = torch.zeros(3).to(self.device)

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch + 1} - Validation', dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar:
                # Move data to device
                images = batch['image'].to(self.device)
                roi_masks = batch['roi_mask'].to(self.device)
                roi_coords = batch['roi_coords'].to(self.device)

                # Extract features
                features = self.feature_extractor.extract_features(images)

                # Prepare ROIs
                rois = ROIBatchProcessor.prepare_rois_for_batch(roi_coords)
                rois = rois.to(self.device)

                # Forward pass
                output = self.model(features=features, rois=rois)
                masks_pred = output['masks']

                # Compute loss
                loss, loss_dict = self.loss_fn(masks_pred, roi_masks)

                # Update metrics
                for key, value in loss_dict.items():
                    val_losses[key] += value.item()

                # Compute predictions
                pred_classes = torch.argmax(masks_pred, dim=1)

                # Update IoU metrics
                for cls in range(3):
                    pred_mask = (pred_classes == cls)
                    target_mask = (roi_masks == cls)
                    intersection[cls] += (pred_mask & target_mask).sum().float()
                    union[cls] += (pred_mask | target_mask).sum().float()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })

        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches

        # Compute mIoU
        iou_per_class = intersection / (union + 1e-6)
        miou = iou_per_class[1:].mean().item()  # Exclude background
        
        # Ensure clean console output after progress bar
        print()  # Add newline after progress bar

        # Log validation metrics
        self.writer.add_scalar('val/total_loss', val_losses['total_loss'], self.epoch)
        self.writer.add_scalar('val/ce_loss', val_losses['ce_loss'], self.epoch)
        self.writer.add_scalar('val/dice_loss', val_losses['dice_loss'], self.epoch)
        self.writer.add_scalar('val/mIoU', miou, self.epoch)

        for cls, iou in enumerate(iou_per_class):
            self.writer.add_scalar(f'val/IoU_class_{cls}', iou.item(), self.epoch)

        return val_losses, miou

    def save_checkpoint(self, miou: float):
        """Save model checkpoint."""
        # Create checkpoint filename
        # checkpoint_epoch_{4桁epoch}_{Height}x{Width}_{mIoU4桁整数}.pth
        miou_int = int(miou * 10000)
        filename = f"checkpoint_epoch_{self.epoch:04d}_640x640_{miou_int:04d}.pth"
        filepath = self.checkpoint_dir / filename

        # Save checkpoint
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_miou': self.best_miou,
            'miou': miou,
            'loss_fn_state': {
                'class_weights': self.loss_fn.class_weights
            }
        }

        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filename}")

        # Update best model
        if miou > self.best_miou:
            self.best_miou = miou
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"New best model! mIoU: {miou:.4f}")

    def train(self, num_epochs: int, start_epoch: int = 0):
        """Train the model."""
        self.epoch = start_epoch

        for epoch in range(start_epoch, num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*50}")

            # Training
            train_losses = self.train_epoch()
            print(f"\nTraining - Loss: {train_losses['total_loss']:.4f}, "
                  f"CE: {train_losses['ce_loss']:.4f}, "
                  f"Dice: {train_losses['dice_loss']:.4f}")

            # Validation
            if (epoch + 1) % self.validate_every == 0:
                val_losses, miou = self.validate()
                
                # Display validation results prominently
                print(f"\n{'='*50}")
                print(f"Validation Results - Epoch {epoch + 1}")
                print(f"{'='*50}")
                print(f"Total Loss: {val_losses['total_loss']:.4f}")
                print(f"CE Loss: {val_losses['ce_loss']:.4f}")
                print(f"Dice Loss: {val_losses['dice_loss']:.4f}")
                print(f"mIoU: {miou:.4f} (excluding background)")
                print(f"{'='*50}\n")

                # Save checkpoint
                if (epoch + 1) % self.save_every == 0:
                    self.save_checkpoint(miou)

                    # Generate validation visualization
                    if self.visualizer is not None:
                        print("Generating validation visualization...")
                        self.visualizer.visualize_validation_images(epoch + 1)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)

            self.epoch += 1

        print("\nTraining completed!")
        self.writer.close()


def create_trainer(
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
    """Create trainer with all components."""
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_annotation_file=train_annotation_file,
        val_annotation_file=val_annotation_file,
        train_image_dir=train_image_dir,
        val_image_dir=val_image_dir,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
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
        mask_size=config['mask_size']
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
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Create scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )

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
        log_dir=config.get('log_dir', 'logs')
    )

    # Create and set validation visualizer
    from pycocotools.coco import COCO
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


if __name__ == "__main__":
    # Example configuration
    config = {
        'num_classes': 3,
        'in_channels': 1024,
        'mid_channels': 256,
        'mask_size': 56,
        'batch_size': 8,
        'num_workers': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_epochs': 100,
        'ce_weight': 1.0,
        'dice_weight': 1.0,
        'dice_classes': [1],  # Only compute Dice for target class
        'use_log_weights': True
    }

    # Load pixel ratios from data analysis
    with open('data_analyze_100.json', 'r') as f:
        data_stats = json.load(f)
    pixel_ratios = data_stats['pixel_ratios']

    # Create trainer
    trainer = create_trainer(
        config=config,
        train_annotation_file='data/annotations/instances_train2017_person_only_no_crowd_100.json',
        val_annotation_file='data/annotations/instances_val2017_person_only_no_crowd_100.json',
        train_image_dir='data/images/train2017',
        val_image_dir='data/images/val2017',
        onnx_model_path='ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
        pixel_ratios=pixel_ratios,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Training on device: {trainer.device}")
    print(f"Train dataset size: {len(trainer.train_loader.dataset)}")
    print(f"Val dataset size: {len(trainer.val_loader.dataset)}")

    # Start training
    # trainer.train(num_epochs=config['num_epochs'])