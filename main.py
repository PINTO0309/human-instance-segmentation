"""Main script for training ROI-based instance segmentation model."""

import argparse
import json
import torch
from pathlib import Path

from src.human_edge_detection.train import create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train ROI-based instance segmentation model')

    # Data arguments
    parser.add_argument('--train_ann', type=str,
                        default='data/annotations/instances_train2017_person_only_no_crowd_100.json',
                        help='Training annotation file')
    parser.add_argument('--val_ann', type=str,
                        default='data/annotations/instances_val2017_person_only_no_crowd_100.json',
                        help='Validation annotation file')
    parser.add_argument('--train_img_dir', type=str,
                        default='data/images/train2017',
                        help='Training images directory')
    parser.add_argument('--val_img_dir', type=str,
                        default='data/images/val2017',
                        help='Validation images directory')
    parser.add_argument('--onnx_model', type=str,
                        default='ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
                        help='YOLO ONNX model path')
    parser.add_argument('--data_stats', type=str,
                        default='data_analyze_full.json',
                        help='Data statistics file')

    # Model arguments
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=1024,
                        help='Input channels from YOLO features')
    parser.add_argument('--mid_channels', type=int, default=256,
                        help='Middle channels in decoder')
    parser.add_argument('--mask_size', type=int, default=56,
                        help='Output mask size')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                        help='Cross-entropy loss weight')
    parser.add_argument('--dice_weight', type=float, default=1.0,
                        help='Dice loss weight')
    parser.add_argument('--dice_classes', type=int, nargs='+', default=[1],
                        help='Classes to compute Dice loss for (default: [1])')
    parser.add_argument('--use_log_weights', action='store_true', default=True,
                        help='Use log weights for class balancing')
    parser.add_argument('--no_log_weights', dest='use_log_weights', action='store_false',
                        help='Disable log weights for class balancing')
    
    # Optimizer and scheduler arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'exponential', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_decay_epochs', type=int, nargs='+', default=[30, 60, 90],
                        help='Epochs to decay learning rate for step scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='Learning rate decay rate')
    
    # Dataset arguments
    parser.add_argument('--image_size', type=int, nargs=2, default=[640, 640],
                        help='Input image size (height width)')
    parser.add_argument('--roi_size', type=int, default=28,
                        help='ROI size after ROIAlign')
    parser.add_argument('--min_roi_size', type=int, default=16,
                        help='Minimum ROI size in pixels')
    parser.add_argument('--feature_stride', type=int, default=8,
                        help='Feature stride of backbone')
    
    # Training control arguments
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--validate_every', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--gradient_clip', type=float, default=0.0,
                        help='Gradient clipping value (0 to disable)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Tensorboard log directory')
    parser.add_argument('--val_output_dir', type=str, default='validation_results',
                        help='Validation visualization output directory')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--resume_epochs', type=int, default=None,
                        help='Number of additional epochs to train when resuming (overrides --epochs)')
    parser.add_argument('--test_only', action='store_true',
                        help='Only run validation')

    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}")
    print(f"Random seed: {args.seed}")

    # Load data statistics
    with open(args.data_stats, 'r') as f:
        data_stats = json.load(f)
    pixel_ratios = data_stats['pixel_ratios']
    
    # Get separation-aware weights if available
    separation_aware_weights = data_stats.get('separation_aware_weights', None)

    print(f"\nData statistics:")
    print(f"  Background: {pixel_ratios['background']:.4f}")
    print(f"  Target: {pixel_ratios['target']:.4f}")
    print(f"  Non-target: {pixel_ratios['non_target']:.4f}")
    
    if separation_aware_weights:
        print(f"\nUsing separation-aware weights:")
        print(f"  Background: {separation_aware_weights['background']:.4f}")
        print(f"  Target: {separation_aware_weights['target']:.4f}")
        print(f"  Non-target: {separation_aware_weights['non_target']:.4f}")

    # Create configuration
    config = {
        # Model parameters
        'num_classes': args.num_classes,
        'in_channels': args.in_channels,
        'mid_channels': args.mid_channels,
        'mask_size': args.mask_size,
        'roi_size': args.roi_size,
        'feature_stride': args.feature_stride,
        
        # Dataset parameters
        'image_size': tuple(args.image_size),
        'min_roi_size': args.min_roi_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        
        # Training parameters
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'ce_weight': args.ce_weight,
        'dice_weight': args.dice_weight,
        'dice_classes': args.dice_classes,
        'use_log_weights': args.use_log_weights,
        
        # Optimizer/Scheduler parameters
        'optimizer': args.optimizer,
        'momentum': args.momentum,
        'scheduler': args.scheduler,
        'min_lr': args.min_lr,
        'lr_decay_epochs': args.lr_decay_epochs,
        'lr_decay_rate': args.lr_decay_rate,
        
        # Training control
        'save_every': args.save_every,
        'validate_every': args.validate_every,
        'gradient_clip': args.gradient_clip,
        
        # Output directories
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'validation_output_dir': args.val_output_dir
    }

    # Create trainer
    trainer = create_trainer(
        config=config,
        train_annotation_file=args.train_ann,
        val_annotation_file=args.val_ann,
        train_image_dir=args.train_img_dir,
        val_image_dir=args.val_img_dir,
        onnx_model_path=args.onnx_model,
        pixel_ratios=pixel_ratios,
        device=args.device,
        separation_aware_weights=separation_aware_weights
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(trainer.train_loader.dataset)} samples")
    print(f"  Validation: {len(trainer.val_loader.dataset)} samples")
    
    print(f"\nTraining configuration:")
    print(f"  Optimizer: {args.optimizer} (lr={args.lr}, weight_decay={args.weight_decay})")
    if args.optimizer == 'sgd':
        print(f"  Momentum: {args.momentum}")
    print(f"  Scheduler: {args.scheduler}")
    if args.scheduler == 'cosine':
        print(f"    Min LR: {args.min_lr}")
    elif args.scheduler == 'step':
        print(f"    Decay epochs: {args.lr_decay_epochs}")
        print(f"    Decay rate: {args.lr_decay_rate}")
    print(f"  Gradient clipping: {args.gradient_clip if args.gradient_clip > 0 else 'disabled'}")
    print(f"  Save every: {args.save_every} epochs")
    print(f"  Validate every: {args.validate_every} epochs")

    # Resume from checkpoint if specified
    start_epoch = 0
    total_epochs = args.epochs
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        trainer.best_miou = checkpoint.get('best_miou', 0.0)
        
        # Calculate total epochs based on resume_epochs or original epochs
        if args.resume_epochs is not None:
            total_epochs = start_epoch + args.resume_epochs
            print(f"  Resuming from epoch {start_epoch + 1}")
            print(f"  Training for {args.resume_epochs} additional epochs (until epoch {total_epochs})")
        else:
            print(f"  Resuming from epoch {start_epoch + 1}")
            print(f"  Training until epoch {total_epochs}")
        
        print(f"  Best mIoU so far: {trainer.best_miou:.4f}")

    # Test only mode
    if args.test_only:
        print("\nRunning validation only...")
        val_losses, miou = trainer.validate()
        print(f"Validation - Loss: {val_losses['total_loss']:.4f}, "
              f"CE: {val_losses['ce_loss']:.4f}, "
              f"Dice: {val_losses['dice_loss']:.4f}, "
              f"mIoU: {miou:.4f}")

        # Generate visualization
        if trainer.visualizer:
            print("Generating validation visualization...")
            trainer.visualizer.visualize_validation_images(start_epoch)
        return

    # Start training
    if args.resume:
        print(f"\nContinuing training for {total_epochs - start_epoch} epochs...")
    else:
        print(f"\nStarting training for {total_epochs} epochs...")
    print("="*50)

    trainer.train(num_epochs=total_epochs, start_epoch=start_epoch)

    print("\nTraining completed!")
    print(f"Best mIoU achieved: {trainer.best_miou:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print(f"Validation visualizations saved in: {args.val_output_dir}")

    # Export best model to ONNX
    best_checkpoint = Path(args.checkpoint_dir) / 'best_model.pth'
    if best_checkpoint.exists():
        print("\nExporting best model to ONNX...")
        from src.human_edge_detection.export_onnx import export_checkpoint_to_onnx

        onnx_path = Path(args.checkpoint_dir) / 'best_model.onnx'
        success = export_checkpoint_to_onnx(
            checkpoint_path=str(best_checkpoint),
            output_path=str(onnx_path),
            config=config,
            device=args.device
        )

        if success:
            print(f"ONNX model saved to: {onnx_path}")


if __name__ == "__main__":
    main()