"""Main script for training ROI-based instance segmentation model."""

import argparse
import json
import torch
from pathlib import Path

from src.human_edge_detection.train import create_trainer


def main():
    parser = argparse.ArgumentParser(description='Train ROI-based instance segmentation model')

    # Data arguments
    parser.add_argument('--train-ann', type=str,
                        default='data/annotations/instances_train2017_person_only_no_crowd_100.json',
                        help='Training annotation file')
    parser.add_argument('--val-ann', type=str,
                        default='data/annotations/instances_val2017_person_only_no_crowd_100.json',
                        help='Validation annotation file')
    parser.add_argument('--train-img-dir', type=str,
                        default='data/images/train2017',
                        help='Training images directory')
    parser.add_argument('--val-img-dir', type=str,
                        default='data/images/val2017',
                        help='Validation images directory')
    parser.add_argument('--onnx-model', type=str,
                        default='ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
                        help='YOLO ONNX model path')
    parser.add_argument('--data-stats', type=str,
                        default='data_analyze_full.json',
                        help='Data statistics file')

    # Model arguments
    parser.add_argument('--num-classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--in-channels', type=int, default=1024,
                        help='Input channels from YOLO features')
    parser.add_argument('--mid-channels', type=int, default=256,
                        help='Middle channels in decoder')
    parser.add_argument('--mask-size', type=int, default=56,
                        help='Output mask size')

    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--ce-weight', type=float, default=1.0,
                        help='Cross-entropy loss weight')
    parser.add_argument('--dice-weight', type=float, default=1.0,
                        help='Dice loss weight')

    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='Tensorboard log directory')
    parser.add_argument('--val-output-dir', type=str, default='validation_results',
                        help='Validation visualization output directory')

    # Other arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true',
                        help='Only run validation')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"Using device: {args.device}")

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
        'num_classes': args.num_classes,
        'in_channels': args.in_channels,
        'mid_channels': args.mid_channels,
        'mask_size': args.mask_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_epochs': args.epochs,
        'ce_weight': args.ce_weight,
        'dice_weight': args.dice_weight,
        'dice_classes': [1],  # Only compute Dice for target class
        'use_log_weights': True,
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

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainer.scheduler and checkpoint.get('scheduler_state_dict'):
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        trainer.best_miou = checkpoint.get('best_miou', 0.0)
        print(f"  Starting from epoch {start_epoch + 1}")
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
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*50)

    trainer.train(num_epochs=args.epochs, start_epoch=start_epoch)

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