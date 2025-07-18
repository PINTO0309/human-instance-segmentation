"""Standalone validation script for trained checkpoints."""

import argparse
import json
import torch
from pathlib import Path
from typing import Dict, Optional

from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
from src.human_edge_detection.model import create_model, ROIBatchProcessor
from src.human_edge_detection.losses import create_loss_function
from src.human_edge_detection.visualize import ValidationVisualizer
from pycocotools.coco import COCO


def validate_checkpoint(
    checkpoint_path: str,
    val_annotation_file: str,
    val_image_dir: str,
    onnx_model_path: str,
    data_stats_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda',
    generate_visualization: bool = True,
    val_output_dir: str = 'validation_results',
    validate_all: bool = False,
    execution_provider: str = 'cuda'
) -> Dict[str, float]:
    """Run validation on a single checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        val_annotation_file: Path to validation annotations
        val_image_dir: Directory containing validation images
        onnx_model_path: Path to YOLO ONNX model
        data_stats_path: Path to data statistics JSON file
        batch_size: Batch size for validation
        num_workers: Number of data loader workers
        device: Device to run on (cuda/cpu)
        generate_visualization: Whether to generate visualization images
        val_output_dir: Output directory for visualizations
        validate_all: Whether to validate all images from the annotation file instead of just the 4 default test images
        execution_provider: Execution provider for ONNX Runtime (cpu/cuda/tensorrt)
        
    Returns:
        Dictionary containing validation metrics
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract epoch number from checkpoint
    epoch = checkpoint.get('epoch', 0)
    best_miou = checkpoint.get('best_miou', 0.0)
    checkpoint_miou = checkpoint.get('miou', 0.0)
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {epoch}")
    print(f"  Best mIoU: {best_miou:.4f}")
    print(f"  Checkpoint mIoU: {checkpoint_miou:.4f}")
    
    # Load data statistics
    with open(data_stats_path, 'r') as f:
        data_stats = json.load(f)
    pixel_ratios = data_stats['pixel_ratios']
    separation_aware_weights = data_stats.get('separation_aware_weights', None)
    
    print(f"\nData statistics:")
    print(f"  Background: {pixel_ratios['background']:.4f}")
    print(f"  Target: {pixel_ratios['target']:.4f}")
    print(f"  Non-target: {pixel_ratios['non_target']:.4f}")
    
    # Create validation data loader directly
    from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
    from torch.utils.data import DataLoader
    
    val_dataset = COCOInstanceSegmentationDataset(
        val_annotation_file,
        val_image_dir
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nValidation dataset size: {len(val_loader.dataset)} samples")
    
    # Setup providers based on execution_provider option
    if execution_provider == 'cpu':
        providers = ['CPUExecutionProvider']
    elif execution_provider == 'tensorrt':
        from src.human_edge_detection.feature_extractor import setup_tensorrt_providers
        providers = setup_tensorrt_providers(Path(onnx_model_path))
    else:  # cuda (default)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Create feature extractor
    feature_extractor = YOLOv9FeatureExtractor(
        onnx_path=onnx_model_path,
        device=device,
        providers=providers
    )
    
    # Create model and load weights
    model = create_model(
        num_classes=3,
        in_channels=1024,
        mid_channels=256,
        mask_size=56
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create loss function
    loss_fn = create_loss_function(
        pixel_ratios=pixel_ratios,
        use_log_weights=True,
        ce_weight=1.0,
        dice_weight=1.0,
        dice_classes=[1],
        device=device,
        separation_aware_weights=separation_aware_weights
    )
    
    # Create a minimal trainer just for validation
    # We'll use dummy components since we're not training
    dummy_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Import Trainer directly to avoid circular import issues
    from src.human_edge_detection.train_tutorial import Trainer
    
    trainer = Trainer(
        model=model,
        feature_extractor=feature_extractor,
        train_loader=val_loader,  # Dummy, not used
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=dummy_optimizer,
        device=device,
        checkpoint_dir='',  # Not used
        log_dir='',  # Not used
        save_every=1,  # Not used
        validate_every=1  # Not used
    )
    
    # Set epoch for proper visualization naming
    trainer.epoch = epoch - 1  # Trainer will add 1 in validate()
    
    # Run validation
    print("\nRunning validation...")
    val_losses, miou = trainer.validate()
    
    # Print results
    print(f"\nValidation Results:")
    print(f"  Total Loss: {val_losses['total_loss']:.4f}")
    print(f"  CE Loss: {val_losses['ce_loss']:.4f}")
    print(f"  Dice Loss: {val_losses['dice_loss']:.4f}")
    print(f"  mIoU: {miou:.4f} (excluding background)")
    
    # Generate visualization if requested
    if generate_visualization:
        print("\nGenerating validation visualization...")
        val_coco = COCO(val_annotation_file)
        visualizer = ValidationVisualizer(
            model=model,
            feature_extractor=feature_extractor,
            coco=val_coco,
            image_dir=val_image_dir,
            output_dir=val_output_dir,
            device=device
        )
        visualizer.visualize_validation_images(epoch, validate_all=validate_all)
        print(f"Visualization saved to: {val_output_dir}/")
    
    # Return metrics
    return {
        'epoch': epoch,
        'total_loss': val_losses['total_loss'],
        'ce_loss': val_losses['ce_loss'],
        'dice_loss': val_losses['dice_loss'],
        'miou': miou,
        'checkpoint_miou': checkpoint_miou,
        'best_miou': best_miou
    }


def validate_multiple_checkpoints(
    checkpoint_pattern: str,
    val_annotation_file: str,
    val_image_dir: str,
    onnx_model_path: str,
    data_stats_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda',
    generate_visualization: bool = False,
    val_output_dir: str = 'validation_results',
    validate_all: bool = False,
    execution_provider: str = 'cuda'
):
    """Validate multiple checkpoints matching a pattern.
    
    Args:
        checkpoint_pattern: Glob pattern for checkpoint files
        Other args same as validate_checkpoint
    """
    from glob import glob
    
    checkpoint_files = sorted(glob(checkpoint_pattern))
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to validate")
    
    results = []
    for checkpoint_path in checkpoint_files:
        print(f"\n{'='*60}")
        print(f"Validating: {Path(checkpoint_path).name}")
        print(f"{'='*60}")
        
        metrics = validate_checkpoint(
            checkpoint_path=checkpoint_path,
            val_annotation_file=val_annotation_file,
            val_image_dir=val_image_dir,
            onnx_model_path=onnx_model_path,
            data_stats_path=data_stats_path,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            generate_visualization=generate_visualization,
            val_output_dir=val_output_dir,
            validate_all=validate_all,
            execution_provider=execution_provider
        )
        
        metrics['checkpoint_file'] = Path(checkpoint_path).name
        results.append(metrics)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<50} {'Epoch':>6} {'mIoU':>8} {'Loss':>8}")
    print("-" * 75)
    
    for result in results:
        print(f"{result['checkpoint_file']:<50} {result['epoch']:>6} "
              f"{result['miou']:>8.4f} {result['total_loss']:>8.4f}")
    
    # Find best checkpoint
    best_result = max(results, key=lambda x: x['miou'])
    print(f"\nBest checkpoint: {best_result['checkpoint_file']}")
    print(f"  mIoU: {best_result['miou']:.4f}")
    print(f"  Epoch: {best_result['epoch']}")


def main():
    parser = argparse.ArgumentParser(description='Validate trained checkpoints')
    
    # Checkpoint argument
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file or glob pattern for multiple files')
    
    # Data arguments
    parser.add_argument('--val_ann', type=str,
                        default='data/annotations/instances_val2017_person_only_no_crowd_100.json',
                        help='Validation annotation file')
    parser.add_argument('--val_img_dir', type=str,
                        default='data/images/val2017',
                        help='Validation images directory')
    parser.add_argument('--onnx_model', type=str,
                        default='ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
                        help='YOLO ONNX model path')
    parser.add_argument('--data_stats', type=str,
                        default='data_analyze_100.json',
                        help='Data statistics file')
    
    # Validation arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Output arguments
    parser.add_argument('--no_visualization', action='store_true',
                        help='Skip generating visualization images')
    parser.add_argument('--val_output_dir', type=str, default='validation_results',
                        help='Validation visualization output directory')
    
    # Multiple checkpoint validation
    parser.add_argument('--multiple', action='store_true',
                        help='Validate multiple checkpoints using glob pattern')
    
    # All images validation
    parser.add_argument('--validate_all', action='store_true',
                        help='Validate all images from the annotation file instead of just the 4 default test images')
    
    # Execution provider
    parser.add_argument('--execution_provider', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'tensorrt'],
                        help='Execution provider for ONNX Runtime (default: cuda)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
        if args.execution_provider in ['cuda', 'tensorrt']:
            print("Switching execution provider to CPU")
            args.execution_provider = 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Using execution provider: {args.execution_provider}")
    
    if args.multiple:
        # Validate multiple checkpoints
        validate_multiple_checkpoints(
            checkpoint_pattern=args.checkpoint,
            val_annotation_file=args.val_ann,
            val_image_dir=args.val_img_dir,
            onnx_model_path=args.onnx_model,
            data_stats_path=args.data_stats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            generate_visualization=not args.no_visualization,
            val_output_dir=args.val_output_dir,
            validate_all=args.validate_all,
            execution_provider=args.execution_provider
        )
    else:
        # Validate single checkpoint
        if '*' in args.checkpoint or '?' in args.checkpoint:
            print("Error: Use --multiple flag for glob patterns")
            return
            
        validate_checkpoint(
            checkpoint_path=args.checkpoint,
            val_annotation_file=args.val_ann,
            val_image_dir=args.val_img_dir,
            onnx_model_path=args.onnx_model,
            data_stats_path=args.data_stats,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            generate_visualization=not args.no_visualization,
            val_output_dir=args.val_output_dir,
            validate_all=args.validate_all,
            execution_provider=args.execution_provider
        )


if __name__ == "__main__":
    main()