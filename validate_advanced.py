#!/usr/bin/env python3
"""Standalone validation script for models trained with run_experiments.py."""

import argparse
import json
import torch
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch.nn as nn

# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
from src.human_edge_detection.model import create_model
from src.human_edge_detection.losses import create_loss_function
from src.human_edge_detection.train_utils import evaluate_model
from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from torch.utils.data import DataLoader
from pycocotools.coco import COCO

# Import advanced components
from src.human_edge_detection.advanced.multi_scale_model import create_multiscale_model
from src.human_edge_detection.advanced.variable_roi_model import (
    create_variable_roi_model,
    create_rgb_enhanced_variable_roi_model
)
from src.human_edge_detection.advanced.distance_aware_loss import create_distance_aware_loss
from src.human_edge_detection.advanced.cascade_segmentation import create_cascade_model, CascadeLoss
from src.human_edge_detection.advanced.visualization_adapter import AdvancedValidationVisualizer
from src.human_edge_detection.experiments.config_manager import ExperimentConfig


def build_model(config: ExperimentConfig, device: str) -> Tuple[nn.Module, Optional[object]]:
    """Build model based on configuration.

    Returns:
        model: The segmentation model
        feature_extractor: Feature extractor (if using base model)
    """
    # Check for RGB hierarchical models first (they don't need YOLO feature extractor)
    if config.model.use_rgb_hierarchical:
        from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model
        
        # Determine if multi-scale fusion is needed
        multi_scale = (hasattr(config.model, 'variable_roi_sizes') and config.model.variable_roi_sizes) or config.multiscale.enabled
        
        model = create_rgb_hierarchical_model(
            num_classes=config.model.num_classes,
            roi_size=config.model.roi_size,
            mask_size=config.model.mask_size,
            use_multi_scale_fusion=multi_scale,
            fusion_method=config.multiscale.fusion_method if multi_scale else 'concat',
            use_attention_module=config.model.use_attention_module,
            # Binary mask refinement modules
            use_boundary_refinement=getattr(config.model, 'use_boundary_refinement', False),
            use_progressive_upsampling=getattr(config.model, 'use_progressive_upsampling', False),
            use_subpixel_conv=getattr(config.model, 'use_subpixel_conv', False),
            use_contour_detection=getattr(config.model, 'use_contour_detection', False),
            use_distance_transform=getattr(config.model, 'use_distance_transform', False),
            # Normalization configuration
            normalization_type=getattr(config.model, 'normalization_type', 'layernorm2d'),
            normalization_groups=getattr(config.model, 'normalization_groups', 8),
            # Pre-trained model configuration
            use_pretrained_unet=getattr(config.model, 'use_pretrained_unet', False),
            pretrained_weights_path=getattr(config.model, 'pretrained_weights_path', ''),
            freeze_pretrained_weights=getattr(config.model, 'freeze_pretrained_weights', False),
            use_full_image_unet=getattr(config.model, 'use_full_image_unet', False),
        )
        
        feature_extractor = None  # RGB model doesn't need external feature extractor
        
    # Check for hierarchical models 
    elif config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
        # Build multi-scale model with hierarchical architecture
        if config.model.variable_roi_sizes:
            # Create custom variable ROI model
            base_model = create_variable_roi_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                roi_sizes=config.model.variable_roi_sizes,
                num_classes=config.model.num_classes,
                mask_size=config.model.mask_size,
                execution_provider=config.model.execution_provider
            )
        else:
            # Create standard multi-scale model
            base_model = create_multiscale_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                num_classes=config.model.num_classes,
                roi_size=config.model.roi_size,
                mask_size=config.model.mask_size,
                fusion_method=config.multiscale.fusion_method,
                execution_provider=config.model.execution_provider
            )

        # Wrap with appropriate hierarchical architecture
        if hasattr(config.model, 'use_hierarchical_unet_v4') and config.model.use_hierarchical_unet_v4:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v4
            model = create_hierarchical_model_unet_v4(base_model)
        elif hasattr(config.model, 'use_hierarchical_unet_v3') and config.model.use_hierarchical_unet_v3:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v3
            model = create_hierarchical_model_unet_v3(base_model)
        elif hasattr(config.model, 'use_hierarchical_unet_v2') and config.model.use_hierarchical_unet_v2:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v2
            model = create_hierarchical_model_unet_v2(base_model)
        elif hasattr(config.model, 'use_hierarchical_unet') and config.model.use_hierarchical_unet:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet
            model = create_hierarchical_model_unet(base_model)
        else:
            from src.human_edge_detection.advanced.hierarchical_segmentation import create_hierarchical_model
            model = create_hierarchical_model(base_model)

        feature_extractor = None

    elif config.multiscale.enabled:
        # Check if variable ROI sizes are specified
        if config.model.variable_roi_sizes:
            # Check if RGB enhancement is enabled
            if config.model.use_rgb_enhancement:
                # RGB enhanced variable ROI model
                model = create_rgb_enhanced_variable_roi_model(
                    onnx_model_path=config.model.onnx_model,
                    target_layers=config.multiscale.target_layers,
                    roi_sizes=config.model.variable_roi_sizes,
                    num_classes=config.model.num_classes,
                    mask_size=config.model.mask_size,
                    execution_provider=config.model.execution_provider,
                    rgb_enhanced_layers=getattr(config.model, 'rgb_enhanced_layers', ['layer_34']),
                    use_rgb_enhancement=True
                )
            else:
                # Standard variable ROI model
                model = create_variable_roi_model(
                    onnx_model_path=config.model.onnx_model,
                    target_layers=config.multiscale.target_layers,
                    roi_sizes=config.model.variable_roi_sizes,
                    num_classes=config.model.num_classes,
                    mask_size=config.model.mask_size,
                    execution_provider=config.model.execution_provider
                )
        else:
            # Standard multi-scale model
            model = create_multiscale_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                num_classes=config.model.num_classes,
                roi_size=config.model.roi_size,
                mask_size=config.model.mask_size,
                fusion_method=config.multiscale.fusion_method,
                execution_provider=config.model.execution_provider
            )

        # Apply cascade if enabled
        if config.cascade.enabled:
            model = create_cascade_model(
                base_model=model,
                num_classes=config.model.num_classes,
                cascade_stages=config.cascade.num_stages,
                share_features=config.cascade.share_features
            )

        feature_extractor = None  # Integrated in model

    else:
        # Base single-scale model
        feature_extractor = YOLOv9FeatureExtractor(
            onnx_path=config.model.onnx_model
        )

        model = create_model(
            num_classes=config.model.num_classes,
            roi_size=config.model.roi_size,
            mask_size=config.model.mask_size
        )

    return model.to(device), feature_extractor


def build_loss_function(
    config: ExperimentConfig,
    data_stats: Optional[Dict] = None,
    device: str = 'cuda'
) -> nn.Module:
    """Build loss function based on configuration."""

    # Check for hierarchical model first
    if config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
        # Check if we need refined hierarchical loss
        if any(getattr(config.model, attr, False) for attr in ['use_boundary_refinement', 'use_active_contour_loss', 'use_progressive_upsampling', 
                                                                 'use_subpixel_conv', 'use_contour_detection', 'use_distance_transform', 'use_boundary_aware_loss']):
            from src.human_edge_detection.advanced.hierarchical_segmentation_refinement import RefinedHierarchicalLoss
            
            return RefinedHierarchicalLoss(
                bg_weight=1.5,
                fg_weight=1.5,
                target_weight=1.2,
                consistency_weight=0.3,
                use_dynamic_weights=False,
                dice_weight=config.training.dice_weight,
                ce_weight=config.training.ce_weight,
                # Refinement parameters
                active_contour_weight=0.1,
                boundary_aware_weight=0.1,
                contour_loss_weight=0.1,
                distance_loss_weight=0.1,
                use_active_contour_loss=getattr(config.model, 'use_active_contour_loss', False),
                use_boundary_aware_loss=getattr(config.model, 'use_boundary_aware_loss', False),
                use_contour_detection=getattr(config.model, 'use_contour_detection', False),
                use_distance_transform=getattr(config.model, 'use_distance_transform', False),
            )
        else:
            from src.human_edge_detection.advanced.hierarchical_segmentation import HierarchicalLoss
            return HierarchicalLoss(
                bg_weight=1.0,
                fg_weight=1.5,
                target_weight=2.0,
                consistency_weight=0.1
            )

    # Extract pixel ratios and separation aware weights from data_stats
    pixel_ratios = None
    separation_aware_weights = None
    if data_stats:
        pixel_ratios = data_stats.get('pixel_ratios', None)
        separation_aware_weights = data_stats.get('separation_aware_weights', None)

    if config.distance_loss.enabled:
        # Distance-aware loss
        loss_fn = create_distance_aware_loss(
            pixel_ratios=pixel_ratios,
            boundary_width=config.distance_loss.boundary_width,
            boundary_weight=config.distance_loss.boundary_weight,
            instance_sep_weight=config.distance_loss.instance_sep_weight,
            ce_weight=config.training.ce_weight,
            dice_weight=config.training.dice_weight,
            adaptive=config.distance_loss.adaptive,
            device=device,
            separation_aware_weights=separation_aware_weights,
            use_focal=config.training.use_focal,
            focal_gamma=config.training.focal_gamma
        )

    elif config.cascade.enabled and config.multiscale.enabled:
        # Cascade loss
        base_loss = create_loss_function(
            pixel_ratios=pixel_ratios,
            ce_weight=config.training.ce_weight,
            dice_weight=config.training.dice_weight,
            device=device,
            separation_aware_weights=separation_aware_weights,
            use_focal=config.training.use_focal,
            focal_gamma=config.training.focal_gamma
        )
        loss_fn = CascadeLoss(
            base_loss=base_loss,
            stage_weights=config.cascade.stage_weights
        )

    else:
        # Standard loss
        loss_fn = create_loss_function(
            pixel_ratios=pixel_ratios,
            ce_weight=config.training.ce_weight,
            dice_weight=config.training.dice_weight,
            device=device,
            separation_aware_weights=separation_aware_weights,
            use_focal=config.training.use_focal,
            focal_gamma=config.training.focal_gamma
        )

    return loss_fn


def validate_advanced_checkpoint(
    checkpoint_path: str,
    device: str = 'cuda',
    generate_visualization: bool = True,
    validate_all: bool = False,
    override_config: Optional[Dict] = None
) -> Dict[str, float]:
    """Validate a checkpoint from run_experiments.py.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to run on (cuda/cpu)
        generate_visualization: Whether to generate visualization images
        validate_all: Whether to validate all images or just default test images
        override_config: Optional config overrides

    Returns:
        Dictionary containing validation metrics
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration from checkpoint
    config_dict = checkpoint['config']
    if override_config:
        # Apply overrides
        for key, value in override_config.items():
            keys = key.split('.')
            d = config_dict
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    # Create config object
    config = ExperimentConfig.from_dict(config_dict)

    # Get experiment directory
    checkpoint_path = Path(checkpoint_path)
    exp_dir = checkpoint_path.parent.parent

    print(f"\nExperiment: {config.name}")
    print(f"Description: {config.description}")
    print(f"Features enabled:")
    print(f"  - Multi-scale: {config.multiscale.enabled}")
    print(f"  - Distance loss: {config.distance_loss.enabled}")
    print(f"  - Focal loss: {config.training.use_focal} (gamma={config.training.focal_gamma})")
    print(f"  - Cascade: {config.cascade.enabled}")
    print(f"  - Relational: {config.relational.enabled}")
    if hasattr(config.model, 'use_rgb_enhancement') and config.model.use_rgb_enhancement:
        enhanced_layers = getattr(config.model, 'rgb_enhanced_layers', ['layer_34'])
        print(f"  - RGB enhancement: {config.model.use_rgb_enhancement} (layers: {', '.join(enhanced_layers)})")

    # Build model
    print("\nBuilding model...")
    model, feature_extractor = build_model(config, device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data statistics
    data_stats = None
    if config.data.data_stats:
        data_stats_path = Path(config.data.data_stats)
        if data_stats_path.exists():
            print(f"Loading data statistics from {data_stats_path}")
            with open(data_stats_path, 'r') as f:
                data_stats = json.load(f)

    # Build loss function
    loss_fn = build_loss_function(config, data_stats, device)

    # Create validation dataset
    print("\nLoading validation dataset...")
    val_dataset = COCOInstanceSegmentationDataset(
        annotation_file=config.data.val_annotation,
        image_dir=config.data.val_img_dir,
        transform=None  # No augmentation for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=create_collate_fn()
    )

    print(f"Validation samples: {len(val_dataset)}")

    # Run validation
    print("\nRunning validation...")
    val_metrics = evaluate_model(
        model, val_loader, loss_fn, device,
        feature_extractor=feature_extractor,
        config=config
    )

    # Print results
    print(f"\nValidation Results:")
    print(f"  Total Loss: {val_metrics['total_loss']:.4f}")
    print(f"  CE Loss: {val_metrics['ce_loss']:.4f}")
    print(f"  Dice Loss: {val_metrics['dice_loss']:.4f}")
    print(f"  mIoU: {val_metrics['miou']:.4f}")

    for i in range(config.model.num_classes):
        print(f"  IoU Class {i}: {val_metrics[f'iou_class_{i}']:.4f}")

    # Generate visualization if requested
    if generate_visualization:
        print("\nGenerating validation visualization...")
        val_coco = COCO(config.data.val_annotation)

        # Create output directory
        vis_output_dir = exp_dir / 'validation_results'
        vis_output_dir.mkdir(exist_ok=True)

        # Use HierarchicalUNetVisualizer for hierarchical UNet models
        if any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
            from src.human_edge_detection.advanced.hierarchical_unet_visualizer import HierarchicalUNetVisualizer
            visualizer = HierarchicalUNetVisualizer(
                model=model,
                feature_extractor=feature_extractor,
                coco=val_coco,
                image_dir=config.data.val_img_dir,
                output_dir=vis_output_dir,
                device=device,
                is_multiscale=config.multiscale.enabled,
                roi_padding=config.data.roi_padding
            )
        else:
            visualizer = AdvancedValidationVisualizer(
                model=model,
                feature_extractor=feature_extractor,
                coco=val_coco,
                image_dir=config.data.val_img_dir,
                output_dir=vis_output_dir,
                device=device,
                is_multiscale=config.multiscale.enabled,
                roi_padding=config.data.roi_padding
            )

        # Use epoch from checkpoint or -1 for display
        epoch = checkpoint.get('epoch', -1)
        visualizer.visualize_validation_images(epoch, validate_all=validate_all)

    return val_metrics


def main():
    parser = argparse.ArgumentParser(description='Validate models trained with run_experiments.py')

    parser.add_argument('checkpoint', type=str,
                        help='Path to checkpoint file (.pth)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on (cuda/cpu)')
    parser.add_argument('--no_visualization', action='store_true',
                        help='Skip generating visualization images')
    parser.add_argument('--validate_all', action='store_true',
                        help='Validate all images instead of default test images')
    parser.add_argument('--override', type=str, action='append',
                        help='Override config values (e.g., --override data.batch_size=16)')

    args = parser.parse_args()

    # Parse config overrides
    override_config = {}
    if args.override:
        for override in args.override:
            key, value = override.split('=')
            # Try to parse value as JSON, otherwise keep as string
            try:
                value = json.loads(value)
            except:
                pass
            override_config[key] = value

    # Run validation
    metrics = validate_advanced_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        generate_visualization=not args.no_visualization,
        validate_all=args.validate_all,
        override_config=override_config
    )

    # Save metrics
    checkpoint_path = Path(args.checkpoint)
    metrics_path = checkpoint_path.parent / f'{checkpoint_path.stem}_validation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved validation metrics to {metrics_path}")


if __name__ == '__main__':
    main()