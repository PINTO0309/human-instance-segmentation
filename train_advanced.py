"""Advanced training script with configurable features."""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import warnings
import logging

# Setup logger
logger = logging.getLogger(__name__)

def compute_gradient_norm(model):
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** 0.5
    return total_norm

def check_for_nan_gradients(model):
    """Check if any gradient contains NaN."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True, name
    return False, None


# Suppress FutureWarning messages
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler
from tqdm import tqdm

# Import base components
from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
from src.human_edge_detection.dataset_adapter import create_collate_fn
from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
from src.human_edge_detection.model import create_model, ROIBatchProcessor
from src.human_edge_detection.losses import create_loss_function
from src.human_edge_detection.train_utils import evaluate_model
from src.human_edge_detection.visualize import ValidationVisualizer

# Import advanced components
from src.human_edge_detection.advanced.multi_scale_extractor import MultiScaleYOLOFeatureExtractor
from src.human_edge_detection.advanced.multi_scale_model import create_multiscale_model
from src.human_edge_detection.advanced.variable_roi_model import (
    create_variable_roi_model, 
    create_rgb_enhanced_variable_roi_model
)
from src.human_edge_detection.advanced.distance_aware_loss import create_distance_aware_loss
from src.human_edge_detection.advanced.cascade_segmentation import create_cascade_model, CascadeLoss
from src.human_edge_detection.advanced.hierarchical_segmentation import HierarchicalLoss
from src.human_edge_detection.advanced.auxiliary_fg_bg_task import (
    MultiTaskSegmentationModel, MultiTaskLoss, AuxiliaryFgBgHead
)

# Import experiment management
from src.human_edge_detection.experiments.config_manager import (
    ExperimentConfig, ConfigManager, create_experiment_dirs
)
from src.human_edge_detection.text_logger import TextLogger


def build_model(config: ExperimentConfig, device: str) -> Tuple[nn.Module, Optional[object]]:
    """Build model based on configuration.

    Returns:
        model: The segmentation model
        feature_extractor: Feature extractor (if using base model)
    """
    # Check for new architecture flags first
    if config.model.use_hierarchical:
        # Build base model first
        if config.model.variable_roi_sizes:
            # Variable ROI model as base
            base_model = create_variable_roi_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                roi_sizes=config.model.variable_roi_sizes,
                num_classes=config.model.num_classes,
                mask_size=config.model.mask_size,
                execution_provider=config.model.execution_provider
            )
        else:
            # Standard multi-scale model as base
            base_model = create_multiscale_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                num_classes=config.model.num_classes,
                roi_size=config.model.roi_size,
                mask_size=config.model.mask_size,
                fusion_method=config.multiscale.fusion_method,
                execution_provider=config.model.execution_provider
            )
        
        # Wrap with hierarchical architecture
        from src.human_edge_detection.advanced.hierarchical_segmentation import create_hierarchical_model
        model = create_hierarchical_model(base_model)
        feature_extractor = None
        
    elif config.model.use_rgb_hierarchical:
        # RGB-based hierarchical model without YOLOv9 features
        from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model
        
        multi_scale = config.multiscale.enabled and config.model.variable_roi_sizes
        
        model = create_rgb_hierarchical_model(
            roi_size=config.model.roi_size,
            mask_size=config.model.mask_size,
            multi_scale=multi_scale,
            roi_sizes=config.model.variable_roi_sizes if multi_scale else None,
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
        )
        
        feature_extractor = None  # RGB model doesn't need external feature extractor
        
    elif config.model.use_hierarchical_unet or config.model.use_hierarchical_unet_v2 or config.model.use_hierarchical_unet_v3 or config.model.use_hierarchical_unet_v4:
        # Build multi-scale model with UNet-based hierarchical architecture
        
        # Check if we should use external features (no integrated feature extractor)
        if config.model.use_external_features:
            # Check if variable ROI sizes are specified
            if config.model.variable_roi_sizes:
                # Create variable ROI model that accepts pre-extracted features
                from src.human_edge_detection.advanced.variable_roi_head_only import create_variable_roi_head_only
                base_model = create_variable_roi_head_only(
                    target_layers=config.multiscale.target_layers,
                    roi_sizes=config.model.variable_roi_sizes,
                    num_classes=config.model.num_classes,
                    mask_size=config.model.mask_size,
                    return_features=True  # Need features for hierarchical head
                )
            else:
                # Create fixed ROI model that accepts pre-extracted features
                from src.human_edge_detection.advanced.multi_scale_head_only import create_multiscale_head_only
                base_model = create_multiscale_head_only(
                    target_layers=config.multiscale.target_layers,
                    num_classes=config.model.num_classes,
                    roi_size=config.model.roi_size,
                    mask_size=config.model.mask_size,
                    fusion_method=config.multiscale.fusion_method,
                    return_features=True  # Need features for hierarchical head
                )
        elif config.model.variable_roi_sizes:
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
        
        # Wrap with appropriate UNet-based hierarchical architecture
        if config.model.use_hierarchical_unet_v4:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v4
            model = create_hierarchical_model_unet_v4(base_model)
        elif config.model.use_hierarchical_unet_v3:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v3
            model = create_hierarchical_model_unet_v3(base_model)
        elif config.model.use_hierarchical_unet_v2:
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet_v2
            model = create_hierarchical_model_unet_v2(base_model)
        else:  # config.model.use_hierarchical_unet (V1)
            from src.human_edge_detection.advanced.hierarchical_segmentation_unet import create_hierarchical_model_unet
            model = create_hierarchical_model_unet(base_model)
        
        # If using external features, create a separate feature extractor
        if config.model.use_external_features:
            from src.human_edge_detection.advanced.multi_scale_extractor import MultiScaleYOLOFeatureExtractor
            feature_extractor = MultiScaleYOLOFeatureExtractor(
                model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                execution_provider=config.model.execution_provider
            )
        else:
            feature_extractor = None
        
    elif config.model.use_class_specific_decoder:
        # Build model with class-specific decoder
        if config.model.variable_roi_sizes:
            # Create custom variable ROI model with class-specific decoder
            from src.human_edge_detection.advanced.class_specific_decoder import ClassBalancedSegmentationHead
            
            # Create base model
            base_model = create_variable_roi_model(
                onnx_model_path=config.model.onnx_model,
                target_layers=config.multiscale.target_layers,
                roi_sizes=config.model.variable_roi_sizes,
                num_classes=config.model.num_classes,
                mask_size=config.model.mask_size,
                execution_provider=config.model.execution_provider
            )
            
            # Replace segmentation head with class-balanced version
            old_head = base_model.segmentation_head
            base_model.segmentation_head = ClassBalancedSegmentationHead(
                in_channels=256,  # Assuming standard fusion output
                num_classes=config.model.num_classes,
                mask_size=config.model.mask_size,
                use_class_specific=True
            )
            # Copy necessary components
            base_model.segmentation_head.var_roi_align = old_head.var_roi_align
            base_model.segmentation_head.feature_fusion = old_head.feature_fusion
            
            model = base_model
        else:
            raise NotImplementedError("Class-specific decoder only implemented for variable ROI models")
        
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

    # Apply auxiliary task wrapper if enabled
    if config.auxiliary_task.enabled:
        # Check if model is already a hierarchical model (which has its own auxiliary outputs)
        is_hierarchical = config.model.use_hierarchical or any(
            getattr(config.model, attr, False) 
            for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 
                         'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4',
                         'use_rgb_hierarchical']
        )
        
        if not is_hierarchical:
            # Only wrap non-hierarchical models
            # Get feature channels based on model type
            if config.multiscale.enabled:
                # Multi-scale model feature channels
                feature_channels = config.multiscale.fusion_channels
            elif config.model.variable_roi_sizes is not None:
                # Variable ROI model feature channels
                if config.model.use_rgb_enhancement:
                    feature_channels = 1024  # RGB enhanced
                else:
                    feature_channels = 1024  # Standard variable ROI
            else:
                # Standard model feature channels
                feature_channels = 1024  # From YOLO extractor
            
            # Wrap model with auxiliary task
            model = MultiTaskSegmentationModel(
                base_segmentation_head=model,
                in_channels=feature_channels,
                mask_size=config.model.mask_size,
                aux_weight=config.auxiliary_task.weight
            )
        else:
            print("Note: Hierarchical models have built-in auxiliary outputs, skipping MultiTaskSegmentationModel wrapper")

    return model.to(device), feature_extractor


def build_loss_function(
    config: ExperimentConfig,
    pixel_ratios: dict,
    device: str,
    separation_aware_weights: Optional[dict] = None
) -> nn.Module:
    """Build loss function based on configuration."""
    
    # Check for hierarchical model first (including RGB hierarchical)
    if config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
        # Check if refinement modules are enabled
        use_refinement = any([
            getattr(config.model, 'use_active_contour_loss', False),
            getattr(config.model, 'use_boundary_aware_loss', False),
            getattr(config.model, 'use_contour_detection', False),
            getattr(config.model, 'use_distance_transform', False),
        ])
        
        # Check if this is the fixed weights config
        use_dynamic_weights = config.name != 'hierarchical_unet_v2_fixed_weights'
        
        if use_refinement:
            from src.human_edge_detection.advanced.hierarchical_segmentation_refinement import RefinedHierarchicalLoss
            
            return RefinedHierarchicalLoss(
                bg_weight=1.5,
                fg_weight=1.5,
                target_weight=1.2,
                consistency_weight=0.3,
                use_dynamic_weights=use_dynamic_weights,
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
                bg_weight=1.5,
                fg_weight=1.5,
                target_weight=1.2,
                consistency_weight=0.3,
                use_dynamic_weights=use_dynamic_weights,
                dice_weight=config.training.dice_weight,
                ce_weight=config.training.ce_weight
            )

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

        # Wrap in cascade loss if needed
        # Note: Only use CascadeLoss if model actually supports cascade
        # Multi-scale models don't support cascade yet
        if config.cascade.enabled and not config.multiscale.enabled:
            loss_fn = CascadeLoss(
                base_loss=loss_fn,
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

    # Wrap with multi-task loss if auxiliary task is enabled
    if config.auxiliary_task.enabled:
        loss_fn = MultiTaskLoss(
            main_loss_fn=loss_fn,
            aux_weight=config.auxiliary_task.weight,
            aux_pos_weight=config.auxiliary_task.pos_weight
        )

    return loss_fn


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    config: ExperimentConfig,
    feature_extractor: Optional[object] = None,
    scaler: Optional[GradScaler] = None,
    writer: Optional[SummaryWriter] = None,
    text_logger: Optional[TextLogger] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    aux_fg_bg_loss = 0
    aux_fg_accuracy = 0
    aux_fg_iou = 0
    # Refinement losses
    active_contour_loss = 0
    boundary_aware_loss = 0
    contour_loss = 0
    distance_transform_loss = 0
    num_batches = 0
    grad_norm_before_clip = 0.0  # Initialize for metrics

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1} - Training', dynamic_ncols=True, leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        images = batch['image'].to(device)
        rois = batch['roi_boxes'].to(device)
        masks = batch['roi_masks'].to(device)

        # Extract features if using base model
        if feature_extractor is not None:
            with torch.no_grad():
                features = feature_extractor.extract_features(images)
        else:
            features = images  # Multi-scale model extracts internally

        optimizer.zero_grad()

        # Mixed precision training
        if config.training.mixed_precision and scaler is not None:
            with torch.amp.autocast('cuda'):
                # Forward pass
                if config.model.use_rgb_hierarchical:
                    # RGB hierarchical model takes full images and ROIs
                    predictions = model(images, rois)
                elif config.multiscale.enabled:
                    # Multi-scale model
                    if config.cascade.enabled:
                        # Check if model supports return_all_stages
                        if hasattr(model, 'cascade_head'):
                            predictions = model(features, rois, return_all_stages=True)
                        else:
                            # Model doesn't support cascade (e.g., when using multi-scale)
                            predictions = model(features, rois)
                    else:
                        predictions = model(features, rois)
                else:
                    # Base model
                    output = model(features=features, rois=rois)
                    predictions = output['masks']

                # Compute loss
                instance_info = batch.get('instance_info') if config.distance_loss.enabled else None

                # Handle hierarchical model output
                if config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
                    if isinstance(predictions, tuple):
                        logits, aux_outputs = predictions
                        loss, loss_dict = loss_fn(logits, masks, aux_outputs)
                    else:
                        # Hierarchical model should return tuple, but didn't
                        print(f"WARNING: Hierarchical model returned {type(predictions)} instead of tuple")
                        print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                        # This should not happen, raise error for debugging
                        raise RuntimeError("Hierarchical model must return (logits, aux_outputs) tuple")
                elif config.cascade.enabled and isinstance(predictions, tuple):
                    # Cascade returns multiple stages
                    loss, loss_dict = loss_fn(predictions, masks, instance_info)
                else:
                    # Check if distance loss is enabled
                    if config.distance_loss.enabled:
                        loss, loss_dict = loss_fn(predictions, masks, instance_info)
                    else:
                        loss, loss_dict = loss_fn(predictions, masks)

            # Backward pass with mixed precision
            scaler.scale(loss).backward()

            # Gradient clipping
            if config.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if feature_extractor is None else model.parameters(),
                    config.training.gradient_clip
                )

            scaler.step(optimizer)
            scaler.update()

        else:
            # Standard training
            if config.model.use_rgb_hierarchical:
                # RGB hierarchical model takes full images and ROIs
                predictions = model(images, rois)
            elif config.multiscale.enabled:
                if config.cascade.enabled:
                    # Check if model supports return_all_stages
                    if hasattr(model, 'cascade_head'):
                        predictions = model(features, rois, return_all_stages=True)
                    else:
                        # Model doesn't support cascade (e.g., when using multi-scale)
                        predictions = model(features, rois)
                else:
                    predictions = model(features, rois)
            else:
                output = model(features=features, rois=rois)
                predictions = output['masks']

            instance_info = batch.get('instance_info') if config.distance_loss.enabled else None

            # Handle hierarchical model output
            if config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
                if isinstance(predictions, tuple):
                    logits, aux_outputs = predictions
                    loss, loss_dict = loss_fn(logits, masks, aux_outputs)
                else:
                    # Hierarchical model should return tuple, but didn't
                    print(f"WARNING: Hierarchical model returned {type(predictions)} instead of tuple")
                    print(f"Predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'N/A'}")
                    # This should not happen, raise error for debugging
                    raise RuntimeError("Hierarchical model must return (logits, aux_outputs) tuple")
            elif config.cascade.enabled and isinstance(predictions, tuple):
                # Cascade returns multiple stages
                loss, loss_dict = loss_fn(predictions, masks, instance_info)
            else:
                # Check if distance loss is enabled
                if config.distance_loss.enabled:
                    loss, loss_dict = loss_fn(predictions, masks, instance_info)
                else:
                    loss, loss_dict = loss_fn(predictions, masks)

            # Check for NaN before backward


            if torch.isnan(loss):


                logger.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")


                logger.warning(f"Loss components: {loss_dict}")


                # Skip this batch


                optimizer.zero_grad()


                continue


                


            loss.backward()


            


            # Check gradients after backward


            grad_norm_before_clip = compute_gradient_norm(model)


            has_nan, nan_param = check_for_nan_gradients(model)


            if has_nan:


                logger.warning(f"NaN gradient detected in {nan_param}")


                optimizer.zero_grad()


                continue

            if config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters() if feature_extractor is None else model.parameters(),
                    config.training.gradient_clip
                )

            optimizer.step()

        # Update metrics
        total_loss += loss.item()
        total_ce_loss += loss_dict.get('ce_loss', 0)
        total_dice_loss += loss_dict.get('dice_loss', 0)
        num_batches += 1
        
        # Track auxiliary metrics if available
        aux_fg_bg_loss += loss_dict.get('aux_fg_bg_loss', 0)
        aux_fg_accuracy += loss_dict.get('aux_fg_accuracy', 0)
        aux_fg_iou += loss_dict.get('aux_fg_iou', 0)
        
        # Track refinement losses if available
        active_contour_loss += loss_dict.get('active_contour', 0)
        boundary_aware_loss += loss_dict.get('boundary_aware', 0)
        contour_loss += loss_dict.get('contour', 0)
        distance_transform_loss += loss_dict.get('distance_transform', 0)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })

        # Log to tensorboard
        if writer is not None and batch_idx % 10 == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)

            # Log additional metrics
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    writer.add_scalar(f'train/{key}', value.item(), global_step)

    # Return epoch metrics
    metrics = {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'dice_loss': total_dice_loss / num_batches,
        'grad_norm': grad_norm_before_clip if num_batches > 0 else 0.0
    }
    
    # Add auxiliary metrics if available
    if aux_fg_bg_loss > 0:
        metrics['aux_fg_bg_loss'] = aux_fg_bg_loss / num_batches
        metrics['aux_fg_accuracy'] = aux_fg_accuracy / num_batches
        metrics['aux_fg_iou'] = aux_fg_iou / num_batches
    
    # Add refinement losses if available
    if active_contour_loss > 0:
        metrics['active_contour'] = active_contour_loss / num_batches
    if boundary_aware_loss > 0:
        metrics['boundary_aware'] = boundary_aware_loss / num_batches
    if contour_loss > 0:
        metrics['contour'] = contour_loss / num_batches
    if distance_transform_loss > 0:
        metrics['distance_transform'] = distance_transform_loss / num_batches
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Advanced training with configurable features')

    # Configuration
    parser.add_argument('--config', type=str, default='baseline',
                        help='Configuration name or path to config file')
    parser.add_argument('--config_modifications', type=str, default='{}',
                        help='JSON string with config modifications')

    # Override options
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--test_only', action='store_true', help='Only run validation')

    args = parser.parse_args()

    # Load configuration
    if args.config in ConfigManager.list_configs():
        config = ConfigManager.get_config(args.config)
    elif Path(args.config).exists():
        config = ExperimentConfig.load(args.config)
    else:
        raise ValueError(f"Unknown config: {args.config}")

    # Apply modifications
    if args.config_modifications != '{}':
        mods = json.loads(args.config_modifications)
        config = ConfigManager.create_custom_config(config.name, mods)

    # Create experiment directories
    exp_dirs = create_experiment_dirs(config)

    # Save configuration
    config.save(exp_dirs['configs'] / 'config.json')

    print(f"\nStarting experiment: {config.name}")
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
    if hasattr(config.model, 'use_hierarchical') and config.model.use_hierarchical:
        print(f"  - Hierarchical architecture: {config.model.use_hierarchical}")
    if hasattr(config.model, 'use_class_specific_decoder') and config.model.use_class_specific_decoder:
        print(f"  - Class-specific decoder: {config.model.use_class_specific_decoder}")

    # Setup device - always use CUDA for PyTorch if available
    # TensorRT is only for ONNX Runtime inference
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device} (PyTorch)")
    if config.model.execution_provider == 'tensorrt':
        print(f"Using TensorRT for ONNX inference")

    # Load data statistics
    with open(config.data.data_stats, 'r') as f:
        data_stats = json.load(f)

    pixel_ratios = data_stats['pixel_ratios']
    separation_aware_weights = data_stats.get('separation_aware_weights')

    # Build model
    print("\nBuilding model...")
    model, feature_extractor = build_model(config, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Build loss function
    print("\nBuilding loss function...")
    loss_fn = build_loss_function(
        config, pixel_ratios, device, separation_aware_weights
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = COCOInstanceSegmentationDataset(
        annotation_file=config.data.train_annotation,
        image_dir=config.data.train_img_dir,
        mask_size=(config.model.mask_size, config.model.mask_size),
        roi_padding=config.data.roi_padding
    )

    val_dataset = COCOInstanceSegmentationDataset(
        annotation_file=config.data.val_annotation,
        image_dir=config.data.val_img_dir,
        mask_size=(config.model.mask_size, config.model.mask_size),
        roi_padding=config.data.roi_padding
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Skip the first ONNX export - we'll do it later with the checkpoint

    # Create dataloaders with custom collate function
    collate_fn = create_collate_fn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        collate_fn=collate_fn
    )

    # Setup optimizer
    params_to_optimize = model.parameters()
    if config.training.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=config.training.learning_rate
        )

    # Setup scheduler
    if config.training.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.min_lr
        )
    elif config.training.scheduler == 'cosine_warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.T_0,
            T_mult=config.training.T_mult,
            eta_min=config.training.eta_min_restart
        )
    else:
        scheduler = None

    # Setup mixed precision
    scaler = GradScaler('cuda') if config.training.mixed_precision else None

    # Setup tensorboard
    writer = SummaryWriter(exp_dirs['logs'])
    
    # Setup text logger
    text_logger = TextLogger(exp_dirs['logs'])
    text_logger.log_config(config.to_dict())

    # Create validation visualizer
    from pycocotools.coco import COCO
    from src.human_edge_detection.advanced.visualization_adapter import AdvancedValidationVisualizer

    val_coco = COCO(config.data.val_annotation)
    
    # Use auxiliary visualizer if auxiliary task is enabled
    if config.auxiliary_task.enabled:
        from src.human_edge_detection.visualize_auxiliary import ValidationVisualizerWithAuxiliary
        visualizer = ValidationVisualizerWithAuxiliary(
            model=model,
            feature_extractor=feature_extractor,
            coco=val_coco,
            image_dir=config.data.val_img_dir,
            output_dir=exp_dirs['visualizations'],
            device=device,
            roi_padding=config.data.roi_padding,
            visualize_auxiliary=config.auxiliary_task.visualize
        )
    # Use HierarchicalUNetVisualizer for hierarchical UNet models
    elif any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
        from src.human_edge_detection.advanced.hierarchical_unet_visualizer import HierarchicalUNetVisualizer
        visualizer = HierarchicalUNetVisualizer(
            model=model,
            feature_extractor=feature_extractor,
            coco=val_coco,
            image_dir=config.data.val_img_dir,
            output_dir=exp_dirs['visualizations'],
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
            output_dir=exp_dirs['visualizations'],
            device=device,
            is_multiscale=config.multiscale.enabled
        )

    # Resume from checkpoint
    start_epoch = 0
    best_miou = 0

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Try to load model state dict with better error handling
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Failed to load model weights with strict=True, trying strict=False...")
            print(f"Error: {e}")
            # Try non-strict loading
            incompatible_keys = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if incompatible_keys.missing_keys:
                print(f"Missing keys: {len(incompatible_keys.missing_keys)}")
                if len(incompatible_keys.missing_keys) < 10:
                    for key in incompatible_keys.missing_keys:
                        print(f"  - {key}")
            if incompatible_keys.unexpected_keys:
                print(f"Unexpected keys: {len(incompatible_keys.unexpected_keys)}")
                if len(incompatible_keys.unexpected_keys) < 10:
                    for key in incompatible_keys.unexpected_keys:
                        print(f"  - {key}")
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_miou = checkpoint.get('best_miou', 0)

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Test only mode
    if args.test_only:
        print("\nRunning validation only...")
        val_metrics = evaluate_model(
            model, val_loader, loss_fn, device,
            feature_extractor=feature_extractor,
            config=config
        )

        print(f"\nValidation Results:")
        print(f"  Total Loss: {val_metrics['total_loss']:.4f}")
        print(f"  mIoU: {val_metrics['miou']:.4f}")
        for i in range(config.model.num_classes):
            print(f"  IoU class {i}: {val_metrics[f'iou_class_{i}']:.4f}")
        return

    # Save untrained model at the beginning (only if not resuming)
    if not args.resume:
        print("\nSaving untrained model...")
        untrained_checkpoint_path = exp_dirs['checkpoints'] / 'untrained_model.pth'
        torch.save({
            'epoch': -1,  # -1 indicates untrained
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_miou': 0.0,
            'config': config.to_dict()
        }, untrained_checkpoint_path)
        print(f"Saved untrained model to {untrained_checkpoint_path}")

        # Export untrained model to ONNX
        try:
            if config.auxiliary_task.enabled:
                # Export two versions for auxiliary-enabled models
                from src.human_edge_detection.export_onnx_advanced_auxiliary import export_model_inference_only
                from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced
                
                # 1. Export with auxiliary branch
                untrained_onnx_path = exp_dirs['checkpoints'] / 'untrained_model.onnx'
                print("\nExporting untrained model to ONNX (with auxiliary branch)...")
                
                # Determine model type
                if config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
                    model_type = 'hierarchical'
                elif config.model.use_class_specific_decoder:
                    model_type = 'class_specific'
                elif config.multiscale.enabled:
                    model_type = 'multiscale'
                else:
                    model_type = 'baseline'
                
                success_with_aux = export_checkpoint_to_onnx_advanced(
                    checkpoint_path=str(untrained_checkpoint_path),
                    output_path=str(untrained_onnx_path),
                    model_type=model_type,
                    config=config.to_dict(),
                    device=device,
                    verify=False,  # Skip verification for untrained models
                    include_auxiliary=True  # Include auxiliary branches in export
                )
                
                # 2. Export without auxiliary branch (optimized for inference)
                untrained_opt_onnx_path = exp_dirs['checkpoints'] / 'untrained_opt_model.onnx'
                print("\nExporting untrained model to ONNX (inference only, no auxiliary)...")
                export_model_inference_only(
                    checkpoint_path=str(untrained_checkpoint_path),
                    output_path=str(untrained_opt_onnx_path),
                    device=device
                )
                success = success_with_aux
            else:
                from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced
                untrained_onnx_path = exp_dirs['checkpoints'] / 'untrained_model.onnx'
                # Determine model type
                if config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
                    model_type = 'hierarchical'
                elif config.model.use_class_specific_decoder:
                    model_type = 'class_specific'
                elif config.multiscale.enabled:
                    model_type = 'multiscale'
                else:
                    model_type = 'baseline'
                
                if hasattr(config.model, 'use_rgb_enhancement') and config.model.use_rgb_enhancement:
                    print("\nExporting RGB enhanced decoder to ONNX...")
                else:
                    print("\nExporting untrained model to ONNX...")
                    
                success = export_checkpoint_to_onnx_advanced(
                    checkpoint_path=str(untrained_checkpoint_path),
                    output_path=str(untrained_onnx_path),
                    model_type=model_type,
                    config=config.to_dict(),
                    device=device,
                    verify=False  # Skip verification for untrained models
                )

            if success:
                if hasattr(config.model, 'use_rgb_enhancement') and config.model.use_rgb_enhancement:
                    print(f"Exported RGB enhanced decoder to {untrained_onnx_path}")
                    print("Note: This ONNX model includes RGB encoder and requires:")
                    print("  - Pre-extracted YOLO features (layer_3, layer_22, layer_34)")
                    print("  - RGB images (3x640x640)")
                    print("  - ROI boxes")
                else:
                    print(f"Exported untrained model to {untrained_onnx_path}")
            else:
                print("Warning: Failed to export untrained model to ONNX")
        except Exception as e:
            print(f"Warning: Failed to export untrained model to ONNX: {e}")
        print("Continuing with training...")

    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")
    
    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            # Train
            train_metrics = train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                epoch, config, feature_extractor, scaler, writer, text_logger
            )

            # Log training metrics
            writer.add_scalar('train/total_loss', train_metrics['total_loss'], epoch)
            writer.add_scalar('train/ce_loss', train_metrics['ce_loss'], epoch)
            writer.add_scalar('train/dice_loss', train_metrics['dice_loss'], epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            
            # Log auxiliary metrics if available
            if 'aux_fg_bg_loss' in train_metrics:
                writer.add_scalar('train/aux_fg_bg_loss', train_metrics['aux_fg_bg_loss'], epoch)
                writer.add_scalar('train/aux_fg_accuracy', train_metrics['aux_fg_accuracy'], epoch)
                writer.add_scalar('train/aux_fg_iou', train_metrics['aux_fg_iou'], epoch)
            
            # Log refinement losses if available
            if 'active_contour' in train_metrics:
                writer.add_scalar('train/active_contour', train_metrics['active_contour'], epoch)
            if 'boundary_aware' in train_metrics:
                writer.add_scalar('train/boundary_aware', train_metrics['boundary_aware'], epoch)
            if 'contour' in train_metrics:
                writer.add_scalar('train/contour', train_metrics['contour'], epoch)
            if 'distance_transform' in train_metrics:
                writer.add_scalar('train/distance_transform', train_metrics['distance_transform'], epoch)

            # Validation
            if epoch % config.training.validate_every == 0:
                val_metrics = evaluate_model(
                    model, val_loader, loss_fn, device,
                    feature_extractor=feature_extractor,
                    config=config
                )

                # Log validation metrics
                writer.add_scalar('val/total_loss', val_metrics['total_loss'], epoch)
                writer.add_scalar('val/miou', val_metrics['miou'], epoch)
                for i in range(config.model.num_classes):
                    writer.add_scalar(f'val/iou_class_{i}', val_metrics[f'iou_class_{i}'], epoch)
                
                # Log validation auxiliary metrics if available
                if 'aux_fg_bg_loss' in val_metrics:
                    writer.add_scalar('val/aux_fg_bg_loss', val_metrics['aux_fg_bg_loss'], epoch)
                    writer.add_scalar('val/aux_fg_accuracy', val_metrics['aux_fg_accuracy'], epoch)
                    writer.add_scalar('val/aux_fg_iou', val_metrics['aux_fg_iou'], epoch)
                
                # Log validation refinement losses if available
                if 'active_contour' in val_metrics:
                    writer.add_scalar('val/active_contour', val_metrics['active_contour'], epoch)
                if 'boundary_aware' in val_metrics:
                    writer.add_scalar('val/boundary_aware', val_metrics['boundary_aware'], epoch)
                if 'contour' in val_metrics:
                    writer.add_scalar('val/contour', val_metrics['contour'], epoch)
                if 'distance_transform' in val_metrics:
                    writer.add_scalar('val/distance_transform', val_metrics['distance_transform'], epoch)

                print(f"\nEpoch {epoch+1} - Validation:")
                print(f"  Loss: {val_metrics['total_loss']:.4f}")
                print(f"  mIoU: {val_metrics['miou']:.4f}")
                
                # Log epoch summary to text file
                text_logger.log_epoch_summary(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    learning_rate=optimizer.param_groups[0]['lr']
                )

                # Generate validation visualization
                print("Generating validation visualization...")
                visualizer.visualize_validation_images(epoch)

                # Save best model
                if val_metrics['miou'] > best_miou:
                    best_miou = val_metrics['miou']
                    checkpoint_path = exp_dirs['checkpoints'] / 'best_model.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_miou': best_miou,
                        'config': config.to_dict()
                    }, checkpoint_path)
                    print(f"  Saved best model (mIoU: {best_miou:.4f})")
                    
                    # Log best model save
                    text_logger.log_best_model(
                        epoch=epoch,
                        miou=best_miou,
                        checkpoint_path=str(checkpoint_path)
                    )
            else:
                # No validation this epoch, but still log training summary
                text_logger.log_epoch_summary(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=None,
                    learning_rate=optimizer.param_groups[0]['lr']
                )

            # Save checkpoint
            if epoch % config.training.save_every == 0:
                checkpoint_path = exp_dirs['checkpoints'] / f'checkpoint_epoch_{epoch+1:04d}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_miou': best_miou,
                    'config': config.to_dict()
                }, checkpoint_path)

            # Update scheduler
            if scheduler:
                scheduler.step()

    except Exception as e:
        print(f"\nTraining error occurred: {e}")
        text_logger.log_error(str(e))
        raise
    finally:
        writer.close()
        text_logger.close()
    
    # Export best model to ONNX
    print("\nExporting best model to ONNX...")
    best_model_path = exp_dirs['checkpoints'] / 'best_model.pth'
    best_onnx_path = exp_dirs['checkpoints'] / 'best_model.onnx'
    
    if best_model_path.exists():
        try:
            if config.auxiliary_task.enabled:
                # Use auxiliary-aware export
                from src.human_edge_detection.export_onnx_advanced_auxiliary import export_model_inference_only
                print("Exporting best model to ONNX (inference only, no auxiliary)...")
                export_model_inference_only(
                    checkpoint_path=str(best_model_path),
                    output_path=str(best_onnx_path),
                    device=device
                )
            else:
                # Use standard export
                from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced
                # Determine model type
                if config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
                    model_type = 'hierarchical'
                elif config.model.use_class_specific_decoder:
                    model_type = 'class_specific'
                elif config.multiscale.enabled:
                    model_type = 'multiscale'
                else:
                    model_type = 'baseline'
                
                export_checkpoint_to_onnx_advanced(
                    checkpoint_path=str(best_model_path),
                    output_path=str(best_onnx_path),
                    model_type=model_type,
                    config=config.to_dict(),
                    device=device,
                    verify=True
                )
            print(f"Best model exported to: {best_onnx_path}")
            
        except Exception as e:
            import traceback
            print(f"Warning: Could not export best model to ONNX: {e}")
            print("Full traceback:")
            traceback.print_exc()
    
    print(f"\nTraining completed! Best mIoU: {best_miou:.4f}")


if __name__ == "__main__":
    main()