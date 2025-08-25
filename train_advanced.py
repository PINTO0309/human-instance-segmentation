"""Advanced training script with configurable features."""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import warnings
import logging
import copy

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

# Suppress TracerWarning messages from PyTorch
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._trace")

# Also suppress warnings about tensor conversion in tracing
warnings.filterwarnings("ignore", message="Converting a tensor to a Python boolean might cause the trace to be incorrect")

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
from src.human_edge_detection.advanced.knowledge_distillation import (
    DistillationLoss, DistillationModelWrapper
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

        # Determine encoder name - use student encoder if distillation is enabled, otherwise use config
        encoder_name = config.distillation.student_encoder if config.distillation.enabled else getattr(config.model, 'encoder_name', 'timm-efficientnet-b3')

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
            # Activation function configuration
            activation_function=getattr(config.model, 'activation_function', 'relu'),
            activation_beta=getattr(config.model, 'activation_beta', 1.0),
            # Pre-trained model configuration
            use_pretrained_unet=getattr(config.model, 'use_pretrained_unet', False),
            pretrained_weights_path=getattr(config.model, 'pretrained_weights_path', ''),
            freeze_pretrained_weights=getattr(config.model, 'freeze_pretrained_weights', False),
            use_full_image_unet=getattr(config.model, 'use_full_image_unet', False),
            encoder_name=encoder_name,  # Pass encoder name
            # Hierarchical segmentation head configuration
            hierarchical_base_channels=getattr(config.model, 'hierarchical_base_channels', 96),
            hierarchical_depth=getattr(config.model, 'hierarchical_depth', 3)
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

    # Handle knowledge distillation if enabled
    if config.distillation.enabled:
        print(f"\nSetting up knowledge distillation from teacher checkpoint: {config.distillation.teacher_checkpoint}")

        # Build teacher model with same configuration but B3 encoder
        teacher_config = copy.deepcopy(config)
        teacher_config.distillation.enabled = False  # Disable distillation for teacher

        # For RGB hierarchical models with distillation
        if config.model.use_rgb_hierarchical:
            from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model

            # Build teacher model (B3)
            teacher_model = create_rgb_hierarchical_model(
                roi_size=config.model.roi_size,
                mask_size=config.model.mask_size,
                multi_scale=config.multiscale.enabled and config.model.variable_roi_sizes,
                roi_sizes=config.model.variable_roi_sizes if config.multiscale.enabled else None,
                fusion_method=config.multiscale.fusion_method if config.multiscale.enabled else 'concat',
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
                # Activation function configuration
                activation_function=getattr(config.model, 'activation_function', 'relu'),
                activation_beta=getattr(config.model, 'activation_beta', 1.0),
                # Pre-trained model configuration - TEACHER USES B3
                use_pretrained_unet=getattr(config.model, 'use_pretrained_unet', False),
                pretrained_weights_path=getattr(config.model, 'pretrained_weights_path', ''),
                freeze_pretrained_weights=False,  # We'll freeze the entire teacher model
                use_full_image_unet=getattr(config.model, 'use_full_image_unet', False),
                encoder_name="timm-efficientnet-b3"  # Teacher uses B3
            )

            # Build student model (B0) - reuse the already created model but ensure it uses B0
            # The model variable already contains the student model built earlier
            # We need to ensure it uses the correct encoder
            if hasattr(model, 'fg_bg_branch') and hasattr(model.fg_bg_branch, 'model'):
                # Update the encoder name for the student model if needed
                pass  # Model already built with correct settings

        # Load teacher checkpoint
        if not config.distillation.teacher_checkpoint:
            raise ValueError("Teacher checkpoint path is empty. Please specify --teacher_checkpoint or set it in the config.")

        if not os.path.exists(config.distillation.teacher_checkpoint):
            raise FileNotFoundError(
                f"Teacher checkpoint not found: {config.distillation.teacher_checkpoint}\n"
                f"Please check the path and ensure the file exists."
            )

        print(f"Loading teacher model from {config.distillation.teacher_checkpoint}")
        try:
            checkpoint = torch.load(config.distillation.teacher_checkpoint, map_location=device)

            # Check if checkpoint needs key remapping (old structure without wrapper)
            state_dict = checkpoint['model_state_dict']
            if 'pretrained_unet.norm_mean' in state_dict:
                # Old checkpoint structure - needs remapping to new wrapper structure
                print("Detected old checkpoint format, remapping keys for PreTrainedPeopleSegmentationUNetWrapper...")
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('pretrained_unet.'):
                        # Split the key to insert 'model' after 'pretrained_unet'
                        parts = key.split('.', 1)  # Split at first dot
                        if len(parts) == 2:
                            # Insert 'model' between 'pretrained_unet' and the rest
                            new_key = f"pretrained_unet.model.{parts[1]}"
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    else:
                        new_state_dict[key] = value
                # Filter out incompatible keys (feature_combiner might have different dimensions)
                model_state = teacher_model.state_dict()
                filtered_state_dict = {}
                for key, value in new_state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Skipping {key} due to shape mismatch: checkpoint {value.shape} vs model {model_state[key].shape}")
                    else:
                        # Key doesn't exist in model (might be from older version)
                        pass

                # Try loading with strict=False to handle missing keys
                incompatible = teacher_model.load_state_dict(filtered_state_dict, strict=False)
                if incompatible.missing_keys:
                    print(f"Warning: Missing keys in teacher checkpoint: {len(incompatible.missing_keys)} keys")
                    if len(incompatible.missing_keys) <= 10:
                        for key in incompatible.missing_keys:
                            print(f"  - {key}")
                if incompatible.unexpected_keys:
                    print(f"Warning: Unexpected keys in teacher checkpoint: {incompatible.unexpected_keys[:5]}...")
            else:
                # New checkpoint structure or doesn't need remapping
                # Filter out incompatible keys
                model_state = teacher_model.state_dict()
                filtered_state_dict = {}
                for key, value in state_dict.items():
                    if key in model_state:
                        if model_state[key].shape == value.shape:
                            filtered_state_dict[key] = value
                        else:
                            print(f"Skipping {key} due to shape mismatch: checkpoint {value.shape} vs model {model_state[key].shape}")
                    else:
                        # Key doesn't exist in model
                        pass

                # Try loading with strict=False to handle missing keys
                incompatible = teacher_model.load_state_dict(filtered_state_dict, strict=False)
                if incompatible.missing_keys:
                    print(f"Warning: Missing keys in teacher checkpoint: {len(incompatible.missing_keys)} keys")
                    if len(incompatible.missing_keys) <= 10:
                        for key in incompatible.missing_keys:
                            print(f"  - {key}")
                if incompatible.unexpected_keys:
                    print(f"Warning: Unexpected keys in teacher checkpoint: {incompatible.unexpected_keys[:5]}...")

            # Fix randomly initialized output_conv in teacher if it wasn't loaded
            if hasattr(teacher_model, 'pretrained_unet') and hasattr(teacher_model.pretrained_unet, 'output_conv'):
                output_conv_key = 'pretrained_unet.output_conv.weight'
                if output_conv_key not in filtered_state_dict and output_conv_key not in state_dict:
                    print("Initializing teacher's output_conv to stable values for binary mask prediction...")
                    with torch.no_grad():
                        # Initialize to convert 1-channel mask to 2-channel [background, foreground]
                        teacher_model.pretrained_unet.output_conv.weight.zero_()
                        # Channel 0: background (inverse of mask)
                        teacher_model.pretrained_unet.output_conv.weight[0, 0] = -1.0
                        # Channel 1: foreground (mask)
                        teacher_model.pretrained_unet.output_conv.weight[1, 0] = 1.0
                        # Zero bias
                        teacher_model.pretrained_unet.output_conv.bias.zero_()

            teacher_model = teacher_model.to(device)
            print(f"Successfully loaded teacher model (epoch {checkpoint.get('epoch', 'unknown')})")
        except Exception as e:
            raise RuntimeError(f"Failed to load teacher checkpoint: {e}")

        # Wrap models with distillation wrapper
        model = DistillationModelWrapper(
            teacher_model=teacher_model,
            student_model=model,
            freeze_teacher=config.distillation.freeze_teacher
        )

        print(f"Distillation setup complete. Student encoder: {config.distillation.student_encoder}")

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

            loss_fn = RefinedHierarchicalLoss(
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

            loss_fn = HierarchicalLoss(
                bg_weight=1.5,
                fg_weight=1.5,
                target_weight=1.2,
                consistency_weight=0.3,
                use_dynamic_weights=use_dynamic_weights,
                dice_weight=config.training.dice_weight,
                ce_weight=config.training.ce_weight
            )

    elif config.distance_loss.enabled:
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
    # BUT skip for hierarchical models as they handle auxiliary outputs internally
    is_hierarchical = config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(
        getattr(config.model, attr, False)
        for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2',
                     'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']
    )

    if config.auxiliary_task.enabled and not is_hierarchical:
        loss_fn = MultiTaskLoss(
            main_loss_fn=loss_fn,
            aux_weight=config.auxiliary_task.weight,
            aux_pos_weight=config.auxiliary_task.pos_weight
        )

    # Wrap with distillation loss if enabled
    if config.distillation.enabled:
        loss_fn = DistillationLoss(
            base_loss_fn=loss_fn,
            temperature=config.distillation.temperature,
            alpha=config.distillation.alpha,
            distill_logits=config.distillation.distill_logits,
            distill_features=config.distillation.distill_features,
            feature_match_layers=config.distillation.feature_match_layers
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

                # Handle distillation model output (returns student and teacher predictions)
                if config.distillation.enabled:
                    # Model is wrapped in DistillationModelWrapper
                    student_predictions, teacher_predictions = predictions
                    # Pass both to distillation loss with masks as targets
                    # DistillationLoss.forward expects (student_outputs, teacher_outputs, targets)
                    # The aux_outputs will be extracted from student/teacher predictions if they are tuples
                    loss, loss_dict = loss_fn(student_predictions, teacher_predictions, masks)
                # Handle hierarchical model output
                elif config.model.use_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
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

            # Handle distillation model output (returns student and teacher predictions)
            if config.distillation.enabled:
                # Model is wrapped in DistillationModelWrapper
                student_predictions, teacher_predictions = predictions
                # Pass both to distillation loss with masks as targets
                # DistillationLoss.forward expects (student_outputs, teacher_outputs, targets)
                # The aux_outputs will be extracted from student/teacher predictions if they are tuples
                loss, loss_dict = loss_fn(student_predictions, teacher_predictions, masks)
            # Handle hierarchical model output
            elif config.model.use_hierarchical or config.model.use_rgb_hierarchical or any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
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
        # Handle both direct and distillation (base_) prefixed loss keys
        total_ce_loss += loss_dict.get('ce_loss', loss_dict.get('base_ce_loss', 0))
        total_dice_loss += loss_dict.get('dice_loss', loss_dict.get('base_dice_loss', 0))
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
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')

    # Distillation options
    parser.add_argument('--teacher_checkpoint', type=str, help='Path to teacher model checkpoint for distillation')
    parser.add_argument('--distillation_temperature', type=float, help='Temperature for distillation')
    parser.add_argument('--distillation_alpha', type=float, help='Alpha weight for distillation (0-1)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Enable mixed precision training for faster training and lower memory usage')

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

    # Override distillation settings from command line if provided
    if args.teacher_checkpoint is not None:
        config.distillation.teacher_checkpoint = args.teacher_checkpoint
        config.distillation.enabled = True  # Auto-enable distillation if teacher checkpoint is provided
        print(f"Overriding teacher checkpoint: {args.teacher_checkpoint}")

    if args.distillation_temperature is not None:
        config.distillation.temperature = args.distillation_temperature
        print(f"Overriding distillation temperature: {args.distillation_temperature}")

    if args.distillation_alpha is not None:
        config.distillation.alpha = args.distillation_alpha
        print(f"Overriding distillation alpha: {args.distillation_alpha}")

    # Override epochs if provided
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
        print(f"Overriding number of epochs: {args.epochs}")

    # Override batch size if provided
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")

    # Override mixed precision if provided
    if args.mixed_precision:
        config.training.mixed_precision = True
        print(f"Enabling mixed precision training")

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
    # Handle mask_size as either int, tuple, or list
    if isinstance(config.model.mask_size, (tuple, list)):
        mask_size = tuple(config.model.mask_size)  # Convert to tuple if list
    else:
        mask_size = (config.model.mask_size, config.model.mask_size)

    # Create transforms for training
    train_transform = None
    if config.data.use_augmentation:
        from src.human_edge_detection.augmentations import (
            get_train_transforms, get_val_transforms, get_roi_safe_transforms
        )

        # Check if this model uses ROI coordinates
        # Even "full-image" models in this project use ROI coordinates for segmentation
        # So we should always use ROI-safe transforms to avoid coordinate misalignment
        uses_roi_coordinates = True  # All models in this project use ROI coordinates

        # The original logic was incorrect - even models with use_full_image_unet=True
        # still extract ROI regions for segmentation, so they need ROI-safe transforms

        if uses_roi_coordinates:
            # Use ROI-safe transforms for all models (avoids geometric transform misalignment)
            train_transform = get_roi_safe_transforms(
                input_size=(640, 640),
                use_heavy_augmentation=config.data.use_heavy_augmentation
            )
            print(f"Data augmentation enabled (ROI-safe mode):")
            print(f"  - Heavy augmentation: {config.data.use_heavy_augmentation}")
            print(f"  - Using ROI-safe transforms (no geometric transforms to avoid ROI misalignment)")
        else:
            # This branch is kept for potential future models that don't use ROI coordinates
            train_transform = get_train_transforms(
                input_size=(640, 640),
                use_heavy_augmentation=config.data.use_heavy_augmentation
            )
            print(f"Data augmentation enabled (standard mode):")
            print(f"  - Heavy augmentation: {config.data.use_heavy_augmentation}")

        val_transform = get_val_transforms(
            input_size=(640, 640)  # Using fixed size as dataset handles resizing
        )
    else:
        val_transform = None
        print("Data augmentation disabled")

    train_dataset = COCOInstanceSegmentationDataset(
        annotation_file=config.data.train_annotation,
        image_dir=config.data.train_img_dir,
        mask_size=mask_size,
        roi_padding=config.data.roi_padding,
        transform=train_transform
    )

    val_dataset = COCOInstanceSegmentationDataset(
        annotation_file=config.data.val_annotation,
        image_dir=config.data.val_img_dir,
        mask_size=mask_size,
        roi_padding=config.data.roi_padding,
        transform=val_transform
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

    # If using distillation, use student model for visualization
    model_for_vis = model.get_student() if isinstance(model, DistillationModelWrapper) else model

    # Use auxiliary visualizer if auxiliary task is enabled
    if config.auxiliary_task.enabled:
        from src.human_edge_detection.visualize_auxiliary import ValidationVisualizerWithAuxiliary
        visualizer = ValidationVisualizerWithAuxiliary(
            model=model_for_vis,
            feature_extractor=feature_extractor,
            coco=val_coco,
            image_dir=config.data.val_img_dir,
            output_dir=exp_dirs['visualizations'],
            device=device,
            roi_padding=config.data.roi_padding,
            visualize_auxiliary=config.auxiliary_task.visualize,
            use_roi_comparison=config.data.use_roi_comparison,
            use_edge_visualize=config.data.use_edge_visualize
        )
    # Use HierarchicalUNetVisualizer for hierarchical UNet models
    elif any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical', 'use_pretrained_unet', 'use_full_image_unet']):
        from src.human_edge_detection.advanced.hierarchical_unet_visualizer import HierarchicalUNetVisualizer
        visualizer = HierarchicalUNetVisualizer(
            model=model_for_vis,
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
            model=model_for_vis,
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

        # Get the model to load weights into (student if distillation, otherwise the model itself)
        model_to_load = model.get_student() if isinstance(model, DistillationModelWrapper) else model

        # Try to load model state dict with better error handling
        try:
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            print(f"Warning: Failed to load model weights with strict=True, trying strict=False...")
            print(f"Error: {e}")
            # Try non-strict loading
            incompatible_keys = model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)
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

        # Fix output_conv weights for PreTrainedPeopleSegmentationUNetWrapper
        # This ensures consistent interpretation of channels after resume
        if hasattr(model_to_load, 'pretrained_unet') and hasattr(model_to_load.pretrained_unet, 'output_conv'):
            print("Fixing pretrained_unet.output_conv weights for consistent channel interpretation...")
            with torch.no_grad():
                model_to_load.pretrained_unet.output_conv.weight.data[0, 0, 0, 0] = 1.0  # Channel 0: background
                model_to_load.pretrained_unet.output_conv.weight.data[1, 0, 0, 0] = -1.0   # Channel 1: foreground
                model_to_load.pretrained_unet.output_conv.bias.data.zero_()
            print("  Channel 0 (background): +1.0")
            print("  Channel 1 (foreground): -1.0")

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Test only mode
    if args.test_only:
        print("\nRunning validation only...")
        val_metrics = evaluate_model(
            model, val_loader, loss_fn, device,
            feature_extractor=feature_extractor,
            config=config,
            epoch=0,
            output_dir=exp_dirs['checkpoints']
        )

        print(f"\nValidation Results:")
        print(f"  Total Loss: {val_metrics['total_loss']:.4f}")
        print(f"\n  Primary Metrics:")
        print(f"    Target IoU (mIoU): {val_metrics['target_iou']:.4f}")
        print(f"    Detection Rate (IoU > 0.5): {val_metrics['detection_rate_0.5']:.2%}")
        print(f"    Detection Rate (IoU > 0.7): {val_metrics['detection_rate_0.7']:.2%}")
        print(f"\n  Class-wise IoU:")
        for i in range(config.model.num_classes):
            class_name = ['Background', 'Target', 'Non-target'][i]
            print(f"    {class_name}: {val_metrics[f'iou_class_{i}']:.4f}")
        print(f"\n  Target Performance:")
        print(f"    Precision: {val_metrics.get('target_precision', 0):.4f}")
        print(f"    Recall: {val_metrics.get('target_recall', 0):.4f}")
        print(f"    F1 Score: {val_metrics.get('target_f1', 0):.4f}")
        print(f"\n  Instance Separation:")
        print(f"    Instance Separation Accuracy: {val_metrics.get('instance_separation_accuracy', 0):.4f}")
        return

    # Save untrained model at the beginning (only if not resuming)
    if not args.resume:
        print("\nSaving untrained model...")
        untrained_checkpoint_path = exp_dirs['checkpoints'] / 'untrained_model.pth'
        # If using distillation, save student model only
        model_to_save = model.get_student() if isinstance(model, DistillationModelWrapper) else model
        torch.save({
            'epoch': -1,  # -1 indicates untrained
            'model_state_dict': model_to_save.state_dict(),
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

    # Setup staged training if enabled (check both staged_training and distillation config)
    staged_training_enabled = False
    stages = []

    # Check for encoder_only_epochs in distillation config (priority)
    if config.distillation.enabled and hasattr(config.distillation, 'encoder_only_epochs') and config.distillation.encoder_only_epochs > 0:
        from src.human_edge_detection.staged_training import (
            StageConfig, get_current_stage, apply_stage_freezing, update_optimizer_for_stage
        )
        staged_training_enabled = True

        # Create stages based on distillation config
        stages = [
            StageConfig(
                name="encoder_only",
                start_epoch=0,
                end_epoch=config.distillation.encoder_only_epochs,
                freeze_encoder=False,
                freeze_decoder=True,
                freeze_rgb_extractor=True,
                learning_rate_scale=getattr(config.distillation, 'encoder_lr_scale', 1.0)
            ),
            StageConfig(
                name="full_model",
                start_epoch=config.distillation.encoder_only_epochs,
                end_epoch=None,
                freeze_encoder=False,
                freeze_decoder=False,
                freeze_rgb_extractor=False,
                learning_rate_scale=getattr(config.distillation, 'full_model_lr_scale', 0.5)
            )
        ]
        print(f"\nStaged training enabled via distillation config: {config.distillation.encoder_only_epochs} encoder-only epochs")

    # Fallback to explicit staged_training config if exists
    elif hasattr(config, 'staged_training') and config.staged_training.get('enabled', False):
        from src.human_edge_detection.staged_training import (
            StageConfig, get_current_stage, apply_stage_freezing, update_optimizer_for_stage
        )
        staged_training_enabled = True

        # Parse stage configurations
        for stage_dict in config.staged_training.get('stages', []):
            stages.append(StageConfig(
                name=stage_dict['name'],
                start_epoch=stage_dict['start_epoch'],
                end_epoch=stage_dict.get('end_epoch'),
                freeze_encoder=stage_dict.get('freeze_encoder', False),
                freeze_decoder=stage_dict.get('freeze_decoder', False),
                freeze_rgb_extractor=stage_dict.get('freeze_rgb_extractor', False),
                learning_rate_scale=stage_dict.get('learning_rate_scale', 1.0)
            ))
        print(f"\nStaged training enabled with {len(stages)} stages")

    if staged_training_enabled:
        current_stage_name = None

    # Training loop
    print(f"\nStarting training for {config.training.num_epochs} epochs...")

    try:
        for epoch in range(start_epoch, config.training.num_epochs):
            # Apply staged training configuration if enabled
            if staged_training_enabled:
                stage = get_current_stage(epoch, stages)
                if stage and stage.name != current_stage_name:
                    print(f"\n{'='*60}")
                    print(f"Entering training stage: {stage.name} (epoch {epoch})")
                    print(f"{'='*60}")

                    # Apply freezing configuration
                    apply_stage_freezing(model, stage, verbose=True)

                    # Update optimizer with new parameters
                    optimizer = update_optimizer_for_stage(
                        optimizer, model, stage, config.training.learning_rate
                    )

                    # Update scheduler if needed
                    if scheduler:
                        # Recreate scheduler with new optimizer
                        if config.training.scheduler == 'cosine':
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                optimizer,
                                T_max=config.training.num_epochs - epoch,
                                eta_min=config.training.min_lr
                            )
                        elif config.training.scheduler == 'cosine_warm_restarts':
                            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                optimizer,
                                T_0=config.training.T_0,
                                T_mult=config.training.T_mult,
                                eta_min=config.training.eta_min_restart
                            )

                    current_stage_name = stage.name

            # Train
            train_metrics = train_epoch(
                model, train_loader, loss_fn, optimizer, device,
                epoch, config, feature_extractor, scaler, writer, text_logger
            )

            # Log training metrics with hierarchical naming

            # 01. Primary Training Metrics
            writer.add_scalar('train/01_primary/total_loss', train_metrics['total_loss'], epoch)
            writer.add_scalar('train/01_primary/learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # 02. Loss Components
            writer.add_scalar('train/02_loss_components/ce_loss', train_metrics['ce_loss'], epoch)
            writer.add_scalar('train/02_loss_components/dice_loss', train_metrics['dice_loss'], epoch)

            # 03. Auxiliary Task Metrics (if available)
            if 'aux_fg_bg_loss' in train_metrics:
                writer.add_scalar('train/03_auxiliary/fg_bg_loss', train_metrics['aux_fg_bg_loss'], epoch)
                writer.add_scalar('train/03_auxiliary/fg_accuracy', train_metrics['aux_fg_accuracy'], epoch)
                writer.add_scalar('train/03_auxiliary/fg_iou', train_metrics['aux_fg_iou'], epoch)

            # 04. Refinement Losses (if available)
            if 'active_contour' in train_metrics:
                writer.add_scalar('train/04_refinement/active_contour', train_metrics['active_contour'], epoch)
            if 'boundary_aware' in train_metrics:
                writer.add_scalar('train/04_refinement/boundary_aware', train_metrics['boundary_aware'], epoch)
            if 'contour' in train_metrics:
                writer.add_scalar('train/04_refinement/contour', train_metrics['contour'], epoch)
            if 'distance_transform' in train_metrics:
                writer.add_scalar('train/04_refinement/distance_transform', train_metrics['distance_transform'], epoch)

            # 05. Other Metrics
            if 'grad_norm' in train_metrics:
                writer.add_scalar('train/05_other/gradient_norm', train_metrics['grad_norm'], epoch)

            # Validation
            if epoch % config.training.validate_every == 0:
                val_metrics = evaluate_model(
                    model, val_loader, loss_fn, device,
                    feature_extractor=feature_extractor,
                    config=config,
                    epoch=epoch,
                    output_dir=exp_dirs['checkpoints']
                )

                # Log validation metrics with hierarchical naming for better organization

                # 01. Primary Metrics (most important)
                writer.add_scalar('val/01_primary/target_iou', val_metrics['target_iou'], epoch)
                writer.add_scalar('val/01_primary/detection_rate_0.5', val_metrics['detection_rate_0.5'], epoch)
                writer.add_scalar('val/01_primary/detection_rate_0.7', val_metrics['detection_rate_0.7'], epoch)

                # 02. Target Performance
                writer.add_scalar('val/02_target/precision', val_metrics.get('target_precision', 0), epoch)
                writer.add_scalar('val/02_target/recall', val_metrics.get('target_recall', 0), epoch)
                writer.add_scalar('val/02_target/f1_score', val_metrics.get('target_f1', 0), epoch)

                # 03. Instance Separation
                writer.add_scalar('val/03_instance/separation_accuracy',
                                val_metrics.get('instance_separation_accuracy', 0), epoch)

                # 04. Class-wise IoU
                class_names = ['background', 'target', 'nontarget']
                for i in range(config.model.num_classes):
                    writer.add_scalar(f'val/04_class_iou/{i}_{class_names[i]}',
                                    val_metrics[f'iou_class_{i}'], epoch)

                # 05. Loss Components
                writer.add_scalar('val/05_loss/total', val_metrics['total_loss'], epoch)
                writer.add_scalar('val/05_loss/ce', val_metrics.get('ce_loss', 0), epoch)
                writer.add_scalar('val/05_loss/dice', val_metrics.get('dice_loss', 0), epoch)

                # 06. Overall Metrics
                writer.add_scalar('val/06_overall/accuracy', val_metrics.get('overall_accuracy', 0), epoch)

                # Legacy metric for backward compatibility
                writer.add_scalar('val/miou', val_metrics['miou'], epoch)

                # 07. Auxiliary Task Metrics (if available)
                if 'aux_fg_bg_loss' in val_metrics:
                    writer.add_scalar('val/07_auxiliary/fg_bg_loss', val_metrics['aux_fg_bg_loss'], epoch)
                    writer.add_scalar('val/07_auxiliary/fg_accuracy', val_metrics['aux_fg_accuracy'], epoch)
                    writer.add_scalar('val/07_auxiliary/fg_iou', val_metrics['aux_fg_iou'], epoch)

                # 08. Refinement Losses (if available)
                if 'active_contour' in val_metrics:
                    writer.add_scalar('val/08_refinement/active_contour', val_metrics['active_contour'], epoch)
                if 'boundary_aware' in val_metrics:
                    writer.add_scalar('val/08_refinement/boundary_aware', val_metrics['boundary_aware'], epoch)
                if 'contour' in val_metrics:
                    writer.add_scalar('val/08_refinement/contour', val_metrics['contour'], epoch)
                if 'distance_transform' in val_metrics:
                    writer.add_scalar('val/08_refinement/distance_transform', val_metrics['distance_transform'], epoch)

                print(f"\nEpoch {epoch+1} - Validation:")
                print(f"  Loss: {val_metrics['total_loss']:.4f}")
                print(f"  Target IoU (mIoU): {val_metrics['target_iou']:.4f}")
                print(f"  Detection Rates: IoU>0.5: {val_metrics['detection_rate_0.5']:.2%}, IoU>0.7: {val_metrics['detection_rate_0.7']:.2%}")
                print(f"  Target Metrics: Prec={val_metrics.get('target_precision', 0):.3f}, Rec={val_metrics.get('target_recall', 0):.3f}, F1={val_metrics.get('target_f1', 0):.3f}")
                print(f"  Instance Separation: {val_metrics.get('instance_separation_accuracy', 0):.3f}")

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
                    # If using distillation, save student model only
                    model_to_save = model.get_student() if isinstance(model, DistillationModelWrapper) else model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
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
                # If using distillation, save student model only
                model_to_save = model.get_student() if isinstance(model, DistillationModelWrapper) else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
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