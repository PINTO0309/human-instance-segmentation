#!/usr/bin/env python3
"""
Export trained model to ONNX format for inference.
Auxiliary branch is excluded for inference efficiency as it's only needed during training.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.onnx
import onnxruntime as ort
import numpy as np
import onnx

try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
from src.human_edge_detection.model import create_model
from src.human_edge_detection.advanced.multi_scale_extractor import MultiScaleYOLOFeatureExtractor
from src.human_edge_detection.advanced.multi_scale_model import create_multiscale_model
from src.human_edge_detection.advanced.variable_roi_model import (
    create_variable_roi_model, create_rgb_enhanced_variable_roi_model
)
from src.human_edge_detection.advanced.auxiliary_fg_bg_task import MultiTaskSegmentationModel
from src.human_edge_detection.experiments.config_manager import ConfigManager


class InferenceOnlyWrapper(nn.Module):
    """Wrapper for ONNX export that excludes auxiliary branch for inference efficiency."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # Extract only the main segmentation head
        if isinstance(model, MultiTaskSegmentationModel):
            self.model = model.main_head
            self.is_multitask_wrapped = True
        else:
            self.model = model
            self.is_multitask_wrapped = False

    def forward(self, *args) -> torch.Tensor:
        """Forward pass returning only main segmentation output.

        Handles various input signatures:
        - (features, roi_indices) for standard models
        - (features, rois) for hierarchical models
        - (layer1, layer2, ..., layerN, rois) for multi-scale models
        """
        # Determine input structure
        if len(args) == 2:
            # Standard two-argument case
            features, rois_or_indices = args
            output = self.model(features, rois_or_indices)
        elif len(args) > 2:
            # Multi-scale case with multiple feature layers
            *feature_layers, rois = args
            # Reconstruct feature dict from layers
            # Assume layers are in the order defined by config
            features = {}
            layer_names = ['layer_3', 'layer_22', 'layer_34']  # Default order
            for i, feat in enumerate(feature_layers):
                if i < len(layer_names):
                    features[layer_names[i]] = feat
            output = self.model(features, rois)
        else:
            # Single argument case
            output = self.model(args[0])

        # Handle models that might still return tuple
        if isinstance(output, tuple):
            return output[0]  # Return only main output
        else:
            return output


def export_model_inference_only(
    checkpoint_path: str,
    output_path: str,
    config_name: Optional[str] = None,
    device: str = 'cuda',
    opset_version: int = 16
):
    """Export model to ONNX format for inference only (no auxiliary branch).

    Args:
        checkpoint_path: Path to model checkpoint
        output_path: Path for output ONNX file
        config_name: Configuration name (if not in checkpoint)
        device: Device to use for export
        opset_version: ONNX opset version
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get configuration
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        from src.human_edge_detection.experiments.config_manager import ExperimentConfig
        config = ExperimentConfig.from_dict(config_dict)
    elif config_name:
        config = ConfigManager.get_config(config_name)
    else:
        raise ValueError("No config in checkpoint and no config_name provided")

    print(f"Configuration: {config.name}")
    print(f"Auxiliary task enabled: {config.auxiliary_task.enabled}")

    # Build model based on configuration - must match training structure exactly
    print("Building model...")

    # Import necessary builders - avoid circular imports
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    from train_advanced import build_model

    # Build the model using the same logic as training
    model, feature_extractor = build_model(config, device)

    # The model is already wrapped with auxiliary task if enabled
    print(f"Model type: {type(model).__name__}")

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Wrap for ONNX export (inference only, no auxiliary branch)
    onnx_model = InferenceOnlyWrapper(model)

    # Create dummy inputs based on model type
    print("Creating dummy inputs...")

    # For hierarchical UNet models with external features, we need feature dict and ROIs
    if any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
        if config.model.use_external_features and config.multiscale.enabled:
            # Multi-scale features for hierarchical models
            dummy_features = {}
            for layer in config.multiscale.target_layers:
                if layer == 'layer_3':
                    channels = 256
                elif layer == 'layer_22':
                    channels = 512
                elif layer == 'layer_34':
                    channels = 1024
                else:
                    channels = 1024
                dummy_features[layer] = torch.randn(1, channels, 80, 80, device=device)

            # ROIs in normalized coordinates [batch_idx, x1, y1, x2, y2]
            dummy_rois = torch.tensor([
                [0, 0.1, 0.1, 0.3, 0.3],
                [0, 0.5, 0.5, 0.8, 0.8]
            ], device=device, dtype=torch.float32)
        else:
            # Images for models with integrated extractors
            dummy_features = torch.randn(1, 3, 640, 640, device=device)
            dummy_rois = torch.tensor([
                [0, 64, 64, 192, 192],
                [0, 320, 320, 512, 512]
            ], device=device, dtype=torch.float32)
    elif config.multiscale.enabled:
        # Multi-scale features
        dummy_features = {
            layer: torch.randn(4, 1024, 80, 80, device=device)
            for layer in config.multiscale.target_layers
        }
        dummy_roi_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        dummy_rois = dummy_roi_indices  # For backward compatibility
    elif config.model.variable_roi_sizes is not None:
        # Variable ROI features
        dummy_features = {
            layer: torch.randn(4, 1024, 80, 80, device=device)
            for layer in config.model.variable_roi_sizes.keys()
        }
        dummy_roi_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        dummy_rois = dummy_roi_indices
    else:
        # Standard features
        # Get feature channels from config
        if hasattr(config.model, 'feature_channels'):
            feat_channels = config.model.feature_channels
        else:
            feat_channels = 1024  # Default YOLO feature channels
        dummy_features = torch.randn(4, feat_channels,
                                    config.model.roi_size,
                                    config.model.roi_size,
                                    device=device)
        dummy_roi_indices = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.long)
        dummy_rois = dummy_roi_indices

    # Export to ONNX
    print(f"Exporting to ONNX (opset {opset_version})...")

    # Handle different input types and create appropriate inputs
    if any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
        # Hierarchical models expect features and rois
        if isinstance(dummy_features, dict):
            # Multi-scale hierarchical model
            input_names = list(dummy_features.keys()) + ['rois']
            dummy_inputs = tuple(dummy_features.values()) + (dummy_rois,)
            # Dynamic axes for multi-scale inputs
            dynamic_axes = {}
            for layer in dummy_features.keys():
                dynamic_axes[layer] = {0: 'batch_size', 2: 'height', 3: 'width'}
            dynamic_axes['rois'] = {0: 'num_rois'}
        else:
            # Standard hierarchical model
            input_names = ['features', 'rois']
            dummy_inputs = (dummy_features, dummy_rois)
            dynamic_axes = {
                'features': {0: 'batch_size'},
                'rois': {0: 'num_rois'}
            }
    elif isinstance(dummy_features, dict):
        # For multi-scale/variable ROI models
        input_names = list(dummy_features.keys()) + ['roi_indices']
        dummy_inputs = tuple(dummy_features.values()) + (dummy_rois,)
        # Dynamic axes
        dynamic_axes = {}
        for layer in dummy_features.keys():
            dynamic_axes[layer] = {0: 'batch_size'}
        dynamic_axes['roi_indices'] = {0: 'num_rois'}
    else:
        # For standard models
        # Check if this is an RGB model (features are images)
        if dummy_features.shape[1] == 3 and dummy_features.shape[2] == 640 and dummy_features.shape[3] == 640:
            # RGB model - use 'images' for input name
            input_names = ['images', 'rois']
            dynamic_axes = {
                'images': {0: 'batch_size'},
                'rois': {0: 'num_rois'}
            }
        else:
            # Standard model - features are per-ROI
            input_names = ['features', 'roi_indices']
            dynamic_axes = {
                'features': {0: 'num_rois'},
                'roi_indices': {0: 'num_rois'}
            }
        dummy_inputs = (dummy_features, dummy_rois)

    # Output configuration (main output only for inference)
    dynamic_axes['masks'] = {0: 'num_rois'}
    output_names = ['masks']
    # Auxiliary output excluded for inference efficiency

    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        verbose=False
    )

    print(f"Model exported to {output_path}")

    # Simplify with onnxsim if available
    if ONNXSIM_AVAILABLE:
        print("\nSimplifying model with onnxsim...")
        try:
            model_onnx = onnx.load(output_path)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, output_path)
                print("Model simplified successfully!")
            else:
                print("Model simplification check failed, keeping original")
        except Exception as e:
            print(f"Warning: Model simplification failed: {e}")
            print("Continuing with original model...")
    else:
        print("\nNote: onnxsim not available. Install with: pip install onnx-simplifier")

    # Verify the exported model
    print("\nVerifying exported model...")
    try:
        # Create ONNX runtime session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(output_path, providers=providers)

        # Get input/output info
        print("\nModel inputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")

        print("\nModel outputs:")
        for out in session.get_outputs():
            print(f"  {out.name}: {out.shape} ({out.type})")

        # Run inference - prepare inputs based on model type
        if any(getattr(config.model, attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4']):
            # Hierarchical models
            if isinstance(dummy_features, dict):
                inputs = {name: feat.cpu().numpy() for name, feat in dummy_features.items()}
                inputs['rois'] = dummy_rois.cpu().numpy()
            else:
                # Check if RGB model
                if dummy_features.shape[1] == 3 and dummy_features.shape[2] == 640 and dummy_features.shape[3] == 640:
                    inputs = {
                        'images': dummy_features.cpu().numpy(),
                        'rois': dummy_rois.cpu().numpy()
                    }
                else:
                    inputs = {
                        'features': dummy_features.cpu().numpy(),
                        'rois': dummy_rois.cpu().numpy()
                    }
        elif isinstance(dummy_features, dict):
            inputs = {name: feat.cpu().numpy() for name, feat in dummy_features.items()}
            inputs['roi_indices'] = dummy_rois.cpu().numpy()
        else:
            # Check if RGB model
            if dummy_features.shape[1] == 3 and dummy_features.shape[2] == 640 and dummy_features.shape[3] == 640:
                inputs = {
                    'images': dummy_features.cpu().numpy(),
                    'rois': dummy_rois.cpu().numpy()
                }
            else:
                inputs = {
                    'features': dummy_features.cpu().numpy(),
                    'roi_indices': dummy_rois.cpu().numpy()
                }

        outputs = session.run(None, inputs)

        print(f"\nInference successful!")
        print(f"Masks output shape: {outputs[0].shape}")
        print("(Auxiliary branch excluded from export for inference efficiency)")

        # Save metadata
        metadata = {
            'config_name': config.name,
            'num_classes': config.model.num_classes,
            'mask_size': config.model.mask_size,
            'auxiliary_task_enabled_during_training': config.auxiliary_task.enabled,
            'auxiliary_included_in_export': False,  # Always false for inference efficiency
            'model_type': 'multiscale' if config.multiscale.enabled else 'standard',
            'opset_version': opset_version,
            'export_mode': 'inference_only'
        }

        metadata_path = Path(output_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"\nMetadata saved to {metadata_path}")

    except Exception as e:
        print(f"Warning: Could not verify model: {e}")

    print("\nExport completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX for inference (auxiliary branch excluded)')
    parser.add_argument('checkpoint', help='Path to model checkpoint')
    parser.add_argument('-o', '--output', default=None, help='Output ONNX file path')
    parser.add_argument('-c', '--config', default=None, help='Config name (if not in checkpoint)')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--opset', type=int, default=16, help='ONNX opset version')

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        args.output = checkpoint_path.with_suffix('.onnx')

    export_model_inference_only(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        config_name=args.config,
        device=args.device,
        opset_version=args.opset
    )


if __name__ == '__main__':
    main()