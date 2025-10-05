#!/usr/bin/env python3
"""Export RGB Hierarchical UNet V2 checkpoints to ONNX format.

This script exports trained checkpoints from B0/B1/B7 architectures to ONNX format.
The output ONNX file follows the same structure as untrained_opt_model.onnx.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import argparse
from pathlib import Path
import json
import re
import sys
from typing import Sequence

import numpy as np
from onnx import helper, numpy_helper

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from src.human_edge_detection.export_onnx_advanced import AdvancedONNXExporter
from src.human_edge_detection.experiments.config_manager import ConfigManager
from train_advanced import build_model


def detect_architecture_from_path(checkpoint_path: str) -> str:
    """Detect architecture (B0/B1/B7) from checkpoint path or filename."""
    path_str = str(checkpoint_path).lower()

    # Check for explicit patterns in filename
    if 'best_model_b0' in path_str or '_b0_' in path_str or 'from_b0' in path_str:
        return 'B0'
    elif 'best_model_b1' in path_str or '_b1_' in path_str or 'from_b1' in path_str:
        return 'B1'
    elif 'best_model_b7' in path_str or '_b7_' in path_str or 'from_b7' in path_str:
        return 'B7'

    # Try to detect from experiment directory name
    if 'from_b0' in path_str:
        return 'B0'
    elif 'from_b1' in path_str:
        return 'B1'
    elif 'from_b7' in path_str:
        return 'B7'

    # Default to B0 if not detected
    print(f"Warning: Could not detect architecture from path, defaulting to B0")
    return 'B0'


def get_experiment_config_from_architecture(arch: str, roi_size: tuple, mask_size: tuple) -> str:
    """Get the appropriate experiment config name based on architecture and sizes."""
    # Map architecture to config name patterns
    roi_str = f"r{roi_size[0]}x{roi_size[1]}"
    mask_str = f"m{mask_size[0]}x{mask_size[1]}"

    # Find matching config
    config_patterns = [
        f"rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_{roi_str}{mask_str}_disttrans_contdet_baware_from_{arch}",
        f"rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_{roi_str}{mask_str}_disttrans_contdet_baware_from_{arch}_enhanced"
    ]

    # Try to find exact match
    available_configs = ConfigManager.list_configs()

    for pattern in config_patterns:
        if pattern in available_configs:
            return pattern

    # If no exact match, use the closest one
    # Default configs for each architecture
    defaults = {
        'B0': 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0',
        'B1': 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B1',
        'B7': 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7',
    }

    return defaults.get(arch, defaults['B0'])


class MaskDilationModule(nn.Module):
    """Post-processing module for mask dilation using MaxPool."""

    def __init__(self, dilation_pixels: int = 1):
        """Initialize dilation module.

        Args:
            dilation_pixels: Number of pixels to dilate (0 means no dilation)
        """
        super().__init__()
        self.dilation_pixels = dilation_pixels

        if dilation_pixels > 0:
            # kernel_size = 2 * dilation_pixels + 1 for symmetric dilation
            kernel_size = 2 * dilation_pixels + 1
            self.maxpool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=1,
                padding=dilation_pixels
            )
        else:
            self.maxpool = None

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply dilation to masks.

        Args:
            masks: Segmentation logits of shape [N, 3, H, W]

        Returns:
            Dilated masks of same shape [N, 3, H, W]
        """
        if self.maxpool is None or self.dilation_pixels == 0:
            return masks

        # Apply softmax to get probabilities
        probs = F.softmax(masks, dim=1)

        # Extract target class (class 1) probabilities
        target_probs = probs[:, 1:2, :, :]  # [N, 1, H, W]

        # Apply dilation via MaxPool
        dilated_target = self.maxpool(target_probs)

        # Update logits for target class where dilation occurred
        # Increase confidence at dilated pixels
        mask_diff = (dilated_target - target_probs) > 0.1  # Threshold for new pixels

        # Boost target class logits at dilated pixels
        masks = masks.clone()
        masks[:, 1:2, :, :] = torch.where(
            mask_diff,
            masks[:, 1:2, :, :] + 2.0,  # Boost confidence
            masks[:, 1:2, :, :]
        )

        return masks


class ModelWithDilation(nn.Module):
    """Wrapper to combine model with dilation post-processing."""

    def __init__(self, base_model: nn.Module, dilation_pixels: int = 1):
        """Initialize model with dilation.

        Args:
            base_model: The base segmentation model
            dilation_pixels: Number of pixels to dilate
        """
        super().__init__()
        self.base_model = base_model
        self.dilation = MaskDilationModule(dilation_pixels) if dilation_pixels > 0 else None

    def forward(self, images: torch.Tensor, rois: torch.Tensor):
        """Forward pass with optional dilation.

        Args:
            images: Input images [B, 3, H, W]
            rois: ROI coordinates [N, 5]

        Returns:
            Segmentation masks with optional dilation
        """
        # Get base model output
        output = self.base_model(images, rois)

        # Handle tuple outputs (with auxiliary)
        if isinstance(output, tuple):
            masks = output[0]
            if self.dilation is not None:
                masks = self.dilation(masks)
            return (masks,) + output[1:] if len(output) > 1 else masks
        else:
            # Single output
            if self.dilation is not None:
                output = self.dilation(output)
            return output


def extract_sizes_from_checkpoint_path(checkpoint_path: str) -> tuple:
    """Extract ROI and mask sizes from checkpoint path if available."""
    path_str = str(checkpoint_path)

    # Try to extract from experiment directory name
    # Pattern: r{height}x{width}m{height}x{width}
    roi_pattern = r'r(\d+)x(\d+)'
    mask_pattern = r'm(\d+)x(\d+)'

    roi_match = re.search(roi_pattern, path_str)
    mask_match = re.search(mask_pattern, path_str)

    roi_size = (64, 48)  # Default
    mask_size = (128, 96)  # Default

    if roi_match:
        roi_size = (int(roi_match.group(1)), int(roi_match.group(2)))
    if mask_match:
        mask_size = (int(mask_match.group(1)), int(mask_match.group(2)))

    return roi_size, mask_size


def replace_target_batchnorms_with_affine(onnx_model_path: Path, target_node_names: Sequence[str]) -> int:
    """Replace specific BatchNormalization nodes with Mul/Add affine operations.

    Args:
        onnx_model_path: Path to the ONNX model to modify.
        target_node_names: Iterable of BatchNormalization node names to replace.

    Returns:
        Number of BatchNormalization nodes replaced.
    """
    if not onnx_model_path.exists():
        print(f"⚠️ ONNX model not found for BatchNormalization replacement: {onnx_model_path}")
        return 0

    try:
        model = onnx.load(str(onnx_model_path))
    except Exception as exc:
        print(f"⚠️ Failed to load ONNX model for BatchNormalization replacement: {exc}")
        return 0

    graph = model.graph
    target_names = set(target_node_names)
    initializer_map = {init.name: init for init in graph.initializer}

    existing_names = set(initializer_map.keys())
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        existing_names.add(value.name)
    for node in graph.node:
        existing_names.update(node.output)

    def _unique_name(base: str) -> str:
        candidate = base
        idx = 0
        while candidate in existing_names:
            idx += 1
            candidate = f"{base}_{idx}"
        existing_names.add(candidate)
        return candidate

    replaced = 0
    new_nodes = []

    for node in graph.node:
        if node.name in target_names and node.op_type == 'BatchNormalization':
            if len(node.input) < 5:
                print(f"⚠️ Skipping BatchNormalization replacement for {node.name}: insufficient inputs")
                new_nodes.append(node)
                continue

            data_name, scale_name, bias_name, mean_name, var_name = node.input[:5]
            missing_initializers = [
                name for name in (scale_name, bias_name, mean_name, var_name)
                if name not in initializer_map
            ]

            if missing_initializers:
                print(
                    f"⚠️ Skipping BatchNormalization replacement for {node.name}: missing initializers {missing_initializers}"
                )
                new_nodes.append(node)
                continue

            epsilon = 1e-5
            for attr in node.attribute:
                if attr.name == 'epsilon':
                    if attr.HasField('f'):
                        epsilon = attr.f
                    elif attr.HasField('i'):
                        epsilon = float(attr.i)

            scale = numpy_helper.to_array(initializer_map[scale_name]).astype(np.float32)
            bias = numpy_helper.to_array(initializer_map[bias_name]).astype(np.float32)
            mean = numpy_helper.to_array(initializer_map[mean_name]).astype(np.float32)
            var = numpy_helper.to_array(initializer_map[var_name]).astype(np.float32)

            denom = np.sqrt(var + epsilon).astype(np.float32)
            affine_scale = scale / denom
            affine_bias = bias - mean * affine_scale

            affine_scale = affine_scale.reshape(1, -1, 1, 1)
            affine_bias = affine_bias.reshape(1, -1, 1, 1)

            affine_scale_name = _unique_name(f"{node.name}_scale")
            affine_bias_name = _unique_name(f"{node.name}_bias")
            mul_output_name = _unique_name(f"{node.name}_mul_output")

            graph.initializer.extend([
                numpy_helper.from_array(affine_scale, affine_scale_name),
                numpy_helper.from_array(affine_bias, affine_bias_name),
            ])

            mul_node = helper.make_node(
                'Mul',
                inputs=[data_name, affine_scale_name],
                outputs=[mul_output_name],
                name=_unique_name(f"{node.name}_Mul")
            )

            primary_output = node.output[0]
            add_node = helper.make_node(
                'Add',
                inputs=[mul_output_name, affine_bias_name],
                outputs=[primary_output],
                name=_unique_name(f"{node.name}_Add")
            )

            new_nodes.extend([mul_node, add_node])
            replaced += 1
        else:
            new_nodes.append(node)

    if replaced:
        graph.ClearField('node')
        graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_model_path))
        print(
            f"✅ Replaced {replaced} BatchNormalization node(s) with Mul/Add primitives in {onnx_model_path}"
        )
    else:
        print("ℹ️ No targeted BatchNormalization nodes found for replacement")

    return replaced


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    architecture: str = None,
    simplify: bool = True,
    opset_version: int = 16,
    batch_size: int = 1,
    dilation_pixels: int = 0,
    image_size=(640, 640)
) -> bool:
    """Export RGB Hierarchical UNet V2 checkpoint to ONNX.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        output_path: Output ONNX file path (optional, auto-generated if None)
        architecture: Model architecture ('B0', 'B1', or 'B7'). Auto-detected if None.
        simplify: Whether to simplify the ONNX model with onnxsim
        opset_version: ONNX opset version
        batch_size: Batch size for export
        dilation_pixels: Number of pixels to dilate masks (0 = no dilation)
        image_size: Input image size - int for square or (height, width) tuple

    Returns:
        Success status
    """
    checkpoint_path = Path(checkpoint_path)

    # Handle image size - convert to (height, width) tuple
    if isinstance(image_size, int):
        image_height = image_size
        image_width = image_size
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_height, image_width = image_size
    else:
        print(f"Error: image_size must be int or (height, width) tuple, got {image_size}")
        return False

    # Check if checkpoint exists
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return False

    # Detect architecture
    if architecture is None:
        architecture = detect_architecture_from_path(checkpoint_path)
    else:
        architecture = architecture.upper()

    if architecture not in ['B0', 'B1', 'B7']:
        print(f"Error: Invalid architecture {architecture}. Must be B0, B1, or B7.")
        return False

    print(f"Using architecture: {architecture}")

    # Generate output path if not provided
    if output_path is None:
        # Use checkpoint filename but replace extension with .onnx
        output_path = checkpoint_path.with_suffix('.onnx')
    else:
        output_path = Path(output_path)

    print(f"Output path: {output_path}")

    # Extract ROI and mask sizes from checkpoint path
    roi_size, mask_size = extract_sizes_from_checkpoint_path(checkpoint_path)
    print(f"ROI size: {roi_size}, Mask size: {mask_size}")

    # Get appropriate experiment config
    config_name = get_experiment_config_from_architecture(architecture, roi_size, mask_size)
    print(f"Using experiment config: {config_name}")

    # Load experiment config
    exp_config = ConfigManager.get_config(config_name)

    if exp_config is None:
        print(f"Error: Could not load experiment config: {config_name}")
        return False

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Build model using the experiment config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = build_model(exp_config, device)

    # Debug: print model type
    print(f"Model type: {type(model).__name__}")
    print(f"Model has rgb_extractor: {hasattr(model, 'rgb_extractor')}")
    print(f"Model has segmentation_head: {hasattr(model, 'segmentation_head')}")

    # Load model weights
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint, strict=True)
        print("Model weights loaded successfully")
    except RuntimeError as e:
        print(f"Warning: Failed to load with strict=True, trying strict=False")
        print(f"Error: {str(e)[:200]}...")
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Model weights loaded with strict=False")

    # Set model to eval mode
    model.eval()

    # Wrap model with dilation if requested
    if dilation_pixels > 0:
        print(f"Adding {dilation_pixels}-pixel dilation post-processing")
        model = ModelWithDilation(model, dilation_pixels)
        model.eval()

    # Create ONNX exporter
    exporter = AdvancedONNXExporter(
        model=model,
        model_type='hierarchical',
        device=device,
        include_auxiliary=False,  # Don't include auxiliary outputs for cleaner ONNX
        mask_size=mask_size,
        image_size=(image_height, image_width)
    )

    # Export to ONNX
    print(f"\nExporting to ONNX format...")
    success = exporter.export(
        output_path=str(output_path),
        batch_size=batch_size,
        opset_version=opset_version,
        verify=False  # Skip verification for RGB models
    )

    if success:
        print(f"✅ Successfully exported to: {output_path}")
        #
        target_batchnorm_nodes = [
            '/model/base_model/segmentation_head/base_head/upsample_bg_fg/upsample_bg_fg.1/BatchNormalization',
            '/model/base_model/segmentation_head/base_head/target_vs_nontarget_branch.4/BatchNormalization',
            '/model/segmentation_head/base_head/upsample_bg_fg/upsample_bg_fg.1/BatchNormalization',
            '/model/segmentation_head/base_head/target_vs_nontarget_branch.4/BatchNormalization',
        ]
        bn_replacements = replace_target_batchnorms_with_affine(output_path, target_batchnorm_nodes)
        if bn_replacements != len(target_batchnorm_nodes):
            print(
                f"ℹ️ Replaced {bn_replacements} of {len(target_batchnorm_nodes)} targeted BatchNormalization nodes."
            )

        # Simplify if requested
        if simplify:
            try:
                import onnxsim
                print("\nSimplifying ONNX model with onnxsim...")

                onnx_model = onnx.load(str(output_path))
                model_sim, check = onnxsim.simplify(
                    onnx_model,
                    check_n=3,
                    skip_fuse_bn=False,
                    skip_shape_inference=False
                )

                if check:
                    onnx.save(model_sim, str(output_path))
                    print("✅ ONNX model simplified successfully")
                else:
                    print("⚠️ Simplification check failed, keeping original model")
            except ImportError:
                print("⚠️ onnxsim not available, skipping simplification")
            except Exception as e:
                print(f"⚠️ Simplification failed: {e}")

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'checkpoint_path': str(checkpoint_path),
            'architecture': architecture,
            'roi_size': list(roi_size),
            'mask_size': list(mask_size),
            'experiment_config': config_name,
            'dilation_pixels': dilation_pixels,
            'image_size': [image_height, image_width],
            'input_format': {
                'images': f'[{batch_size}, 3, {image_height}, {image_width}] - RGB input images',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            },
            'output_format': {
                'masks': f'[N, 3, {mask_size[0]}, {mask_size[1]}] - Segmentation logits for each ROI',
                'binary_masks': f'[B, 1, {image_height}, {image_width}] - Binary foreground/background masks from pretrained UNet'
            }
        }

        # Add checkpoint info if available
        if isinstance(checkpoint, dict):
            if 'epoch' in checkpoint:
                metadata['epoch'] = checkpoint['epoch']
            if 'best_miou' in checkpoint:
                metadata['best_miou'] = float(checkpoint['best_miou'])
            if 'val_loss' in checkpoint:
                metadata['val_loss'] = float(checkpoint['val_loss'])

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Metadata saved to: {metadata_path}")

        # Print model info
        model_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\nModel size: {model_size:.2f} MB")

        return True
    else:
        print(f"❌ Export failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Export RGB Hierarchical UNet V2 checkpoints to ONNX format'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint file (.pth)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output ONNX file path (default: same as checkpoint with .onnx extension)'
    )
    parser.add_argument(
        '--architecture', '--arch', '-a',
        type=str,
        choices=['B0', 'B1', 'B7', 'b0', 'b1', 'b7'],
        default=None,
        help='Model architecture (auto-detected if not specified)'
    )
    parser.add_argument(
        '--no_simplify',
        action='store_true',
        help='Skip ONNX simplification'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=16,
        help='ONNX opset version (default: 16)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for export (default: 1)'
    )
    parser.add_argument(
        '--dilation_pixels',
        type=int,
        default=0,
        help='Number of pixels to dilate masks in post-processing (0 = no dilation, default: 0)'
    )
    parser.add_argument(
        '--image_size',
        type=str,
        default='640,640',
        help='Input image size - single value for square (e.g., 640) or "height,width" (e.g., 480,640). Default: 640,640'
    )

    args = parser.parse_args()

    # Parse image_size argument
    if ',' in args.image_size:
        # Parse as height,width
        parts = args.image_size.split(',')
        if len(parts) != 2:
            print(f"Error: image_size must be a single value or 'height,width' format")
            sys.exit(1)
        try:
            image_size = (int(parts[0]), int(parts[1]))
        except ValueError:
            print(f"Error: Invalid image_size format: {args.image_size}")
            sys.exit(1)
    else:
        # Parse as single value for square image
        try:
            image_size = int(args.image_size)
        except ValueError:
            print(f"Error: Invalid image_size format: {args.image_size}")
            sys.exit(1)

    # Run export
    success = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        architecture=args.architecture,
        simplify=not args.no_simplify,
        opset_version=args.opset,
        batch_size=args.batch_size,
        dilation_pixels=args.dilation_pixels,
        image_size=image_size
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
