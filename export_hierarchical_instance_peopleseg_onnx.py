#!/usr/bin/env python3
"""Export RGB Hierarchical UNet V2 checkpoints to ONNX format.

This script exports trained checkpoints from B0/B1/B7 architectures to ONNX format.
The output ONNX file follows the same structure as untrained_opt_model.onnx.
"""

import torch
import onnx
import argparse
from pathlib import Path
import json
import re
import sys

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


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str = None,
    architecture: str = None,
    simplify: bool = True,
    opset_version: int = 16,
    batch_size: int = 1
) -> bool:
    """Export RGB Hierarchical UNet V2 checkpoint to ONNX.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        output_path: Output ONNX file path (optional, auto-generated if None)
        architecture: Model architecture ('B0', 'B1', or 'B7'). Auto-detected if None.
        simplify: Whether to simplify the ONNX model with onnxsim
        opset_version: ONNX opset version
        batch_size: Batch size for export

    Returns:
        Success status
    """
    checkpoint_path = Path(checkpoint_path)

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

    # Create ONNX exporter
    exporter = AdvancedONNXExporter(
        model=model,
        model_type='hierarchical',
        device=device,
        include_auxiliary=False,  # Don't include auxiliary outputs for cleaner ONNX
        mask_size=mask_size
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
            'input_format': {
                'images': f'[{batch_size}, 3, 640, 640] - RGB input images',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            },
            'output_format': {
                'masks': f'[N, 3, {mask_size[0]}, {mask_size[1]}] - Segmentation logits for each ROI'
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

    args = parser.parse_args()

    # Run export
    success = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        architecture=args.architecture,
        simplify=not args.no_simplify,
        opset_version=args.opset,
        batch_size=args.batch_size
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()