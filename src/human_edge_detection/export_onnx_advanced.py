"""Advanced ONNX export for both baseline and multiscale models."""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import json

from .model import create_model, ROISegmentationHead
from .advanced.multi_scale_model import MultiScaleSegmentationModel

try:
    import onnxsim
    ONNXSIM_AVAILABLE = True
except ImportError:
    ONNXSIM_AVAILABLE = False


class AdvancedONNXExporter:
    """Export both baseline and multiscale models to ONNX format."""

    def __init__(self, model: nn.Module, model_type: str = 'baseline', device: str = 'cpu'):
        """Initialize exporter.

        Args:
            model: Trained segmentation model
            model_type: Type of model ('baseline' or 'multiscale')
            device: Device to use for export
        """
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device

    def export(
        self,
        output_path: str,
        batch_size: int = 1,
        opset_version: int = 16,
        verify: bool = True
    ) -> bool:
        """Export model to ONNX format.

        Args:
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            opset_version: ONNX opset version
            verify: Whether to verify the exported model

        Returns:
            Success status
        """
        if self.model_type == 'baseline':
            return self._export_baseline(output_path, batch_size, opset_version, verify)
        elif self.model_type == 'multiscale':
            return self._export_multiscale(output_path, batch_size, opset_version, verify)
        elif self.model_type == 'variable_roi':
            return self._export_variable_roi(output_path, batch_size, opset_version, verify)
        else:
            print(f"Unknown model type: {self.model_type}")
            return False

    def _export_baseline(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Export baseline model."""
        self.model.eval()

        # Get segmentation head
        seg_head = self.model.segmentation_head
        seg_head.eval()

        # Create dummy inputs
        dummy_features = torch.randn(batch_size, 1024, 80, 80).to(self.device)
        num_rois = 2
        dummy_rois = torch.zeros(num_rois, 5).to(self.device)
        dummy_rois[:, 0] = torch.arange(num_rois) % batch_size
        dummy_rois[:, 1:] = torch.tensor([
            [100, 100, 300, 300],
            [200, 200, 400, 400]
        ]).float()

        print(f"Exporting baseline model to {output_path}")

        try:
            torch.onnx.export(
                seg_head,
                (dummy_features, dummy_rois),
                output_path,
                input_names=['features', 'rois'],
                output_names=['masks'],
                dynamic_axes={
                    'features': {0: 'batch_size'},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois'}
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )

            print("Export successful!")

            if ONNXSIM_AVAILABLE:
                self._simplify_model(output_path)

            if verify:
                return self._verify_baseline(output_path, dummy_features, dummy_rois)

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _export_multiscale(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Export multiscale model.

        Note: Multiscale models are complex and include ONNX models internally.
        We export just the segmentation head part that takes pre-extracted features.
        """
        print("Exporting multiscale model segmentation head...")
        print("Note: This exports only the segmentation head. Feature extraction should be done separately.")

        self.model.eval()

        # Get segmentation head from multiscale model
        seg_head = self.model.segmentation_head
        seg_head.eval()

        # Create dummy inputs for multiscale features
        # Multiscale models expect a dictionary of features
        dummy_features = {}
        if hasattr(self.model, 'ms_extractor') and hasattr(self.model.ms_extractor, 'target_layers'):
            # Create features for each target layer
            for layer_name in self.model.ms_extractor.target_layers:
                if layer_name == 'layer_3':
                    # Higher resolution feature map (256 channels, 160x160)
                    dummy_features[layer_name] = torch.randn(batch_size, 256, 160, 160).to(self.device)
                elif layer_name == 'layer_22':
                    # Mid-level feature map (512 channels, 80x80)
                    dummy_features[layer_name] = torch.randn(batch_size, 512, 80, 80).to(self.device)
                elif layer_name == 'layer_34':
                    # Deep feature map (1024 channels, 80x80)
                    dummy_features[layer_name] = torch.randn(batch_size, 1024, 80, 80).to(self.device)
                else:
                    # Default to 512 channels, 80x80
                    dummy_features[layer_name] = torch.randn(batch_size, 512, 80, 80).to(self.device)
        else:
            # Default features for common YOLO layers
            dummy_features['layer_3'] = torch.randn(batch_size, 256, 160, 160).to(self.device)
            dummy_features['layer_22'] = torch.randn(batch_size, 512, 80, 80).to(self.device)
            dummy_features['layer_34'] = torch.randn(batch_size, 1024, 80, 80).to(self.device)

        num_rois = 2
        dummy_rois = torch.zeros(num_rois, 5).to(self.device)
        dummy_rois[:, 0] = torch.arange(num_rois) % batch_size
        dummy_rois[:, 1:] = torch.tensor([
            [100, 100, 300, 300],
            [200, 200, 400, 400]
        ]).float()

        print(f"Exporting to {output_path}")
        print(f"Feature layers: {list(dummy_features.keys())}")

        # Create a wrapper that handles the dictionary input
        class MultiScaleSegmentationWrapper(nn.Module):
            def __init__(self, seg_head, layer_names):
                super().__init__()
                self.seg_head = seg_head
                self.layer_names = sorted(layer_names)

            def forward(self, *args):
                """Convert tensor inputs back to dictionary for seg_head."""
                # Last argument is rois, others are features
                feature_tensors = args[:-1]
                rois = args[-1]

                # Reconstruct feature dictionary from positional arguments
                features = {name: tensor for name, tensor in zip(self.layer_names, feature_tensors)}
                return self.seg_head(features, rois)

        wrapper = MultiScaleSegmentationWrapper(seg_head, list(dummy_features.keys()))
        wrapper.eval()

        # Prepare inputs as separate tensors
        feature_tensors = [dummy_features[name] for name in sorted(dummy_features.keys())]

        try:
            # Export with all features as separate inputs
            input_names = [f'features_{name}' for name in sorted(dummy_features.keys())] + ['rois']

            torch.onnx.export(
                wrapper,
                tuple(feature_tensors + [dummy_rois]),
                output_path,
                input_names=input_names,
                output_names=['masks'],
                dynamic_axes={
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois'}
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )

            print("Export successful!")
            print("Note: This ONNX model requires pre-extracted multiscale features as input.")

            if ONNXSIM_AVAILABLE:
                self._simplify_model(output_path)

            # Skip verification for now as it's complex with multiple inputs
            if verify:
                print("Verification skipped for multiscale models due to complexity.")

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            print("Multiscale models with dynamic architectures are challenging to export to ONNX.")
            print("Consider using the PyTorch model directly or implementing custom ONNX operators.")
            return False

    def _export_variable_roi(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Export variable ROI model using ONNX-compatible version."""
        print("Exporting variable ROI model with ONNX-compatible operations...")

        self.model.eval()

        # Import ONNX-compatible version
        from .advanced.variable_roi_onnx import create_onnx_variable_roi_segmentation_head

        # Get original segmentation head configuration
        orig_seg_head = self.model.segmentation_head

        # Extract configuration from original model
        feature_channels = {}
        feature_strides = {}
        roi_sizes = {}

        # Get configuration from the model
        if hasattr(orig_seg_head, 'var_roi_align'):
            feature_strides = orig_seg_head.var_roi_align.feature_strides
            roi_sizes = orig_seg_head.var_roi_align.roi_sizes

        # Infer feature channels from the model
        if hasattr(self.model, 'extractor') and hasattr(self.model.extractor, 'FEATURE_SPECS'):
            for layer_id in roi_sizes.keys():
                spec = self.model.extractor.FEATURE_SPECS.get(layer_id, {})
                feature_channels[layer_id] = spec.get('channels', 512)
        else:
            # Default channel configuration
            feature_channels = {
                'layer_3': 256,
                'layer_22': 512,
                'layer_34': 1024
            }

        # Create ONNX-compatible segmentation head
        # Note: We use the original model's segmentation head directly
        # instead of creating a new one and transferring weights
        onnx_seg_head = orig_seg_head
        onnx_seg_head.to(self.device)
        onnx_seg_head.eval()

        # Create dummy inputs
        dummy_features = {}
        for layer_id, channels in feature_channels.items():
            if layer_id in ['layer_3', 'layer_19']:
                # High resolution features
                dummy_features[layer_id] = torch.randn(batch_size, channels, 160, 160).to(self.device)
            else:
                # Standard resolution features
                dummy_features[layer_id] = torch.randn(batch_size, channels, 80, 80).to(self.device)

        num_rois = 2
        dummy_rois = torch.zeros(num_rois, 5).to(self.device)
        dummy_rois[:, 0] = torch.arange(num_rois) % batch_size
        dummy_rois[:, 1:] = torch.tensor([
            [100, 100, 300, 300],
            [200, 200, 400, 400]
        ]).float()

        print(f"Exporting to {output_path}")
        print(f"Feature layers: {list(dummy_features.keys())}")
        print(f"ROI sizes: {roi_sizes}")

        # Create wrapper for ONNX export
        class VariableROIWrapper(nn.Module):
            def __init__(self, seg_head, layer_names):
                super().__init__()
                self.seg_head = seg_head
                self.layer_names = sorted(layer_names)

            def forward(self, *args):
                # Last argument is rois
                feature_tensors = args[:-1]
                rois = args[-1]

                # Reconstruct feature dictionary
                features = {name: tensor for name, tensor in zip(self.layer_names, feature_tensors)}
                return self.seg_head(features, rois)

        wrapper = VariableROIWrapper(onnx_seg_head, list(dummy_features.keys()))
        wrapper.eval()

        # Prepare inputs
        feature_tensors = [dummy_features[name] for name in sorted(dummy_features.keys())]

        try:
            # Export with fixed opset version
            input_names = [f'features_{name}' for name in sorted(dummy_features.keys())] + ['rois']

            torch.onnx.export(
                wrapper,
                tuple(feature_tensors + [dummy_rois]),
                output_path,
                input_names=input_names,
                output_names=['masks'],
                dynamic_axes={
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois'}
                },
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )

            print("Export successful!")

            if ONNXSIM_AVAILABLE:
                self._simplify_model(output_path)

            if verify:
                print("Verification skipped for variable ROI models due to complexity.")

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _simplify_model(self, model_path: str):
        """Simplify ONNX model with onnxsim."""
        print("Simplifying ONNX model with onnxsim...")
        try:
            model_onnx = onnx.load(model_path)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, model_path)
                print("Model simplified successfully!")
            else:
                print("Model simplification check failed, keeping original")
        except Exception as e:
            print(f"Warning: Model simplification failed: {e}")

    def _verify_baseline(
        self,
        onnx_path: str,
        dummy_features: torch.Tensor,
        dummy_rois: torch.Tensor
    ) -> bool:
        """Verify baseline ONNX model."""
        print("Verifying baseline ONNX model...")

        try:
            # Check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure is valid")

            # Test inference
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Prepare inputs
            ort_inputs = {
                'features': dummy_features.cpu().numpy(),
                'rois': dummy_rois.cpu().numpy()
            }

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            print(f"ONNX output shape: {ort_outputs[0].shape}")
            print("Verification passed!")
            return True

        except Exception as e:
            print(f"Verification failed: {e}")
            return False

    def _verify_multiscale(
        self,
        onnx_path: str,
        dummy_images: torch.Tensor,
        dummy_rois: torch.Tensor
    ) -> bool:
        """Verify multiscale ONNX model."""
        print("Verifying multiscale ONNX model...")

        try:
            # Check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure is valid")

            # Test inference
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Prepare inputs
            ort_inputs = {
                'images': dummy_images.cpu().numpy(),
                'rois': dummy_rois.cpu().numpy()
            }

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            print(f"ONNX output shape: {ort_outputs[0].shape}")
            print("Verification passed!")
            return True

        except Exception as e:
            print(f"Verification failed: {e}")
            return False


def export_checkpoint_to_onnx_advanced(
    checkpoint_path: str,
    output_path: str,
    model_type: str,
    config: Optional[Dict] = None,
    device: str = 'cpu',
    verify: bool = True,
    opset_version: int = 16
) -> bool:
    """Export a checkpoint to ONNX format with model type support.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to save ONNX model
        model_type: Type of model ('baseline' or 'multiscale')
        config: Model configuration
        device: Device to use for export
        verify: Whether to verify the exported model
        opset_version: ONNX opset version

    Returns:
        Success status
    """
    print(f"\nExporting {model_type} model to ONNX...")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint if not provided
    if config is None and 'config' in checkpoint:
        config = checkpoint['config']

    # Check if it's a variable ROI model
    if config and isinstance(config, dict):
        model_config = config.get('model', {})
        if model_config.get('variable_roi_sizes'):
            model_type = 'variable_roi'

    if model_type == 'baseline':
        # Create baseline model
        from .model import create_model

        model_config = config.get('model', {}) if config else {}
        model = create_model(
            num_classes=model_config.get('num_classes', 3),
            roi_size=model_config.get('roi_size', 28),
            mask_size=model_config.get('mask_size', 56)
        )

    elif model_type == 'multiscale':
        # Create multiscale model
        from .advanced.multi_scale_model import create_multiscale_model

        model_config = config.get('model', {}) if config else {}
        multiscale_config = config.get('multiscale', {}) if config else {}

        model = create_multiscale_model(
            onnx_model_path=model_config.get('onnx_model', 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'),
            target_layers=multiscale_config.get('target_layers', ['layer_159', 'layer_180']),
            num_classes=model_config.get('num_classes', 3),
            roi_size=model_config.get('roi_size', 28),
            mask_size=model_config.get('mask_size', 56),
            fusion_method=multiscale_config.get('fusion_method', 'adaptive'),
            execution_provider=model_config.get('execution_provider', 'cpu')
        )

    elif model_type == 'variable_roi':
        # Create variable ROI model
        from .advanced.variable_roi_model import create_variable_roi_model

        model_config = config.get('model', {}) if config else {}
        multiscale_config = config.get('multiscale', {}) if config else {}

        model = create_variable_roi_model(
            onnx_model_path=model_config.get('onnx_model', 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'),
            target_layers=multiscale_config.get('target_layers', ['layer_3', 'layer_22', 'layer_34']),
            roi_sizes=model_config.get('variable_roi_sizes', {'layer_3': 56, 'layer_22': 28, 'layer_34': 28}),
            num_classes=model_config.get('num_classes', 3),
            mask_size=model_config.get('mask_size', 56),
            execution_provider=model_config.get('execution_provider', 'cpu')
        )

    else:
        print(f"Unknown model type: {model_type}")
        return False

    # Load model weights with strict=False for architecture changes
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create exporter
    exporter = AdvancedONNXExporter(model, model_type=model_type, device=device)

    # Export to ONNX
    success = exporter.export(
        output_path=output_path,
        verify=verify,
        opset_version=opset_version
    )

    if success:
        print(f"Successfully exported model to {output_path}")

        # Save metadata
        metadata_path = Path(output_path).with_suffix('.json')
        metadata = {
            'model_type': model_type,
            'checkpoint_path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', -1),
            'best_miou': checkpoint.get('best_miou', -1),
            'config': config
        }

        if model_type == 'baseline':
            metadata['input_format'] = {
                'features': '[B, 1024, 80, 80] - YOLO intermediate features',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            }
        else:
            # For multiscale, we export only the segmentation head
            metadata['input_format'] = {
                'features_layer_3': '[B, 256, 160, 160] - High-resolution YOLO features',
                'features_layer_22': '[B, 512, 80, 80] - Mid-level YOLO features',
                'features_layer_34': '[B, 1024, 80, 80] - Deep YOLO features',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            }
            metadata['note'] = 'This model requires pre-extracted YOLO features. Use the YOLO ONNX model separately for feature extraction.'

        metadata['output_format'] = {
            'masks': '[N, 3, 56, 56] - Segmentation logits for each ROI'
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    return success