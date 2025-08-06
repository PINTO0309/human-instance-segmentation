"""Advanced ONNX export for both baseline and multiscale models."""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import json
import warnings

# Suppress TracerWarning messages during ONNX export
warnings.filterwarnings("ignore", category=UserWarning, module="torch.jit._trace")

from .model import create_model, ROISegmentationHead
from .advanced.multi_scale_model import MultiScaleSegmentationModel

try:
    from onnxsim import simplify as onnxsim_simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    try:
        import onnx_simplifier as onnxsim
        ONNXSIM_AVAILABLE = True
    except ImportError:
        ONNXSIM_AVAILABLE = False


class AdvancedONNXExporter:
    """Export both baseline and multiscale models to ONNX format."""

    def __init__(self, model: nn.Module, model_type: str = 'baseline', device: str = 'cpu', include_auxiliary: bool = False, mask_size=None):
        """Initialize exporter.

        Args:
            model: Trained segmentation model
            model_type: Type of model ('baseline' or 'multiscale')
            device: Device to use for export
            include_auxiliary: Whether to include auxiliary outputs in ONNX export
            mask_size: Mask size (int or tuple) from config
        """
        # Handle DistillationModelWrapper - export only student model
        from src.human_edge_detection.advanced.knowledge_distillation import DistillationModelWrapper
        if isinstance(model, DistillationModelWrapper):
            model = model.get_student()
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.include_auxiliary = include_auxiliary
        
        # Extract mask size
        if mask_size is not None:
            self.mask_size = mask_size
        elif hasattr(model, 'segmentation_head') and hasattr(model.segmentation_head, 'mask_size'):
            self.mask_size = model.segmentation_head.mask_size
        elif hasattr(model, 'main_head') and hasattr(model.main_head, 'segmentation_head') and hasattr(model.main_head.segmentation_head, 'mask_size'):
            self.mask_size = model.main_head.segmentation_head.mask_size
        else:
            self.mask_size = 56  # Default
        
        # Convert to tuple if int
        if isinstance(self.mask_size, int):
            self.mask_height = self.mask_size
            self.mask_width = self.mask_size
        else:
            self.mask_height, self.mask_width = self.mask_size

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
        elif self.model_type == 'hierarchical':
            return self._export_hierarchical(output_path, batch_size, opset_version, verify)
        elif self.model_type == 'class_specific':
            return self._export_class_specific(output_path, batch_size, opset_version, verify)
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
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
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
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
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

    def _export_hierarchical(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Export hierarchical segmentation model."""
        print("Exporting hierarchical segmentation model...")

        self.model.eval()

        # Check if this is an RGB hierarchical model
        # Check the actual model class or if it's wrapped in MultiTaskSegmentationModel
        model_to_check = self.model
        if hasattr(self.model, 'main_head'):
            # If wrapped in MultiTaskSegmentationModel, check the main head
            model_to_check = self.model.main_head

        is_rgb_hierarchical = (
            hasattr(model_to_check, 'rgb_extractor') or
            model_to_check.__class__.__name__ in ['HierarchicalRGBSegmentationModel', 'MultiScaleRGBSegmentationModel'] or
            (hasattr(model_to_check, 'segmentation_head') and
             hasattr(model_to_check.segmentation_head, '__class__') and
             'HierarchicalSegmentationHeadUNetV2' in model_to_check.segmentation_head.__class__.__name__)
        )

        if is_rgb_hierarchical:
            # RGB hierarchical models take images as input
            print("Detected RGB hierarchical model - using image input")
            dummy_image = torch.randn(batch_size, 3, 640, 640).to(self.device)
            num_rois = 2
            dummy_rois = torch.zeros(num_rois, 5).to(self.device)
            dummy_rois[:, 0] = torch.arange(num_rois) % batch_size
            dummy_rois[:, 1:] = torch.tensor([
                [100, 100, 300, 300],
                [200, 200, 400, 400]
            ]).float()

            class RGBHierarchicalWrapper(nn.Module):
                def __init__(self, model, include_auxiliary=False):
                    super().__init__()
                    self.model = model
                    self.include_auxiliary = include_auxiliary

                def forward(self, images, rois):
                    output = self.model(images, rois)
                    if isinstance(output, tuple) and self.include_auxiliary:
                        # Extract main output and auxiliary outputs
                        masks = output[0]
                        aux_dict = output[1] if len(output) > 1 else {}

                        # Return specific outputs in the expected order
                        if isinstance(aux_dict, dict):
                            bg_fg_logits = aux_dict.get('bg_fg_logits', torch.zeros_like(masks)[:, :2])
                            target_nontarget_logits = aux_dict.get('target_nontarget_logits', torch.zeros_like(masks)[:, :2])
                            return masks, bg_fg_logits, target_nontarget_logits
                        else:
                            return output
                    elif isinstance(output, tuple):
                        return output[0]  # Return only logits
                    return output

            wrapper = RGBHierarchicalWrapper(self.model, self.include_auxiliary)
            wrapper.eval()

            try:
                # Determine output names and dynamic axes
                if self.include_auxiliary:
                    output_names = ['masks', 'bg_fg_logits', 'target_nontarget_logits']
                    dynamic_axes = {
                        'images': {0: 'batch_size'},
                        'rois': {0: 'num_rois'},
                        'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)},
                        'bg_fg_logits': {0: 'num_rois'},
                        'target_nontarget_logits': {0: 'num_rois'}
                    }
                else:
                    output_names = ['masks']
                    dynamic_axes = {
                        'images': {0: 'batch_size'},
                        'rois': {0: 'num_rois'},
                        'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
                    }

                torch.onnx.export(
                    wrapper,
                    (dummy_image, dummy_rois),
                    output_path,
                    input_names=['images', 'rois'],
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=False
                )

                print("Export successful!")

                if ONNXSIM_AVAILABLE:
                    self._simplify_model(output_path)

                if verify:
                    print("Verification skipped for RGB hierarchical models.")

                return True

            except Exception as e:
                print(f"Export failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        # Non-RGB hierarchical models need hierarchical_head
        if not hasattr(self.model, 'hierarchical_head'):
            print("Model does not have hierarchical head")
            return False

        # Create dummy inputs based on the base model structure
        dummy_features = {}

        # Determine the feature structure from base model
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'extractor'):
            # Variable ROI model structure
            for layer_id in ['layer_3', 'layer_22', 'layer_34']:
                if layer_id in ['layer_3']:
                    dummy_features[layer_id] = torch.randn(batch_size, 256, 160, 160).to(self.device)
                elif layer_id == 'layer_22':
                    dummy_features[layer_id] = torch.randn(batch_size, 512, 80, 80).to(self.device)
                else:
                    dummy_features[layer_id] = torch.randn(batch_size, 1024, 80, 80).to(self.device)
        else:
            # Default features
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

        # Check if this model has integrated feature extractor (MultiScaleSegmentationModel)
        has_integrated_extractor = (
            hasattr(self.model, 'base_model') and
            hasattr(self.model.base_model, 'feature_extractor') and
            not hasattr(self.model.base_model, 'extractor')  # Variable ROI has 'extractor'
        )

        if has_integrated_extractor:
            # For models with integrated feature extractor, use image input
            print("Model has integrated feature extractor - using image input")
            dummy_image = torch.randn(batch_size, 3, 640, 640).to(self.device)

            class HierarchicalImageWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, images, rois):
                    output = self.model(images, rois)
                    if isinstance(output, tuple):
                        return output[0]  # Return only logits for ONNX
                    return output

            wrapper = HierarchicalImageWrapper(self.model)
            wrapper.eval()

            export_inputs = (dummy_image, dummy_rois)
            input_names = ['images', 'rois']

            try:
                torch.onnx.export(
                    wrapper,
                    export_inputs,
                    output_path,
                    input_names=input_names,
                    output_names=['masks'],
                    dynamic_axes={
                        'images': {0: 'batch_size'},
                        'rois': {0: 'num_rois'},
                        'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
                    },
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=False
                )

                print("Export successful!")

                if ONNXSIM_AVAILABLE:
                    self._simplify_model(output_path)

                return True

            except Exception as e:
                print(f"Export failed: {e}")
                import traceback
                traceback.print_exc()
                return False

        # Check if this is a V3 model and if we should use static version for ONNX
        use_static_v3 = False
        if hasattr(self.model, 'hierarchical_head'):
            head_class_name = self.model.hierarchical_head.__class__.__name__
            if 'V3' in head_class_name and 'Static' not in head_class_name:
                print("Detected V3 model - will use static version for clean ONNX export")
                use_static_v3 = True

                # Create static V3 head
                from .advanced.hierarchical_segmentation_unet import HierarchicalSegmentationHeadUNetV3Static
                in_channels = self.model.hierarchical_head.shared_features[0].in_channels
                static_head = HierarchicalSegmentationHeadUNetV3Static(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )

                # Move static head to the same device as the original model
                static_head = static_head.to(self.device)

                # Copy weights from original to static model
                # This is a simplified weight copy - in production you'd want more careful mapping
                print("Note: Weight transfer from dynamic to static V3 not implemented")
                print("For production use, implement proper weight mapping or train static model directly")

                # Replace the head temporarily
                original_head = self.model.hierarchical_head
                self.model.hierarchical_head = static_head

        # Create wrapper that calls the hierarchical model properly
        class HierarchicalWrapper(nn.Module):
            def __init__(self, model, layer_names, include_auxiliary=False):
                super().__init__()
                self.model = model
                self.layer_names = sorted(layer_names)
                self.include_auxiliary = include_auxiliary

            def forward(self, *args):
                # Last argument is rois
                feature_tensors = args[:-1]
                rois = args[-1]

                # Reconstruct feature dictionary
                features = {name: tensor for name, tensor in zip(self.layer_names, feature_tensors)}

                # Call hierarchical model - it returns (logits, aux_outputs)
                output = self.model(features, rois)
                if isinstance(output, tuple) and self.include_auxiliary:
                    # Extract main output and auxiliary outputs
                    masks = output[0]
                    aux_dict = output[1] if len(output) > 1 else {}

                    # Return specific outputs in the expected order
                    if isinstance(aux_dict, dict):
                        bg_fg_logits = aux_dict.get('bg_fg_logits', torch.zeros_like(masks)[:, :2])
                        target_nontarget_logits = aux_dict.get('target_nontarget_logits', torch.zeros_like(masks)[:, :2])
                        return masks, bg_fg_logits, target_nontarget_logits
                    else:
                        return output  # Return all outputs including auxiliary
                elif isinstance(output, tuple):
                    return output[0]  # Return only logits for ONNX
                return output

        wrapper = HierarchicalWrapper(self.model, list(dummy_features.keys()), self.include_auxiliary)
        wrapper.eval()

        # Prepare inputs
        feature_tensors = [dummy_features[name] for name in sorted(dummy_features.keys())]
        export_inputs = tuple(feature_tensors + [dummy_rois])
        input_names = [f'features_{name}' for name in sorted(dummy_features.keys())] + ['rois']

        try:
            # Determine output names and dynamic axes
            if self.include_auxiliary:
                output_names = ['masks', 'bg_fg_logits', 'target_nontarget_logits']
                dynamic_axes = {
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)},
                    'bg_fg_logits': {0: 'num_rois'},
                    'target_nontarget_logits': {0: 'num_rois'}
                }
            else:
                output_names = ['masks']
                dynamic_axes = {
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
                }

            torch.onnx.export(
                wrapper,
                export_inputs,
                output_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )

            print("Export successful!")

            if ONNXSIM_AVAILABLE:
                self._simplify_model(output_path)

            if verify:
                print("Verification skipped for hierarchical models due to complexity.")

            # Restore original head if we used static V3
            if use_static_v3:
                self.model.hierarchical_head = original_head
                print("Restored original V3 head after export")

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            import traceback
            traceback.print_exc()

            # Restore original head on error too
            if use_static_v3 and 'original_head' in locals():
                self.model.hierarchical_head = original_head

            return False

    def _export_class_specific(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Export class-specific decoder model."""
        print("Exporting class-specific decoder model...")

        self.model.eval()

        # Class-specific models are similar to variable ROI but with custom decoder
        return self._export_variable_roi_core(output_path, batch_size, opset_version, verify)

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

        return self._export_variable_roi_core(output_path, batch_size, opset_version, verify)

    def _export_variable_roi_core(
        self,
        output_path: str,
        batch_size: int,
        opset_version: int,
        verify: bool
    ) -> bool:
        """Core export logic for variable ROI-based models."""

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

        # Check if this is an RGB enhanced model
        is_rgb_enhanced = hasattr(self.model, 'rgb_encoder') and self.model.rgb_encoder is not None
        if is_rgb_enhanced and hasattr(orig_seg_head, 'rgb_enhanced_layers'):
            # For RGB enhanced models, the feature channels are already correct in the original model
            # The RGB features are concatenated internally, so we don't need to adjust channels here
            pass

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
        if is_rgb_enhanced:
            # RGB enhanced model - export decoder only
            print("RGB enhanced model detected - exporting decoder only")

            # Create wrapper that includes RGB encoder
            class RGBEnhancedDecoderWrapper(nn.Module):
                def __init__(self, rgb_encoder, seg_head):
                    super().__init__()
                    self.rgb_encoder = rgb_encoder
                    self.seg_head = seg_head

                def forward(self, *args):
                    # Args: features for each layer + rgb_images + rois
                    # The order matches the sorted layer names
                    *feature_tensors, rgb_images, rois = args

                    # Extract RGB features from raw images
                    rgb_features = self.rgb_encoder(rgb_images)

                    # Reconstruct feature dictionary
                    layer_names = sorted(self.seg_head.var_roi_align.feature_strides.keys())
                    features = {name: tensor for name, tensor in zip(layer_names, feature_tensors)}

                    # Call segmentation head with RGB features
                    return self.seg_head(features, rois, rgb_features)

            # Get RGB encoder from the model
            rgb_encoder = self.model.rgb_encoder
            rgb_encoder.eval()

            wrapper = RGBEnhancedDecoderWrapper(rgb_encoder, onnx_seg_head)

            # Prepare dummy inputs for decoder
            # RGB images (B, 3, 640, 640)
            dummy_rgb_images = torch.randn(batch_size, 3, 640, 640).to(self.device)

            # Feature tensors + RGB images + ROIs
            feature_tensors = [dummy_features[name] for name in sorted(dummy_features.keys())]
            export_inputs = tuple(feature_tensors + [dummy_rgb_images, dummy_rois])
            input_names = [f'features_{name}' for name in sorted(dummy_features.keys())] + ['rgb_images', 'rois']

        else:
            # Standard variable ROI model
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

            # Standard model uses pre-extracted features
            feature_tensors = [dummy_features[name] for name in sorted(dummy_features.keys())]
            export_inputs = tuple(feature_tensors + [dummy_rois])
            input_names = [f'features_{name}' for name in sorted(dummy_features.keys())] + ['rois']

        wrapper.eval()

        try:
            # Prepare dynamic axes
            if is_rgb_enhanced:
                # For RGB enhanced decoder
                dynamic_axes = {
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rgb_images': {0: 'batch_size'},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
                }
            else:
                # For standard model
                dynamic_axes = {
                    **{f'features_{name}': {0: 'batch_size'} for name in sorted(dummy_features.keys())},
                    'rois': {0: 'num_rois'},
                    'masks': {0: 'num_rois', 2: str(self.mask_height), 3: str(self.mask_width)}
                }

            # Export with fixed opset version
            torch.onnx.export(
                wrapper,
                export_inputs,
                output_path,
                input_names=input_names,
                output_names=['masks'],
                dynamic_axes=dynamic_axes,
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
            import sys
            import io
            import contextlib

            model_onnx = onnx.load(model_path)

            # Suppress onnxsim output by redirecting stdout and stderr
            f_stdout = io.StringIO()
            f_stderr = io.StringIO()
            with contextlib.redirect_stdout(f_stdout), contextlib.redirect_stderr(f_stderr):
                # Try to use the imported simplify function
                if 'onnxsim_simplify' in globals():
                    model_simp, check = onnxsim_simplify(model_onnx, check_n=3)
                else:
                    model_simp, check = onnxsim.simplify(model_onnx, check_n=3)

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
    opset_version: int = 16,
    include_auxiliary: bool = False
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint if not provided
    if config is None and 'config' in checkpoint:
        config = checkpoint['config']

    # Check model type from config
    if config and isinstance(config, dict):
        model_config = config.get('model', {})

        # Check for new architecture types first
        if model_config.get('use_hierarchical', False) or any(model_config.get(attr, False) for attr in ['use_hierarchical_unet', 'use_hierarchical_unet_v2', 'use_hierarchical_unet_v3', 'use_hierarchical_unet_v4', 'use_rgb_hierarchical']):
            model_type = 'hierarchical'
        elif model_config.get('use_class_specific_decoder', False):
            model_type = 'class_specific'
        elif model_config.get('variable_roi_sizes'):
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
        from .advanced.variable_roi_model import create_variable_roi_model, create_rgb_enhanced_variable_roi_model

        model_config = config.get('model', {}) if config else {}
        multiscale_config = config.get('multiscale', {}) if config else {}

        # Check if RGB enhancement is enabled
        use_rgb_enhancement = model_config.get('use_rgb_enhancement', False)

        if use_rgb_enhancement:
            model = create_rgb_enhanced_variable_roi_model(
                onnx_model_path=model_config.get('onnx_model', 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'),
                target_layers=multiscale_config.get('target_layers', ['layer_3', 'layer_22', 'layer_34']),
                roi_sizes=model_config.get('variable_roi_sizes', {'layer_3': 56, 'layer_22': 28, 'layer_34': 28}),
                num_classes=model_config.get('num_classes', 3),
                mask_size=model_config.get('mask_size', 56),
                execution_provider=model_config.get('execution_provider', 'cpu'),
                rgb_enhanced_layers=model_config.get('rgb_enhanced_layers', ['layer_34']),
                use_rgb_enhancement=True
            )
        else:
            model = create_variable_roi_model(
                onnx_model_path=model_config.get('onnx_model', 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'),
                target_layers=multiscale_config.get('target_layers', ['layer_3', 'layer_22', 'layer_34']),
                roi_sizes=model_config.get('variable_roi_sizes', {'layer_3': 56, 'layer_22': 28, 'layer_34': 28}),
                num_classes=model_config.get('num_classes', 3),
                mask_size=model_config.get('mask_size', 56),
                execution_provider=model_config.get('execution_provider', 'cpu')
            )

    elif model_type == 'hierarchical' or model_type == 'class_specific':
        # Use the same model building logic as in train_advanced.py
        from train_advanced import build_model

        # Convert config dict to ExperimentConfig object if needed
        from .experiments.config_manager import ExperimentConfig
        if isinstance(config, ExperimentConfig):
            exp_config = config
        else:
            exp_config = ExperimentConfig.from_dict(config)

        # Build model
        model, _ = build_model(exp_config, device)

    else:
        print(f"Unknown model type: {model_type}")
        return False

    # Load model weights with strict=False for architecture changes
    # If model is wrapped in DistillationModelWrapper, load into student model
    from src.human_edge_detection.advanced.knowledge_distillation import DistillationModelWrapper
    if isinstance(model, DistillationModelWrapper):
        # Load weights into student model only
        model_to_load = model.get_student()
    else:
        model_to_load = model
    
    try:
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except RuntimeError as e:
        print(f"Warning: Failed to load model weights with strict=True, trying strict=False...")
        print(f"Error: {e}")
        model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Extract mask_size from config
    mask_size = None
    if config:
        if isinstance(config, dict):
            model_config = config.get('model', {})
            if isinstance(model_config, dict):
                mask_size = model_config.get('mask_size', 56)
            elif hasattr(model_config, 'mask_size'):
                mask_size = model_config.mask_size
        elif hasattr(config, 'model') and hasattr(config.model, 'mask_size'):
            mask_size = config.model.mask_size
    
    # Create exporter
    exporter = AdvancedONNXExporter(model, model_type=model_type, device=device, include_auxiliary=include_auxiliary, mask_size=mask_size)

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
            'best_miou': checkpoint.get('best_miou', -1)
        }

        # Convert config to dict if it's not already
        if config:
            if isinstance(config, dict):
                # Check if nested objects need conversion
                config_dict = {}
                for k, v in config.items():
                    if hasattr(v, 'to_dict'):
                        config_dict[k] = v.to_dict()
                    elif hasattr(v, '__dict__'):
                        config_dict[k] = vars(v)
                    else:
                        config_dict[k] = v
                metadata['config'] = config_dict
            else:
                metadata['config'] = config if isinstance(config, dict) else None

        if model_type == 'baseline':
            metadata['input_format'] = {
                'features': '[B, 1024, 80, 80] - YOLO intermediate features',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            }
        else:
            # Check if this is an RGB enhanced model
            is_rgb_enhanced = False
            if config and isinstance(config, dict):
                model_config = config.get('model', {})
                if isinstance(model_config, dict):
                    is_rgb_enhanced = model_config.get('use_rgb_enhancement', False)
                elif hasattr(model_config, 'use_rgb_enhancement'):
                    is_rgb_enhanced = model_config.use_rgb_enhancement

            if is_rgb_enhanced:
                # RGB enhanced model includes RGB encoder
                metadata['input_format'] = {
                    'features_layer_3': '[B, 256, 160, 160] - High-resolution YOLO features',
                    'features_layer_22': '[B, 512, 80, 80] - Mid-level YOLO features',
                    'features_layer_34': '[B, 1024, 80, 80] - Deep YOLO features',
                    'rgb_images': '[B, 3, 640, 640] - RGB input images',
                    'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
                }
                metadata['note'] = 'This model includes RGB encoder and requires both pre-extracted YOLO features and RGB images.'
                metadata['rgb_enhanced'] = True
                metadata['rgb_enhanced_layers'] = ['layer_34']
                if config and isinstance(config, dict):
                    model_config = config.get('model', {})
                    if isinstance(model_config, dict):
                        metadata['rgb_enhanced_layers'] = model_config.get('rgb_enhanced_layers', ['layer_34'])
                    elif hasattr(model_config, 'rgb_enhanced_layers'):
                        metadata['rgb_enhanced_layers'] = model_config.rgb_enhanced_layers
            else:
                # Standard multiscale model
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