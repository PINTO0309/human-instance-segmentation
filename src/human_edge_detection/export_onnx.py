"""Export trained model to ONNX format."""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

from .model import create_model, ROISegmentationHead

try:
    from onnxsim import simplify as onnxsim_simplify
    ONNXSIM_AVAILABLE = True
except ImportError:
    try:
        import onnx_simplifier as onnxsim
        ONNXSIM_AVAILABLE = True
    except ImportError:
        ONNXSIM_AVAILABLE = False


class ONNXExporter:
    """Export segmentation model to ONNX format."""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """Initialize exporter.

        Args:
            model: Trained segmentation model
            device: Device to use for export
        """
        self.model = model.to(device)
        self.device = device

    def export_segmentation_head(
        self,
        output_path: str,
        batch_size: int = 1,
        opset_version: int = 16,
        verify: bool = True
    ) -> bool:
        """Export only the segmentation head to ONNX.

        This exports just the decoder part, assuming features will be
        provided by the YOLO model at inference time.

        Args:
            output_path: Path to save ONNX model
            batch_size: Batch size for export
            opset_version: ONNX opset version
            verify: Whether to verify the exported model

        Returns:
            Success status
        """
        self.model.eval()

        # Get segmentation head
        seg_head = self.model.segmentation_head
        seg_head.eval()

        # Create dummy inputs
        # Features: (B, 1024, 80, 80)
        dummy_features = torch.randn(batch_size, 1024, 80, 80).to(self.device)

        # ROIs: (N, 5) format [batch_idx, x1, y1, x2, y2]
        # For export, we'll use dynamic ROI count
        num_rois = 2  # Example with 2 ROIs
        dummy_rois = torch.zeros(num_rois, 5).to(self.device)
        dummy_rois[:, 0] = torch.arange(num_rois) % batch_size  # Batch indices
        dummy_rois[:, 1:] = torch.tensor([
            [100, 100, 300, 300],  # First ROI
            [200, 200, 400, 400]   # Second ROI
        ]).float()

        # Export to ONNX
        print(f"Exporting segmentation head to {output_path}")

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

            # Simplify with onnxsim if available
            if ONNXSIM_AVAILABLE:
                print("Simplifying ONNX model with onnxsim...")
                try:
                    model_onnx = onnx.load(output_path)
                    # Try to use the imported simplify function
                    if 'onnxsim_simplify' in globals():
                        model_simp, check = onnxsim_simplify(model_onnx, check_n=3)
                    else:
                        model_simp, check = onnxsim.simplify(model_onnx, check_n=3)
                    if check:
                        onnx.save(model_simp, output_path)
                        print("Model simplified successfully!")
                    else:
                        print("Model simplification check failed, keeping original")
                except Exception as e:
                    print(f"Warning: Model simplification failed: {e}")

            if verify:
                return self._verify_onnx_model(output_path, dummy_features, dummy_rois)

            return True

        except Exception as e:
            print(f"Export failed: {e}")
            return False

    def _verify_onnx_model(
        self,
        onnx_path: str,
        dummy_features: torch.Tensor,
        dummy_rois: torch.Tensor
    ) -> bool:
        """Verify exported ONNX model."""
        print("Verifying ONNX model...")

        try:
            # Check ONNX model
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print("ONNX model structure is valid")

            # Test inference with GPU providers if available
            providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            ort_session = ort.InferenceSession(onnx_path, providers=providers)

            # Prepare inputs
            ort_inputs = {
                'features': dummy_features.cpu().numpy(),
                'rois': dummy_rois.cpu().numpy()
            }

            # Run inference
            ort_outputs = ort_session.run(None, ort_inputs)

            # Compare with PyTorch output
            self.model.eval()
            with torch.no_grad():
                torch_output = self.model.segmentation_head(dummy_features, dummy_rois)

            # Check output shape
            print(f"ONNX output shape: {ort_outputs[0].shape}")
            print(f"PyTorch output shape: {torch_output.shape}")

            # Check numerical similarity
            torch_out_np = torch_output.cpu().numpy()
            onnx_out_np = ort_outputs[0]

            max_diff = np.max(np.abs(torch_out_np - onnx_out_np))
            mean_diff = np.mean(np.abs(torch_out_np - onnx_out_np))

            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")

            # Check if differences are acceptable
            if max_diff < 1e-3:
                print("Verification passed! Outputs are numerically similar.")
                return True
            else:
                print("Warning: Large numerical differences detected.")
                return False

        except Exception as e:
            print(f"Verification failed: {e}")
            return False

    def export_combined_model(
        self,
        yolo_onnx_path: str,
        output_path: str,
        opset_version: int = 13
    ) -> bool:
        """Export combined model (YOLO features + segmentation head).

        This creates a single ONNX model that includes both feature extraction
        and segmentation, but this is more complex and may not be necessary.

        Args:
            yolo_onnx_path: Path to YOLO ONNX model
            output_path: Path to save combined ONNX model
            opset_version: ONNX opset version

        Returns:
            Success status
        """
        # This is a more advanced feature that would require
        # combining the YOLO model and segmentation head into one graph
        # For now, we recommend using separate models
        print("Combined model export is not yet implemented.")
        print("Please use the segmentation head export and run YOLO separately.")
        return False


def export_checkpoint_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config: Optional[Dict] = None,
    device: str = 'cpu',
    verify: bool = True,
    opset_version: int = 16
) -> bool:
    """Export a checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Path to save ONNX model
        config: Model configuration (if None, uses defaults)
        device: Device to use for export
        verify: Whether to verify the exported model

    Returns:
        Success status
    """
    # Default config
    if config is None:
        config = {
            'num_classes': 3,
            'in_channels': 1024,
            'mid_channels': 256,
            'mask_size': 56,
            'roi_size': 28  # Default to enhanced model
        }

    # Create model
    model = create_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        mid_channels=config['mid_channels'],
        mask_size=config['mask_size']
    )

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create exporter
    exporter = ONNXExporter(model, device=device)

    # Export to ONNX
    success = exporter.export_segmentation_head(
        output_path=output_path,
        verify=verify,
        opset_version=opset_version
    )

    if success:
        print(f"Successfully exported model to {output_path}")

        # Save metadata
        metadata_path = Path(output_path).with_suffix('.json')
        import json
        metadata = {
            'checkpoint_path': str(checkpoint_path),
            'epoch': checkpoint.get('epoch', -1),
            'miou': checkpoint.get('miou', -1),
            'config': config,
            'input_format': {
                'features': '[B, 1024, 80, 80] - YOLO intermediate features',
                'rois': '[N, 5] - ROIs in format [batch_idx, x1, y1, x2, y2]'
            },
            'output_format': {
                'masks': '[N, 3, 56, 56] - Segmentation logits for each ROI'
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {metadata_path}")

    return success


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save ONNX model')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for export')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification')

    args = parser.parse_args()

    # Export
    success = export_checkpoint_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        device=args.device,
        verify=not args.no_verify
    )

    if not success:
        exit(1)