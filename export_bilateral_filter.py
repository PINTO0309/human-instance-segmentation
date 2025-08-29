"""Export bilateral filter models to ONNX format."""

import argparse
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path

from src.human_edge_detection.bilateral_filter import (
    BilateralFilter,
    FastBilateralFilter,
    EdgePreservingFilter,
    BinaryMaskBilateralFilter,
    MorphologicalBilateralFilter
)


def export_bilateral_filter(
    model_type: str = "fast",
    output_path: str = "bilateral_filter.onnx",
    input_shape: tuple = (1, 3, 224, 224),
    kernel_size: int = 5,
    sigma_spatial: float = 1.0,
    sigma_range: float = 0.1,
    verify: bool = True
):
    """Export bilateral filter to ONNX format.
    
    Args:
        model_type: Type of filter ("bilateral", "fast", or "edge_preserving")
        output_path: Path to save ONNX model
        input_shape: Input tensor shape (B, C, H, W)
        kernel_size: Filter kernel size
        sigma_spatial: Spatial domain standard deviation
        sigma_range: Range domain standard deviation
        verify: Whether to verify the exported model
    """
    
    # Create model based on type
    if model_type == "bilateral":
        print("Note: Standard bilateral filter is computationally expensive for ONNX.")
        print("Consider using 'fast' or 'edge_preserving' for better performance.")
        model = BilateralFilter(
            kernel_size=kernel_size,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range
        )
    elif model_type == "fast":
        model = FastBilateralFilter(
            kernel_size=kernel_size,
            sigma_spatial=sigma_spatial,
            sigma_range=sigma_range
        )
    elif model_type == "edge_preserving":
        model = EdgePreservingFilter(
            radius=kernel_size // 2,
            eps=sigma_range
        )
    elif model_type == "binary_mask":
        model = BinaryMaskBilateralFilter(
            kernel_size=kernel_size,
            sigma_spatial=sigma_spatial,
            threshold=0.5
        )
    elif model_type == "morphological":
        model = MorphologicalBilateralFilter(
            kernel_size=kernel_size,
            sigma=sigma_spatial,
            morph_size=3
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Export to ONNX
    print(f"Exporting {model_type} bilateral filter to ONNX...")
    
    # Define dynamic axes for flexible input sizes
    dynamic_axes = {
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch', 2: 'height', 3: 'width'}
    }
    
    # Export with dynamic shapes
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify the model if requested
    if verify:
        print("\nVerifying exported ONNX model...")
        
        # Check ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")
        
        # Test inference
        ort_session = ort.InferenceSession(output_path)
        
        # Test with original input size
        test_input = dummy_input.numpy()
        ort_output = ort_session.run(None, {'input': test_input})[0]
        
        # Compare with PyTorch output
        with torch.no_grad():
            pytorch_output = model(dummy_input).numpy()
        
        # Check output shape
        assert ort_output.shape == pytorch_output.shape, \
            f"Shape mismatch: ONNX {ort_output.shape} vs PyTorch {pytorch_output.shape}"
        
        # Check numerical accuracy
        max_diff = np.abs(ort_output - pytorch_output).max()
        mean_diff = np.abs(ort_output - pytorch_output).mean()
        
        print(f"✓ Output shape matches: {ort_output.shape}")
        print(f"✓ Max difference: {max_diff:.6f}")
        print(f"✓ Mean difference: {mean_diff:.6f}")
        
        # Test with different input size (preserve channel count)
        test_shape = (2, input_shape[1], 128, 128)  # Keep same number of channels
        test_input_2 = np.random.randn(*test_shape).astype(np.float32)
        ort_output_2 = ort_session.run(None, {'input': test_input_2})[0]
        
        print(f"✓ Dynamic shape test passed: input {test_shape} -> output {ort_output_2.shape}")
        
        print("\nONNX model verification successful!")
        
        # Print model info
        print(f"\nModel information:")
        print(f"- Type: {model_type}")
        print(f"- Input shape: {input_shape}")
        print(f"- Kernel size: {kernel_size}")
        print(f"- Sigma spatial: {sigma_spatial}")
        print(f"- Sigma range: {sigma_range}")
        print(f"- File size: {Path(output_path).stat().st_size / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description="Export bilateral filter to ONNX")
    parser.add_argument(
        "--model_type",
        type=str,
        default="fast",
        choices=["bilateral", "fast", "edge_preserving", "binary_mask", "morphological"],
        help="Type of bilateral filter to export"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bilateral_filter.onnx",
        help="Output ONNX file path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for export"
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=3,
        help="Number of input channels"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Input height"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Input width"
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="Filter kernel size (must be odd)"
    )
    parser.add_argument(
        "--sigma_spatial",
        type=float,
        default=1.0,
        help="Spatial domain standard deviation"
    )
    parser.add_argument(
        "--sigma_range",
        type=float,
        default=0.1,
        help="Range domain standard deviation"
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip model verification"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export model
    export_bilateral_filter(
        model_type=args.model_type,
        output_path=str(output_path),
        input_shape=(args.batch_size, args.channels, args.height, args.width),
        kernel_size=args.kernel_size,
        sigma_spatial=args.sigma_spatial,
        sigma_range=args.sigma_range,
        verify=not args.no_verify
    )
    
    # Export all types if requested
    if args.model_type == "all":
        for model_type in ["fast", "edge_preserving", "binary_mask", "morphological"]:
            output_name = output_path.stem + f"_{model_type}" + output_path.suffix
            output_full_path = output_path.parent / output_name
            
            print(f"\n{'='*60}")
            print(f"Exporting {model_type} model...")
            print(f"{'='*60}\n")
            
            export_bilateral_filter(
                model_type=model_type,
                output_path=str(output_full_path),
                input_shape=(args.batch_size, args.channels, args.height, args.width),
                kernel_size=args.kernel_size,
                sigma_spatial=args.sigma_spatial,
                sigma_range=args.sigma_range,
                verify=not args.no_verify
            )


if __name__ == "__main__":
    main()