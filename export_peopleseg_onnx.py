#!/usr/bin/env python3
"""Export distilled models (B0/B1/B7) to ONNX and optimize with onnxsim."""

import torch
import onnx
import onnxsim
import os
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export distilled model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--height', type=int, default=640,
                        help='Input height (default: 640)')
    parser.add_argument('--width', type=int, default=640,
                        help='Input width (default: 640)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic batch size, height, and width')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip onnxsim optimization')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: same as checkpoint)')
    parser.add_argument('--opset', type=int, default=16,
                        help='ONNX opset version (default: 16)')
    return parser.parse_args()

def detect_model_type(checkpoint_path):
    """Detect model type from checkpoint path."""
    path_str = str(checkpoint_path)
    
    # Extract model type from path or filename
    if 'best_model_b0' in path_str or '_b0_' in path_str:
        return 'b0', 'timm-efficientnet-b0'
    elif 'best_model_b1' in path_str or '_b1_' in path_str:
        return 'b1', 'timm-efficientnet-b1'
    elif 'best_model_b7' in path_str or '_b7_' in path_str:
        return 'b7', 'timm-efficientnet-b7'
    else:
        # Try to detect from path pattern
        if 'b0' in path_str.lower():
            return 'b0', 'timm-efficientnet-b0'
        elif 'b1' in path_str.lower():
            return 'b1', 'timm-efficientnet-b1'
        elif 'b7' in path_str.lower():
            return 'b7', 'timm-efficientnet-b7'
        else:
            print("Warning: Could not detect model type, defaulting to B0")
            return 'b0', 'timm-efficientnet-b0'

def export_distilled_model_to_onnx(args):
    """Export the distilled model to ONNX format."""
    
    # Paths
    checkpoint_path = args.checkpoint
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return False
    
    # Detect model type
    model_type, encoder_name = detect_model_type(checkpoint_path)
    print(f"Detected model type: {model_type.upper()} (encoder: {encoder_name})")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(checkpoint_path).parent
    
    # Use the same filename as checkpoint, just change extension
    checkpoint_filename = Path(checkpoint_path).stem  # Get filename without extension
    onnx_path = output_dir / f"{checkpoint_filename}.onnx"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint (weights_only=False needed for custom classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Import the model architecture
    import segmentation_models_pytorch as smp
    
    # Create model
    print(f"Creating distilled {model_type.upper()} model...")
    print(f"  Input size: {args.batch_size}x3x{args.height}x{args.width}")
    
    # Load state dict first to detect decoder channels
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove prefixes for inspection
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        elif k.startswith('unet.'):
            clean_state_dict[k[5:]] = v  
        else:
            clean_state_dict[k] = v
    
    # Auto-detect decoder channels from state dict
    decoder_channels = None
    for key in clean_state_dict.keys():
        if 'decoder.blocks.0.conv1.0.weight' in key:
            # Try to find all decoder block channels
            channels = []
            for i in range(5):  # Usually 5 decoder blocks
                key_name = f'decoder.blocks.{i}.conv1.0.weight'
                if key_name in clean_state_dict:
                    channels.append(clean_state_dict[key_name].shape[0])
            
            if len(channels) == 5:
                decoder_channels = tuple(channels)
                print(f"Auto-detected decoder channels: {decoder_channels}")
                break
    
    # Fallback to default if not detected
    if decoder_channels is None:
        if model_type == 'b7':
            # B7 typically uses these channels
            decoder_channels = (256, 128, 64, 32, 16)
        elif model_type == 'b1':
            decoder_channels = (256, 128, 64, 32, 16)
        else:  # b0
            decoder_channels = (256, 128, 64, 32, 16)
        print(f"Using default decoder channels: {decoder_channels}")
    
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,  # No pre-training, this is distilled
        in_channels=3,
        classes=1,
        decoder_channels=decoder_channels,
        decoder_use_batchnorm=True
    )
    
    # Try to load state dict
    try:
        model.load_state_dict(clean_state_dict, strict=True)
        print("Model weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"Warning: Strict loading failed, trying with strict=False")
        print(f"Error was: {str(e)[:200]}...")
        model.load_state_dict(clean_state_dict, strict=False)
        print("Model weights loaded with strict=False")
    
    # Set to evaluation mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(args.batch_size, 3, args.height, args.width)
    
    # Configure dynamic axes
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
        print("Dynamic batch size, height, and width enabled")
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    # Verify the exported model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(str(onnx_path))
    try:
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification successful")
    except onnx.checker.ValidationError as e:
        print(f"ONNX model validation failed: {e}")
        return False
    
    # Get original model size
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\nOriginal ONNX model size: {original_size:.2f} MB")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    if args.no_optimize:
        print("\nSkipping onnxsim optimization (--no-optimize flag)")
        print(f"\nModel exported to: {onnx_path}")
        return True
    
    # Optimize with onnxsim
    print("\nOptimizing with onnxsim...")
    try:
        # Configure parameters based on whether using dynamic dimensions
        if args.dynamic:
            # For dynamic dimensions, provide test input shapes
            simplify_params = {
                'test_input_shapes': {'input': [args.batch_size, 3, args.height, args.width]}
            }
        else:
            # For fixed dimensions, use overwrite_input_shapes (new API)
            simplify_params = {
                'overwrite_input_shapes': {'input': [args.batch_size, 3, args.height, args.width]}
            }
        
        model_opt, check = onnxsim.simplify(
            onnx_model,
            check_n=3,
            skip_fuse_bn=False,
            skip_shape_inference=False,
            **simplify_params
        )
        
        if check:
            print("Optimization successful")
            
            # Save optimized model, overwriting original
            onnx.save(model_opt, str(onnx_path))
            
            # Get optimized model size
            optimized_size = os.path.getsize(onnx_path) / (1024 * 1024)
            print(f"Optimized ONNX model size: {optimized_size:.2f} MB")
            reduction = original_size - optimized_size
            reduction_percent = (reduction / original_size) * 100 if original_size > 0 else 0
            if reduction > 0:
                print(f"Size reduction: {reduction:.2f} MB ({reduction_percent:.1f}%)")
            else:
                print(f"Size change: {-reduction:.2f} MB ({-reduction_percent:.1f}% increase)")
            
            print(f"\nOptimized model saved to: {onnx_path}")
            
            # Test inference with optimized model
            print("\nTesting inference...")
            import onnxruntime as ort
            
            # Create test input
            test_input = torch.randn(args.batch_size, 3, args.height, args.width).numpy()
            
            # Test optimized model
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})[0]
            
            # Print input/output shapes
            print(f"\nModel I/O shapes:")
            print(f"  Input shape:  {test_input.shape}")
            print(f"  Output shape: {output.shape}")
            
            return True
        else:
            print("Optimization failed verification")
            return False
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Validate arguments
    if args.batch_size < 1:
        print("Error: Batch size must be at least 1")
        return 1
    
    if args.height < 32 or args.width < 32:
        print("Error: Height and width must be at least 32")
        return 1
    
    # Run export
    success = export_distilled_model_to_onnx(args)
    
    if success:
        print("\n✅ Export completed successfully")
        return 0
    else:
        print("\n❌ Export failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())