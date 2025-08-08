#!/usr/bin/env python3
"""
Script to analyze YOLOv9 ONNX models and understand available intermediate features.
"""

import onnx
import onnxruntime as ort
import numpy as np
import sys
import os
from pathlib import Path

def analyze_onnx_model(model_path):
    """Analyze an ONNX model to understand its structure and outputs."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_path}")
    print(f"{'='*60}")
    
    # Load the ONNX model
    try:
        model = onnx.load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  IR version: {model.ir_version}")
        print(f"  Producer: {model.producer_name} {model.producer_version}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Get model inputs
    print(f"\nInputs ({len(model.graph.input)}):")
    for i, input_tensor in enumerate(model.graph.input):
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"  {i}: {input_tensor.name} - {input_tensor.type.tensor_type.elem_type} {shape}")
    
    # Get model outputs
    print(f"\nOutputs ({len(model.graph.output)}):")
    for i, output_tensor in enumerate(model.graph.output):
        shape = [dim.dim_value if dim.dim_value > 0 else dim.dim_param 
                for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"  {i}: {output_tensor.name} - {output_tensor.type.tensor_type.elem_type} {shape}")
    
    # Get all intermediate nodes/outputs
    print(f"\nIntermediate nodes ({len(model.graph.node)}):")
    intermediate_outputs = set()
    for node in model.graph.node:
        for output in node.output:
            intermediate_outputs.add(output)
    
    print(f"  Total intermediate outputs: {len(intermediate_outputs)}")
    
    # Find nodes that might be feature extraction points
    feature_nodes = []
    for node in model.graph.node:
        if any(keyword in node.name.lower() for keyword in ['conv', 'feature', 'backbone', 'neck', 'fpn']):
            if node.op_type in ['Conv', 'Add', 'Concat', 'Mul']:
                feature_nodes.append((node.name, node.op_type, node.output))
    
    print(f"\nPotential feature extraction nodes ({len(feature_nodes)}):")
    for name, op_type, outputs in feature_nodes[:10]:  # Show first 10
        print(f"  {name} ({op_type}) -> {outputs}")
    if len(feature_nodes) > 10:
        print(f"  ... and {len(feature_nodes) - 10} more")
    
    # Try to run inference to understand actual output shapes
    print(f"\nTesting inference with ONNXRuntime:")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input details
        input_details = session.get_inputs()
        output_details = session.get_outputs()
        
        print(f"  Input details:")
        for inp in input_details:
            print(f"    {inp.name}: {inp.type} {inp.shape}")
        
        print(f"  Output details:")
        for out in output_details:
            print(f"    {out.name}: {out.type} {out.shape}")
        
        # Create dummy input with fixed dimensions
        input_shape = [1, 3, 640, 640]  # Fixed shape for testing
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        print(f"  Created dummy input: {dummy_input.shape}")
        
        # Run inference
        input_name = input_details[0].name
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"  Inference successful! Output shapes:")
        for i, output in enumerate(outputs):
            print(f"    Output {i}: {output.shape} (dtype: {output.dtype})")
            
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
    
    print(f"\n{'='*60}\n")

def main():
    """Main function to analyze all ONNX models in ext_extractor."""
    ext_dir = Path("ext_extractor")
    
    if not ext_dir.exists():
        print("ext_extractor directory not found!")
        return
    
    # Find all ONNX models
    onnx_models = list(ext_dir.glob("*.onnx"))
    
    if not onnx_models:
        print("No ONNX models found in ext_extractor/")
        return
    
    print(f"Found {len(onnx_models)} ONNX models:")
    for model in onnx_models:
        print(f"  - {model.name}")
    
    # Analyze each model
    for model_path in onnx_models:
        analyze_onnx_model(str(model_path))
    
    # Summary
    print("\nSUMMARY:")
    print("=" * 60)
    print("Key findings:")
    print("- Model inputs: Typically Nx3x640x640 (batch, channels, height, width)")
    print("- Feature extraction capabilities depend on model variant (n, t, s, e, c)")
    print("- Look for intermediate outputs that could serve as feature maps")
    print("- Consider modifying models to expose intermediate features if needed")

if __name__ == "__main__":
    main()