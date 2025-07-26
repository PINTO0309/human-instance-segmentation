"""Check normalization layers in ONNX models."""

import onnx
import numpy as np
from pathlib import Path
import sys

def check_onnx_normalization(onnx_path):
    """Check normalization layers in an ONNX model."""
    print(f"\nChecking ONNX model: {onnx_path}")
    
    if not Path(onnx_path).exists():
        print(f"ONNX model not found: {onnx_path}")
        return
    
    # Load ONNX model
    model = onnx.load(onnx_path)
    
    # Count different types of normalization nodes
    layernorm_count = 0
    groupnorm_count = 0
    batchnorm_count = 0
    instancenorm_count = 0
    
    # Also track specific operations that might indicate normalization
    reduce_mean_count = 0
    reshape_count = 0
    
    # Check all nodes
    for node in model.graph.node:
        op_type = node.op_type
        
        if op_type == "LayerNormalization":
            layernorm_count += 1
        elif op_type == "GroupNormalization":
            groupnorm_count += 1
        elif op_type == "BatchNormalization":
            batchnorm_count += 1
        elif op_type == "InstanceNormalization":
            instancenorm_count += 1
        elif op_type == "ReduceMean":
            reduce_mean_count += 1
        elif op_type == "Reshape":
            reshape_count += 1
    
    print(f"\nNormalization nodes found:")
    print(f"  LayerNormalization: {layernorm_count}")
    print(f"  GroupNormalization: {groupnorm_count}")
    print(f"  BatchNormalization: {batchnorm_count}")
    print(f"  InstanceNormalization: {instancenorm_count}")
    print(f"\nOther relevant nodes:")
    print(f"  ReduceMean: {reduce_mean_count}")
    print(f"  Reshape: {reshape_count}")
    
    # Check for patterns that might indicate custom normalization
    # GroupNorm in PyTorch might be exported as a series of operations
    print("\nChecking for GroupNorm patterns...")
    
    # Look for specific patterns in node names
    groupnorm_pattern_count = 0
    layernorm_pattern_count = 0
    
    for node in model.graph.node:
        # Check node names and inputs/outputs
        node_str = str(node).lower()
        if 'groupnorm' in node_str:
            groupnorm_pattern_count += 1
        elif 'layernorm' in node_str or 'layer_norm' in node_str:
            layernorm_pattern_count += 1
    
    if groupnorm_pattern_count > 0:
        print(f"  Found {groupnorm_pattern_count} nodes with 'groupnorm' in name")
    if layernorm_pattern_count > 0:
        print(f"  Found {layernorm_pattern_count} nodes with 'layernorm' in name")
    
    # Check initializers for weight/bias patterns
    print("\nChecking weight/bias patterns...")
    weight_shapes = {}
    for init in model.graph.initializer:
        if 'weight' in init.name.lower() or 'bias' in init.name.lower():
            shape = list(init.dims)
            if 'norm' in init.name.lower():
                norm_type = "unknown"
                if 'group' in init.name.lower():
                    norm_type = "groupnorm"
                elif 'layer' in init.name.lower():
                    norm_type = "layernorm"
                elif 'batch' in init.name.lower():
                    norm_type = "batchnorm"
                
                if norm_type not in weight_shapes:
                    weight_shapes[norm_type] = []
                weight_shapes[norm_type].append((init.name, shape))
    
    for norm_type, shapes in weight_shapes.items():
        print(f"\n{norm_type} weights/biases: {len(shapes)}")
        if len(shapes) > 0 and len(shapes) <= 5:
            for name, shape in shapes[:5]:
                print(f"  {name}: {shape}")

def check_experiment_onnx(exp_name):
    """Check all ONNX models in an experiment directory."""
    exp_dir = Path(f"experiments/{exp_name}")
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Check all .onnx files
    onnx_files = list(checkpoint_dir.glob("*.onnx"))
    if not onnx_files:
        print(f"No ONNX files found in {checkpoint_dir}")
        return
        
    for onnx_path in onnx_files:
        check_onnx_normalization(onnx_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        exp_name = "rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_groupnorm"
    
    print(f"Checking ONNX models in experiment: {exp_name}")
    check_experiment_onnx(exp_name)
    
    # Also check specific ONNX file if provided
    if len(sys.argv) > 2:
        onnx_file = sys.argv[2]
        check_onnx_normalization(onnx_file)