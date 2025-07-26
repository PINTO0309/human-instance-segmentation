"""Compare PyTorch model with ONNX export to verify GroupNorm."""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys

def count_norm_layers_pytorch(model):
    """Count normalization layers in PyTorch model."""
    counts = {
        'GroupNorm': 0,
        'LayerNorm': 0,
        'BatchNorm2d': 0,
        'InstanceNorm2d': 0,
        'LayerNorm2d': 0
    }
    
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            counts['GroupNorm'] += 1
            print(f"  GroupNorm found: {name} (groups={module.num_groups}, channels={module.num_channels})")
        elif isinstance(module, nn.LayerNorm):
            counts['LayerNorm'] += 1
        elif isinstance(module, nn.BatchNorm2d):
            counts['BatchNorm2d'] += 1
        elif isinstance(module, nn.InstanceNorm2d):
            counts['InstanceNorm2d'] += 1
        elif hasattr(module, '__class__') and 'LayerNorm2d' in module.__class__.__name__:
            counts['LayerNorm2d'] += 1
            print(f"  LayerNorm2d found: {name}")
    
    return counts

def test_model_outputs(pytorch_model, onnx_path, device='cpu'):
    """Compare outputs between PyTorch and ONNX models."""
    print("\nTesting model outputs...")
    
    # Create dummy inputs
    batch_size = 2
    roi_size = 64
    num_rois = 4
    
    # RGB images
    dummy_images = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # ROIs
    dummy_rois = []
    for b in range(batch_size):
        for _ in range(num_rois // batch_size):
            x1, y1 = np.random.randint(0, 500, size=2)
            x2, y2 = x1 + roi_size, y1 + roi_size
            dummy_rois.append([b, x1, y1, x2, y2])
    dummy_rois = torch.tensor(dummy_rois, dtype=torch.float32).to(device)
    
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_images, dummy_rois)
        if isinstance(pytorch_output, tuple):
            pytorch_output = pytorch_output[0]
    
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # ONNX inference
    if Path(onnx_path).exists():
        providers = ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Prepare ONNX inputs
        ort_inputs = {
            'images': dummy_images.cpu().numpy(),
            'rois': dummy_rois.cpu().numpy()
        }
        
        # Run inference
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_output = ort_outputs[0]
        
        print(f"ONNX output shape: {onnx_output.shape}")
        
        # Compare outputs
        pytorch_output_np = pytorch_output.cpu().numpy()
        max_diff = np.abs(pytorch_output_np - onnx_output).max()
        mean_diff = np.abs(pytorch_output_np - onnx_output).mean()
        
        print(f"\nOutput comparison:")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")
        
        if max_diff < 1e-3:
            print("  ✓ Outputs match closely - model export is correct")
        else:
            print("  ⚠ Large difference detected - check export")

def check_model_and_onnx(checkpoint_path, onnx_path=None):
    """Load PyTorch model and compare with ONNX export."""
    print(f"\nChecking checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config
    config = checkpoint.get('config', {})
    if isinstance(config, dict):
        model_config = config.get('model', {})
        norm_type = model_config.get('normalization_type', 'not specified')
        norm_groups = model_config.get('normalization_groups', 'not specified')
        print(f"Config normalization_type: {norm_type}")
        print(f"Config normalization_groups: {norm_groups}")
    
    # Build model using train_advanced logic
    from train_advanced import build_model
    from src.human_edge_detection.experiments.config_manager import ExperimentConfig
    
    # Convert config to ExperimentConfig
    exp_config = ExperimentConfig.from_dict(config)
    
    # Build model
    model, _ = build_model(exp_config, 'cpu')
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    
    # Count normalization layers
    print("\nPyTorch model normalization layers:")
    counts = count_norm_layers_pytorch(model)
    for norm_type, count in counts.items():
        if count > 0:
            print(f"  {norm_type}: {count}")
    
    # Check ONNX if path provided
    if onnx_path and Path(onnx_path).exists():
        print(f"\nChecking ONNX export: {onnx_path}")
        test_model_outputs(model, onnx_path)
    
    return model, counts

def main():
    exp_name = "rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_groupnorm"
    
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    
    exp_dir = Path(f"experiments/{exp_name}")
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    checkpoint_dir = exp_dir / "checkpoints"
    
    # Check untrained model
    untrained_pth = checkpoint_dir / "untrained_model.pth"
    untrained_onnx = checkpoint_dir / "untrained_model.onnx"
    if untrained_pth.exists():
        print("\n" + "="*60)
        print("UNTRAINED MODEL")
        print("="*60)
        check_model_and_onnx(untrained_pth, untrained_onnx)
    
    # Check best model
    best_pth = checkpoint_dir / "best_model.pth"
    best_onnx = checkpoint_dir / "best_model.onnx"
    if best_pth.exists():
        print("\n" + "="*60)
        print("BEST MODEL")
        print("="*60)
        check_model_and_onnx(best_pth, best_onnx)

if __name__ == "__main__":
    main()