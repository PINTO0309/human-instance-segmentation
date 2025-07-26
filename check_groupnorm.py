"""Check if a model is using GroupNorm."""

import torch
import torch.nn as nn
from pathlib import Path
import sys

def check_normalization_layers(model, prefix=""):
    """Recursively check normalization layers in a model."""
    norm_layers = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.GroupNorm):
            norm_layers.append(f"{prefix}{name}: GroupNorm(groups={module.num_groups}, channels={module.num_channels})")
        elif isinstance(module, nn.BatchNorm2d):
            norm_layers.append(f"{prefix}{name}: BatchNorm2d(features={module.num_features})")
        elif isinstance(module, nn.LayerNorm):
            norm_layers.append(f"{prefix}{name}: LayerNorm(shape={module.normalized_shape})")
        elif hasattr(module, '__class__') and 'LayerNorm2d' in module.__class__.__name__:
            norm_layers.append(f"{prefix}{name}: LayerNorm2d")
            
    return norm_layers

def check_checkpoint(checkpoint_path):
    """Check normalization layers in a checkpoint."""
    print(f"\nChecking checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check if config is in checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            model_config = config.get('model', {})
            norm_type = model_config.get('normalization_type', 'not specified')
            norm_groups = model_config.get('normalization_groups', 'not specified')
            print(f"Config normalization_type: {norm_type}")
            print(f"Config normalization_groups: {norm_groups}")
        else:
            print("Config is not a dict")
    else:
        print("No config in checkpoint")
    
    # Check model state dict for GroupNorm layers
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', {}))
    
    groupnorm_layers = []
    layernorm_layers = []
    batchnorm_layers = []
    
    for key in state_dict.keys():
        if 'num_groups' in key:
            groupnorm_layers.append(key)
        elif 'norm' in key.lower() and ('weight' in key or 'bias' in key):
            if 'batch' in key.lower():
                batchnorm_layers.append(key)
            else:
                layernorm_layers.append(key)
    
    print(f"\nGroupNorm layers found: {len(groupnorm_layers)}")
    if groupnorm_layers:
        print("Examples:", groupnorm_layers[:5])
    
    print(f"\nLayerNorm layers found: {len(layernorm_layers)}")
    if layernorm_layers:
        print("Examples:", layernorm_layers[:5])
        
    print(f"\nBatchNorm layers found: {len(batchnorm_layers)}")
    if batchnorm_layers:
        print("Examples:", batchnorm_layers[:5])

def check_experiment_dir(exp_name):
    """Check all checkpoints in an experiment directory."""
    exp_dir = Path(f"experiments/{exp_name}")
    if not exp_dir.exists():
        print(f"Experiment directory not found: {exp_dir}")
        return
    
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Check all .pth files
    for checkpoint_path in checkpoint_dir.glob("*.pth"):
        check_checkpoint(checkpoint_path)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        exp_name = sys.argv[1]
    else:
        exp_name = "rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_groupnorm"
    
    print(f"Checking experiment: {exp_name}")
    check_experiment_dir(exp_name)
    
    # Also check the regular non-groupnorm version for comparison
    comparison_exp = "rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_boundaryrefinement"
    print(f"\n\nFor comparison, checking: {comparison_exp}")
    check_experiment_dir(comparison_exp)