#\!/usr/bin/env python3
"""Check if BatchNorm layers are in training mode."""

import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.human_edge_detection.experiments.config_manager import ConfigManager
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model

# Load config
config_name = 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48'
config_manager = ConfigManager()
config = config_manager.get_config(config_name)

# Create model
model = create_rgb_hierarchical_model(
    num_classes=config.model.num_classes,
    roi_size=config.model.roi_size,
    mask_size=config.model.mask_size,
    use_attention_module=config.model.use_attention_module,
    use_pretrained_unet=config.model.use_pretrained_unet,
    pretrained_weights_path=config.model.pretrained_weights_path,
    freeze_pretrained_weights=config.model.freeze_pretrained_weights,
    use_full_image_unet=config.model.use_full_image_unet,
)

# Check BatchNorm layers in pretrained_unet
print("Checking BatchNorm layers in pretrained_unet:")
for name, module in model.pretrained_unet.named_modules():
    if 'BatchNorm' in module.__class__.__name__:
        print(f"{name}: training={module.training}, track_running_stats={module.track_running_stats}")
        # Check if running_mean/running_var are being tracked
        if hasattr(module, 'running_mean') and module.running_mean is not None:
            print(f"  Running mean shape: {module.running_mean.shape}")
        break  # Just show first few

# Check if model has any trainable BatchNorm
print("\nChecking for trainable parameters in BatchNorm layers:")
for name, param in model.named_parameters():
    if 'pretrained_unet' in name and ('bn' in name or 'norm' in name) and param.requires_grad:
        print(f"TRAINABLE BN param: {name}")

