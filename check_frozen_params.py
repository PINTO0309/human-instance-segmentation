#\!/usr/bin/env python3
"""Check which parameters are frozen in the model."""

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

# Check frozen parameters
total_params = 0
frozen_params = 0
trainable_params = 0

print("\nAnalyzing model parameters:")
print("-" * 80)

for name, param in model.named_parameters():
    total_params += param.numel()
    if not param.requires_grad:
        frozen_params += param.numel()
        if 'pretrained_unet' in name[:20]:  # Show first few frozen params
            print(f"FROZEN: {name} - shape: {list(param.shape)}")
    else:
        trainable_params += param.numel()
        if trainable_params < 1000000:  # Show first few trainable params
            print(f"TRAINABLE: {name} - shape: {list(param.shape)}")

print("-" * 80)
print(f"\nTotal parameters: {total_params:,}")
print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# Check specific components
print("\n\nComponent breakdown:")
components = {
    'pretrained_unet': 0,
    'feature_processor': 0,
    'segmentation_head': 0,
    'other': 0
}

for name, param in model.named_parameters():
    if 'pretrained_unet' in name:
        components['pretrained_unet'] += param.numel()
    elif 'feature_processor' in name:
        components['feature_processor'] += param.numel()
    elif 'segmentation_head' in name:
        components['segmentation_head'] += param.numel()
    else:
        components['other'] += param.numel()

for comp, count in components.items():
    print(f"{comp}: {count:,} parameters")
