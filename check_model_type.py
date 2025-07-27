#\!/usr/bin/env python3
"""Check model type and structure."""

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

print(f"Model type: {type(model).__name__}")
print(f"Model class: {model.__class__.__module__}.{model.__class__.__name__}")

# Check if it has pretrained_unet
if hasattr(model, 'pretrained_unet'):
    print(f"\nHas pretrained_unet: Yes")
    print(f"Pretrained UNet type: {type(model.pretrained_unet).__name__}")
else:
    print(f"\nHas pretrained_unet: No")

# Check forward method signature
import inspect
sig = inspect.signature(model.forward)
print(f"\nForward method signature: {sig}")

