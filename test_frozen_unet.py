#\!/usr/bin/env python3
"""Test that frozen UNet produces consistent outputs."""

import torch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.human_edge_detection.experiments.config_manager import ConfigManager
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model
from PIL import Image

# Load config
config_name = 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48'
config_manager = ConfigManager()
config = config_manager.get_config(config_name)

# Create two fresh models
print("Creating two fresh models with frozen weights...")
model1 = create_rgb_hierarchical_model(
    num_classes=config.model.num_classes,
    roi_size=config.model.roi_size,
    mask_size=config.model.mask_size,
    use_attention_module=config.model.use_attention_module,
    use_pretrained_unet=config.model.use_pretrained_unet,
    pretrained_weights_path=config.model.pretrained_weights_path,
    freeze_pretrained_weights=config.model.freeze_pretrained_weights,
    use_full_image_unet=config.model.use_full_image_unet,
)

model2 = create_rgb_hierarchical_model(
    num_classes=config.model.num_classes,
    roi_size=config.model.roi_size,
    mask_size=config.model.mask_size,
    use_attention_module=config.model.use_attention_module,
    use_pretrained_unet=config.model.use_pretrained_unet,
    pretrained_weights_path=config.model.pretrained_weights_path,
    freeze_pretrained_weights=config.model.freeze_pretrained_weights,
    use_full_image_unet=config.model.use_full_image_unet,
)

# Both models should be in eval mode
model1.eval()
model2.eval()

# Load a test image
test_image_path = Path(config.data.val_img_dir) / '000000127987.jpg'
image = Image.open(test_image_path).convert('RGB')
image = image.resize((640, 640), Image.BILINEAR)
image_np = np.array(image)
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0)

# Get UNet outputs from both models
with torch.no_grad():
    # Direct UNet output from model1
    unet_output1 = model1.pretrained_unet(image_tensor)
    
    # Direct UNet output from model2
    unet_output2 = model2.pretrained_unet(image_tensor)
    
    # Compare outputs
    diff = torch.abs(unet_output1 - unet_output2)
    
    print(f"\nUNet output comparison between two fresh models:")
    print(f"Model1 output shape: {unet_output1.shape}")
    print(f"Model2 output shape: {unet_output2.shape}")
    print(f"Max difference: {diff.max().item():.10f}")
    print(f"Mean difference: {diff.mean().item():.10f}")
    print(f"Are outputs identical? {torch.allclose(unet_output1, unet_output2, atol=1e-6)}")
    
    # Check if pretrained_unet is in eval mode
    print(f"\nModel1 pretrained_unet training mode: {model1.pretrained_unet.model.training}")
    print(f"Model2 pretrained_unet training mode: {model2.pretrained_unet.model.training}")
    
    # Simulate training mode and check again
    print("\nSetting models to train mode...")
    model1.train()
    model2.train()
    
    print(f"Model1 pretrained_unet training mode after train(): {model1.pretrained_unet.model.training}")
    print(f"Model2 pretrained_unet training mode after train(): {model2.pretrained_unet.model.training}")
    
    # Get outputs again in train mode
    unet_output1_train = model1.pretrained_unet(image_tensor)
    unet_output2_train = model2.pretrained_unet(image_tensor)
    
    diff_train = torch.abs(unet_output1_train - unet_output2_train)
    print(f"\nIn train mode - Max difference: {diff_train.max().item():.10f}")
    print(f"In train mode - Are outputs identical? {torch.allclose(unet_output1_train, unet_output2_train, atol=1e-6)}")

