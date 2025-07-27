#\!/usr/bin/env python3
"""Compare UNet outputs between pre-trained weights only and trained model."""

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

# Create two models
print("Creating model with random initialization...")
model_random = create_rgb_hierarchical_model(
    num_classes=config.model.num_classes,
    roi_size=config.model.roi_size,
    mask_size=config.model.mask_size,
    use_attention_module=config.model.use_attention_module,
    use_pretrained_unet=config.model.use_pretrained_unet,
    pretrained_weights_path=config.model.pretrained_weights_path,
    freeze_pretrained_weights=config.model.freeze_pretrained_weights,
    use_full_image_unet=config.model.use_full_image_unet,
)

print("\nCreating model with trained weights...")
model_trained = create_rgb_hierarchical_model(
    num_classes=config.model.num_classes,
    roi_size=config.model.roi_size,
    mask_size=config.model.mask_size,
    use_attention_module=config.model.use_attention_module,
    use_pretrained_unet=config.model.use_pretrained_unet,
    pretrained_weights_path=config.model.pretrained_weights_path,
    freeze_pretrained_weights=config.model.freeze_pretrained_weights,
    use_full_image_unet=config.model.use_full_image_unet,
)

# Load checkpoint for trained model
checkpoint_path = Path('experiments') / config_name / 'checkpoints' / 'checkpoint_epoch_0001.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model_trained.load_state_dict(checkpoint['model_state_dict'])

# Set both models to eval mode
model_random.eval()
model_trained.eval()

# Load a test image
test_image_path = Path(config.data.val_img_dir) / '000000127987.jpg'
image = Image.open(test_image_path).convert('RGB')
image = image.resize((640, 640), Image.BILINEAR)
image_np = np.array(image)
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
image_tensor = image_tensor.unsqueeze(0)

# Get UNet outputs from both models
with torch.no_grad():
    # Direct UNet output from random model
    unet_output_random = model_random.pretrained_unet(image_tensor)
    
    # Direct UNet output from trained model
    unet_output_trained = model_trained.pretrained_unet(image_tensor)
    
    # Compare outputs
    diff = torch.abs(unet_output_random - unet_output_trained)
    
    print(f"\nUNet output comparison:")
    print(f"Random model output shape: {unet_output_random.shape}")
    print(f"Trained model output shape: {unet_output_trained.shape}")
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")
    print(f"Are outputs identical? {torch.allclose(unet_output_random, unet_output_trained, atol=1e-6)}")
    
    # Also check some statistics
    print(f"\nRandom model UNet output - min: {unet_output_random.min().item():.4f}, max: {unet_output_random.max().item():.4f}, mean: {unet_output_random.mean().item():.4f}")
    print(f"Trained model UNet output - min: {unet_output_trained.min().item():.4f}, max: {unet_output_trained.max().item():.4f}, mean: {unet_output_trained.mean().item():.4f}")

