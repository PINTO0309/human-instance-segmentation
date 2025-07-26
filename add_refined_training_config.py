#!/usr/bin/env python3
"""Script to add modified training configurations for refined models to prevent NaN loss."""

import json
from pathlib import Path

# Read current config_manager.py
config_path = Path("src/human_edge_detection/experiments/config_manager.py")
with open(config_path, 'r') as f:
    content = f.read()

# Find the refined configurations and update their training parameters
refined_configs = [
    'rgb_hierarchical_unet_v2_attention_r112m224_refined',
    'rgb_hierarchical_unet_v2_attention_r112m160_refined',
    'rgb_hierarchical_unet_v2_attention_r112m112_refined',
    'rgb_hierarchical_unet_v2_attention_r96m96_refined',
    'rgb_hierarchical_unet_v2_attention_r80m160_refined',
    'rgb_hierarchical_unet_v2_attention_r80m80_refined',
    'rgb_hierarchical_unet_v2_attention_r64m128_refined',
    'rgb_hierarchical_unet_v2_attention_r64m80_refined',
    'rgb_hierarchical_unet_v2_attention_r64m64_refined',
    'rgb_hierarchical_unet_v2_attention_r56m112_refined',
    'rgb_hierarchical_unet_v2_attention_r56m56_refined',
    'rgb_hierarchical_unet_v2_attention_r48m96_refined',
    'rgb_hierarchical_unet_v2_attention_r48m48_refined',
    'rgb_hierarchical_unet_v2_attention_r40m80_refined',
    'rgb_hierarchical_unet_v2_attention_r40m40_refined',
    'rgb_hierarchical_unet_v2_attention_r32m64_refined',
]

# Modified training config for refined models
refined_training_config = """training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=10,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )"""

# Replace training configs for refined models
import re

for config_name in refined_configs:
    # Find the config block
    pattern = rf"'{config_name}'.*?training=TrainingConfig\([^)]+\)"
    
    def replacer(match):
        # Replace the training config part
        return re.sub(
            r'training=TrainingConfig\([^)]+\)',
            refined_training_config,
            match.group(0)
        )
    
    content = re.sub(pattern, replacer, content, flags=re.DOTALL)

# Write back the modified content
with open(config_path, 'w') as f:
    f.write(content)

print("Updated training configurations for refined models:")
print("- learning_rate: 5e-5 → 1e-5")
print("- warmup_epochs: 5 → 10")
print("- gradient_clip: 5.0 → 1.0")
print("- weight_decay: 0.001 → 0.0001")
print("- min_lr: 1e-6 → 1e-7")