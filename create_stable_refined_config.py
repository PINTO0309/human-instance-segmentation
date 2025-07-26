#!/usr/bin/env python3
"""Create a more stable refined configuration with gradual refinement activation."""

import json
from pathlib import Path

# Read the current config
with open('src/human_edge_detection/experiments/config_manager.py', 'r') as f:
    content = f.read()

# Create new stable configuration
stable_config = '''
        # Stable refined configuration - start with minimal refinements
        'rgb_hierarchical_unet_v2_attention_r64m64_refined_stable': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_stable',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Start with only boundary refinement
                use_boundary_refinement=True,
                use_active_contour_loss=False,  # Disabled initially
                use_progressive_upsampling=False,  # Disabled - can be unstable
                use_subpixel_conv=False,
                use_contour_detection=False,  # Disabled initially
                use_distance_transform=False,  # Disabled initially
                use_boundary_aware_loss=True,  # Keep this for better boundaries
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-6,     # Further reduced
                warmup_epochs=15,       # More warmup
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=0.5,      # More aggressive clipping
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.00001,   # Very small weight decay
                min_lr=1e-8,            # Lower minimum
                mixed_precision=False   # Disable for stability
            )
        ),
'''

# Find where to insert the new config (after the last refined config)
import re
pattern = r"('rgb_hierarchical_unet_v2_attention_r64m64_refined':[^}]+\}\s*\),)"
match = re.search(pattern, content, re.DOTALL)

if match:
    # Insert after the match
    insert_pos = match.end()
    new_content = content[:insert_pos] + "\n" + stable_config + content[insert_pos:]
    
    # Write back
    with open('src/human_edge_detection/experiments/config_manager.py', 'w') as f:
        f.write(new_content)
    
    print("Created stable refined configuration: rgb_hierarchical_unet_v2_attention_r64m64_refined_stable")
    print("\nKey differences:")
    print("- Only 2 refinement modules enabled (boundary_refinement, boundary_aware_loss)")
    print("- Learning rate: 5e-6 (reduced from 1e-5)")
    print("- Warmup epochs: 15 (increased from 10)")
    print("- Gradient clip: 0.5 (reduced from 1.0)")
    print("- Mixed precision: False (disabled for stability)")
    print("- Weight decay: 1e-5 (reduced from 1e-4)")
else:
    print("Could not find insertion point in config_manager.py")