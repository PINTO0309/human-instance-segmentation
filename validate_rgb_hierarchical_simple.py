#!/usr/bin/env python3
"""Simple validation script for RGB Hierarchical UNet V2 model."""

import torch
import warnings
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Set batch size based on available memory
BATCH_SIZE = 2  # Reduced for memory constraints

def main():
    checkpoint_path = "experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware/checkpoints/checkpoint_epoch_0070.pth"

    print("Simple validation script for RGB Hierarchical UNet V2")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Batch size: {BATCH_SIZE}")

    # Direct command to train_advanced.py with test_only flag
    import sys
    sys.argv = [
        'train_advanced.py',
        '--config', 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware',
        '--resume', checkpoint_path,
        '--test_only',
        '--config_modifications', f'{{"training.batch_size":{BATCH_SIZE},"data.use_edge_visualize":true}}'
    ]

    # Import and run train_advanced
    import train_advanced
    train_advanced.main()

    # After validation, run visualization
    print("\n" + "="*60)
    print("Running visualization with visualize_auxiliary.py...")
    print("="*60)

    # Import necessary modules
    import torch
    import json
    from pathlib import Path
    from pycocotools.coco import COCO
    from src.human_edge_detection.experiments.config_manager import ExperimentConfig
    from src.human_edge_detection.visualize_auxiliary import ValidationVisualizerWithAuxiliary

    # Load checkpoint to get model
    checkpoint = torch.load(checkpoint_path, map_location='cuda')

    # Load config
    config_name = 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware'
    config_path = Path('experiments/configs') / f'{config_name}.json'

    if not config_path.exists():
        # If config file doesn't exist, use the configuration from checkpoint
        config_dict = checkpoint['config']
    else:
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)

    config = ExperimentConfig.from_dict(config_dict)

    # Build model (RGB hierarchical doesn't need feature extractor)
    from train_advanced import build_model
    model, feature_extractor = build_model(config, 'cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create COCO object
    val_coco = COCO(config.data.val_annotation)

    # Create visualizer
    from pathlib import Path
    exp_dir = Path('experiments') / config.name
    vis_output_dir = exp_dir / 'visualizations'
    vis_output_dir.mkdir(exist_ok=True)

    visualizer = ValidationVisualizerWithAuxiliary(
        model=model,
        feature_extractor=feature_extractor,
        coco=val_coco,
        image_dir=config.data.val_img_dir,
        output_dir=vis_output_dir,
        device='cuda',
        roi_padding=config.data.roi_padding,
        visualize_auxiliary=True,  # Force auxiliary visualization
        use_roi_comparison=config.data.use_roi_comparison,
        use_edge_visualize=True  # Force edge visualization
    )

    # Get epoch from checkpoint
    epoch = checkpoint.get('epoch', 0)

    # Run visualization
    visualizer.visualize_validation_images(epoch)
    print(f"\nVisualization complete! Images saved to: {vis_output_dir}")

if __name__ == '__main__':
    main()