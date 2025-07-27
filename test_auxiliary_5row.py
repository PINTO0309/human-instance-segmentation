#!/usr/bin/env python3
"""Test 5-row visualization with auxiliary visualizer."""

import sys
import torch
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.human_edge_detection.experiments.config_manager import ConfigManager
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model
from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
from src.human_edge_detection.visualize_auxiliary import ValidationVisualizerWithAuxiliary
from pycocotools.coco import COCO


def test_visualization(checkpoint_path=None):
    """Test visualization with full-image UNet model using auxiliary visualizer."""
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
        use_boundary_refinement=config.model.use_boundary_refinement,
        activation_function=config.model.activation_function,
        activation_beta=config.model.activation_beta,
        normalization_type=config.model.normalization_type,
        normalization_groups=config.model.normalization_groups,
        normalization_mix_ratio=config.model.normalization_mix_ratio,
        use_pretrained_unet=config.model.use_pretrained_unet,
        pretrained_weights_path=config.model.pretrained_weights_path,
        freeze_pretrained_weights=config.model.freeze_pretrained_weights,
        use_full_image_unet=config.model.use_full_image_unet,
    )
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path('experiments') / config_name / 'checkpoints' / 'best_model.pth'
    else:
        checkpoint_path = Path(checkpoint_path)
        
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            # Standard checkpoint format
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        elif 'state_dict' in checkpoint:
            # Pre-trained weights format (e.g., ext_extractor/2020-09-23a.pth)
            print("WARNING: This appears to be a pre-trained weights file, not a full model checkpoint")
            print("The model already loads these weights during initialization")
        else:
            # Direct state dict
            try:
                model.load_state_dict(checkpoint)
                print("Loaded direct state dict")
            except:
                print("ERROR: Unknown checkpoint format")
                raise
    else:
        print(f"WARNING: No checkpoint found at {checkpoint_path}, using randomly initialized model")
    
    # Move model to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Create feature extractor
    feature_extractor = YOLOv9FeatureExtractor(
        'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
        device=device
    )
    
    # Load COCO dataset
    val_coco = COCO(config.data.val_annotation)
    
    # Create auxiliary visualizer (as used in run_experiments.py)
    visualizer = ValidationVisualizerWithAuxiliary(
        model=model,
        feature_extractor=feature_extractor,
        coco=val_coco,
        image_dir=config.data.val_img_dir,
        output_dir='test_auxiliary_visualization',
        device=device,
        roi_padding=config.data.roi_padding,
        visualize_auxiliary=True
    )
    
    # Run visualization for epoch 0
    print("Running auxiliary visualization test...")
    visualizer.visualize_validation_images(epoch=0)
    print("Visualization complete! Check test_auxiliary_visualization/ directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test 5-row visualization with auxiliary visualizer')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (default: checkpoints/CONFIG_NAME/latest_model.pth)')
    args = parser.parse_args()
    
    test_visualization(checkpoint_path=args.checkpoint)