#!/usr/bin/env python3
"""Test script to verify activation function switching in RGB Hierarchical UNet V2."""

import torch
import warnings
warnings.filterwarnings("ignore")

from src.human_edge_detection.experiments.config_manager import ExperimentConfig
from train_advanced import build_model
from src.human_edge_detection.advanced.activation_utils import Swish, SwishInplace

def test_activation_switching():
    """Test that activation functions can be switched between ReLU and Swish."""
    
    # Test config name
    config_name = 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware'
    
    # Test 1: Default activation (should be ReLU)
    print("Test 1: Default activation (ReLU)")
    print("-" * 50)
    
    config_dict = {
        'name': config_name,
        'model': {
            'use_rgb_hierarchical': True,
            'use_external_features': False,
            'use_pretrained_unet': True,
            'pretrained_weights_path': "ext_extractor/2020-09-23a.pth",
            'freeze_pretrained_weights': True,
            'use_full_image_unet': True,
            'use_attention_module': True,
            'use_boundary_refinement': False,
            'use_boundary_aware_loss': True,
            'use_contour_detection': True,
            'use_distance_transform': True,
            'roi_size': (64, 48),
            'mask_size': (64, 48),
            # Default activation
            'activation_function': 'relu',
            'activation_beta': 1.0
        },
        'data': {
            'roi_padding': 1.5,
            'val_annotation': 'data/annotations/instances_val2017_person_only_no_crowd_100.json',
            'val_img_dir': 'data/images/val2017'
        }
    }
    
    config = ExperimentConfig.from_dict(config_dict)
    model, _ = build_model(config, 'cuda')
    
    # Check activation functions in the model
    print("Checking activation functions in the model...")
    
    # Check rgb_feature_extractor
    found_relu = False
    found_swish = False
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            found_relu = True
            if not found_swish:  # Only print first few occurrences
                print(f"  Found ReLU at: {name}")
        elif isinstance(module, torch.nn.SiLU):
            found_swish = True
            print(f"  Found Swish/SiLU at: {name}")
    
    print(f"\nReLU found: {found_relu}")
    print(f"Swish/SiLU found: {found_swish}")
    
    # Test 2: Swish activation
    print("\n\nTest 2: Swish activation")
    print("-" * 50)
    
    # Create new config dict with swish activation
    config_dict_swish = {
        'name': config_name,
        'model': {
            'use_rgb_hierarchical': True,
            'use_external_features': False,
            'use_pretrained_unet': True,
            'pretrained_weights_path': "ext_extractor/2020-09-23a.pth",
            'freeze_pretrained_weights': True,
            'use_full_image_unet': True,
            'use_attention_module': True,
            'use_boundary_refinement': False,
            'use_boundary_aware_loss': True,
            'use_contour_detection': True,
            'use_distance_transform': True,
            'roi_size': (64, 48),
            'mask_size': (64, 48),
            # Swish activation
            'activation_function': 'swish',
            'activation_beta': 1.0
        },
        'data': {
            'roi_padding': 1.5,
            'val_annotation': 'data/annotations/instances_val2017_person_only_no_crowd_100.json',
            'val_img_dir': 'data/images/val2017'
        }
    }
    
    config = ExperimentConfig.from_dict(config_dict_swish)
    model, _ = build_model(config, 'cuda')
    
    # Check activation functions in the model
    print("Checking activation functions in the model...")
    
    found_relu = False
    found_swish = False
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            found_relu = True
            # In pre-trained UNet, ReLU might still exist
            if 'pretrained_unet' not in name:
                print(f"  Found ReLU at: {name}")
        elif isinstance(module, (torch.nn.SiLU, Swish, SwishInplace)):
            found_swish = True
            if 'rgb_feature_extractor' in name or 'segmentation_head' in name:
                print(f"  Found Swish/SiLU at: {name}")
    
    print(f"\nReLU found: {found_relu}")
    print(f"Swish/SiLU found: {found_swish}")
    
    # Test forward pass
    print("\n\nTest 3: Forward pass test")
    print("-" * 50)
    
    # Create dummy inputs
    batch_size = 2
    images = torch.randn(batch_size, 3, 640, 640).cuda()
    
    # Create dummy ROIs
    rois = torch.tensor([
        [0, 100, 100, 200, 200],
        [0, 150, 150, 250, 250],
        [1, 200, 200, 300, 300],
        [1, 250, 250, 350, 350]
    ], dtype=torch.float32).cuda()
    
    model.eval()
    with torch.no_grad():
        try:
            outputs, aux_outputs = model(images, rois)
            print(f"Forward pass successful!")
            print(f"Output shape: {outputs.shape}")
            print(f"Auxiliary outputs: {list(aux_outputs.keys())}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
    
    print("\n✓ Activation function switching test completed!")


if __name__ == "__main__":
    test_activation_switching()