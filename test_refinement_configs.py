#!/usr/bin/env python3
"""Test script to verify binary mask refinement configurations."""

import torch
from src.human_edge_detection.experiments.config_manager import ConfigManager
from train_advanced import build_model, build_loss_function
import json


def test_refinement_config(config_name: str):
    """Test a single refinement configuration."""
    print(f"\nTesting configuration: {config_name}")
    print("=" * 60)
    
    try:
        # Load configuration
        config = ConfigManager.get_config(config_name)
        print(f"✓ Configuration loaded successfully")
        
        # Print refinement flags
        print("\nRefinement modules enabled:")
        refinement_flags = [
            'use_boundary_refinement',
            'use_active_contour_loss', 
            'use_progressive_upsampling',
            'use_subpixel_conv',
            'use_contour_detection',
            'use_distance_transform',
            'use_boundary_aware_loss'
        ]
        
        for flag in refinement_flags:
            value = getattr(config.model, flag, False)
            if value:
                print(f"  - {flag}: {value}")
        
        # Build model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nBuilding model on {device}...")
        model, feature_extractor = build_model(config, device)
        print(f"✓ Model built successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Build loss function
        print("\nBuilding loss function...")
        try:
            with open('data_analyze_full.json', 'r') as f:
                data_stats = json.load(f)
            pixel_ratios = data_stats['pixel_ratios']
            separation_aware_weights = data_stats.get('separation_aware_weights')
        except FileNotFoundError:
            print("  Warning: data_analyze_full.json not found, using default values")
            pixel_ratios = {'background': 0.486, 'target': 0.366, 'non_target': 0.148}
            separation_aware_weights = {'background': 0.538, 'target': 0.750, 'non_target': 1.712}
        
        loss_fn = build_loss_function(
            config,
            pixel_ratios,
            device,
            separation_aware_weights
        )
        print(f"✓ Loss function built successfully")
        print(f"  Loss type: {type(loss_fn).__name__}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        batch_size = 2
        dummy_images = torch.randn(batch_size, 3, 640, 640).to(device)
        num_rois = 3
        dummy_rois = torch.zeros(num_rois, 5).to(device)
        dummy_rois[:, 0] = torch.arange(num_rois) % batch_size
        dummy_rois[:, 1:] = torch.tensor([
            [100, 100, 300, 300],
            [200, 200, 400, 400],
            [150, 150, 350, 350]
        ]).float()
        
        model.eval()
        with torch.no_grad():
            if config.model.use_rgb_hierarchical:
                output = model(dummy_images, dummy_rois)
            else:
                # For other models, would need feature extraction
                print("  (Skipping forward pass for non-RGB models)")
                return True
        
        if isinstance(output, tuple):
            masks, aux_outputs = output
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {masks.shape}")
            print(f"  Auxiliary outputs: {list(aux_outputs.keys())}")
        else:
            masks = output
            print(f"✓ Forward pass successful")
            print(f"  Output shape: {masks.shape}")
        
        # Test loss computation
        print("\nTesting loss computation...")
        dummy_targets = torch.randint(0, 3, (num_rois, config.model.mask_size, config.model.mask_size)).to(device)
        
        if isinstance(output, tuple):
            loss, loss_components = loss_fn(masks, dummy_targets, aux_outputs)
        else:
            loss, loss_components = loss_fn(masks, dummy_targets)
            
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Loss components: {list(loss_components.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test all refinement configurations."""
    print("Testing Binary Mask Refinement Configurations")
    print("=" * 80)
    
    # List of refinement configurations to test
    refinement_configs = [
        'rgb_hierarchical_unet_v2_attention_r112m224_refined',
        'rgb_hierarchical_unet_v2_attention_r96m96_refined',
        'rgb_hierarchical_unet_v2_attention_r64m64_refined',
    ]
    
    success_count = 0
    for config_name in refinement_configs:
        if test_refinement_config(config_name):
            success_count += 1
    
    print("\n" + "=" * 80)
    print(f"Summary: {success_count}/{len(refinement_configs)} configurations tested successfully")
    
    # Test ONNX export for one configuration
    print("\nTesting ONNX export...")
    test_config = refinement_configs[0]
    config = ConfigManager.get_config(test_config)
    
    from run_experiments import export_untrained_model_to_onnx
    success = export_untrained_model_to_onnx(test_config)
    
    if success:
        print("✓ ONNX export successful")
    else:
        print("✗ ONNX export failed")


if __name__ == "__main__":
    main()