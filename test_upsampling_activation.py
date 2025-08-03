#!/usr/bin/env python3
"""Test script to verify upsampling activation function switching in RGB Hierarchical UNet V2."""

import torch
import warnings
warnings.filterwarnings("ignore")

from src.human_edge_detection.experiments.config_manager import ExperimentConfig
from train_advanced import build_model

def test_upsampling_activation_switching():
    """Test that downsampling and upsampling can use different activation functions."""
    
    # Test config name
    config_name = 'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware_swish_relu_up'
    
    print("Test: Swish downsampling, ReLU upsampling")
    print("-" * 60)
    
    # Get the predefined config
    from src.human_edge_detection.experiments.config_manager import ConfigManager
    config = ConfigManager.get_config(config_name)
    
    # Build the model
    model, _ = build_model(config, 'cuda')
    
    # Check activation functions in the model
    print("\nChecking activation functions in the model...")
    print(f"Config activation_function: {config.model.activation_function}")
    print(f"Config upsampling_activation_function: {config.model.upsampling_activation_function}")
    
    # Specific checks for target_vs_nontarget_branch
    print("\nChecking target_vs_nontarget_branch activations:")
    
    # For use_attention_module=True case (ModuleList)
    if hasattr(model, 'segmentation_head') and hasattr(model.segmentation_head, 'base_head'):
        base_head = model.segmentation_head.base_head
        if hasattr(base_head, 'target_vs_nontarget_branch'):
            branch = base_head.target_vs_nontarget_branch
            
            if isinstance(branch, torch.nn.ModuleList):
                print("  Branch type: ModuleList")
                for i, module in enumerate(branch):
                    print(f"  [{i}] {module.__class__.__name__}", end="")
                    
                    # Check for activation functions
                    if hasattr(module, 'activation'):
                        activation_type = module.activation.__class__.__name__
                        print(f" - activation: {activation_type}")
                    elif hasattr(module, 'activation1'):
                        activation_type = module.activation1.__class__.__name__
                        print(f" - activation1: {activation_type}")
                    else:
                        print("")
            else:
                print("  Branch type: Sequential")
                for i, module in enumerate(branch):
                    print(f"  [{i}] {module.__class__.__name__}", end="")
                    
                    # Check for activation functions
                    if module.__class__.__name__ in ['ReLU', 'SiLU', 'Swish', 'SwishInplace']:
                        print(f" <- Activation")
                    elif hasattr(module, 'activation'):
                        activation_type = module.activation.__class__.__name__
                        print(f" - activation: {activation_type}")
                    elif hasattr(module, 'activation1'):
                        activation_type = module.activation1.__class__.__name__
                        print(f" - activation1: {activation_type}")
                    else:
                        print("")
    
    # Count activations
    swish_count = 0
    relu_count = 0
    
    for name, module in model.named_modules():
        if 'target_vs_nontarget_branch' in name:
            if isinstance(module, torch.nn.ReLU):
                relu_count += 1
                print(f"\n  Found ReLU at: {name}")
            elif isinstance(module, (torch.nn.SiLU,)) or module.__class__.__name__ in ['Swish', 'SwishInplace']:
                swish_count += 1
                print(f"\n  Found Swish/SiLU at: {name}")
    
    print(f"\nIn target_vs_nontarget_branch:")
    print(f"  ReLU count: {relu_count}")
    print(f"  Swish count: {swish_count}")
    
    # Test forward pass
    print("\n\nTest: Forward pass")
    print("-" * 60)
    
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
            import traceback
            traceback.print_exc()
    
    print("\n✓ Upsampling activation function test completed!")


if __name__ == "__main__":
    test_upsampling_activation_switching()