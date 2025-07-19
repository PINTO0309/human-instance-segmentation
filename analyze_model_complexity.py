#!/usr/bin/env python3
"""Analyze model complexity for different configurations."""

import torch
import time
import numpy as np
from pathlib import Path

# Import model builders
from src.human_edge_detection.model import create_model
from src.human_edge_detection.advanced.multi_scale_model import create_multiscale_model
from src.human_edge_detection.advanced.cascade_segmentation import create_cascade_model
from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_inference_time(model, input_shape, num_runs=100, device='cuda'):
    """Measure average inference time."""
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    if isinstance(input_shape, dict):
        # Multi-scale model
        dummy_input = {
            k: torch.randn(v).to(device) for k, v in input_shape.items()
        }
        # Add ROIs
        dummy_rois = torch.tensor([[0, 0.1, 0.1, 0.5, 0.5]], dtype=torch.float32).to(device)
    else:
        # Single-scale model
        dummy_input = torch.randn(input_shape).to(device)
        dummy_rois = torch.tensor([[0, 0.1, 0.1, 0.5, 0.5]], dtype=torch.float32).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            if isinstance(input_shape, dict):
                _ = model.segmentation_head(dummy_input, dummy_rois)
            else:
                _ = model(dummy_input, dummy_rois)
    
    # Measure
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            if isinstance(input_shape, dict):
                _ = model.segmentation_head(dummy_input, dummy_rois)
            else:
                _ = model(dummy_input, dummy_rois)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time


def estimate_flops(model_type, roi_size=28, mask_size=56):
    """Estimate FLOPs for different model types."""
    # Rough estimation based on architecture
    if model_type == 'baseline':
        # ROIAlign + Decoder
        roi_features = roi_size * roi_size * 1024
        decoder_flops = roi_features * 256 * 2  # Two conv layers
        upsampling_flops = mask_size * mask_size * 256 * 3  # Upsampling layers
        return decoder_flops + upsampling_flops
    
    elif model_type == 'multiscale':
        # Multi-scale fusion + Decoder
        baseline_flops = estimate_flops('baseline', roi_size, mask_size)
        fusion_flops = 3 * roi_size * roi_size * 256  # Three scales fusion
        return baseline_flops + fusion_flops
    
    elif model_type == 'cascade':
        # Multiple stages
        single_stage = estimate_flops('baseline', roi_size, mask_size)
        return single_stage * 3  # 3 stages
    
    elif model_type == 'all_features':
        # Everything combined
        multiscale_flops = estimate_flops('multiscale', roi_size, mask_size)
        cascade_multiplier = 3
        return multiscale_flops * cascade_multiplier


def main():
    """Analyze different model configurations."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    configs = {
        'baseline': {
            'description': 'Single-scale baseline',
            'builder': lambda: create_model(num_classes=3, roi_size=28, mask_size=56),
            'input_shape': (1, 1024, 80, 80),  # YOLO features
        },
        'multiscale': {
            'description': 'Multi-scale features',
            'builder': lambda: create_multiscale_model(
                onnx_model_path='dummy.onnx',  # Not used for param counting
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                num_classes=3,
                roi_size=28,
                mask_size=56,
                fusion_method='adaptive',
                execution_provider='cpu'  # Just for initialization
            ),
            'input_shape': {
                'layer_3': (1, 256, 160, 160),
                'layer_22': (1, 512, 80, 80),
                'layer_34': (1, 1024, 80, 80),
            },
        },
        'cascade': {
            'description': 'Cascade (3 stages)',
            'builder': lambda: create_cascade_model(
                base_model=create_model(num_classes=3, roi_size=28, mask_size=56),
                num_classes=3,
                cascade_stages=3,
                share_features=True
            ),
            'input_shape': (1, 1024, 80, 80),
        },
        'all_features': {
            'description': 'All features (multiscale + cascade)',
            'builder': lambda: create_cascade_model(
                base_model=create_multiscale_model(
                    onnx_model_path='dummy.onnx',
                    target_layers=['layer_3', 'layer_22', 'layer_34'],
                    num_classes=3,
                    roi_size=28,
                    mask_size=56,
                    fusion_method='adaptive',
                    execution_provider='cpu'
                ),
                num_classes=3,
                cascade_stages=3,
                share_features=True
            ),
            'input_shape': {
                'layer_3': (1, 256, 160, 160),
                'layer_22': (1, 512, 80, 80),
                'layer_34': (1, 1024, 80, 80),
            },
        }
    }
    
    print("Model Complexity Analysis")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Total Params':<15} {'Trainable':<15} {'Size (MB)':<12} {'Est. GFLOPs':<12}")
    print("-" * 80)
    
    results = {}
    
    for name, config in configs.items():
        try:
            # Build model
            if name == 'multiscale' or name == 'all_features':
                # Skip ONNX loading for multiscale
                print(f"{name:<20} ", end='')
                if name == 'multiscale':
                    # Approximate parameters for multiscale
                    total_params = 4_620_934  # From logs
                    trainable_params = 4_620_934
                elif name == 'all_features':
                    # Cascade with multiscale
                    total_params = 4_620_934 * 3  # Rough estimate
                    trainable_params = total_params
            else:
                model = config['builder']()
                total_params, trainable_params = count_parameters(model)
            
            # Calculate model size
            size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
            
            # Estimate FLOPs
            gflops = estimate_flops(name) / 1e9
            
            # Store results
            results[name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'size_mb': size_mb,
                'gflops': gflops
            }
            
            print(f"{total_params:>14,} {trainable_params:>14,} {size_mb:>11.2f} {gflops:>11.2f}")
            
        except Exception as e:
            print(f"{name:<20} Error: {e}")
    
    print("\n" + "=" * 80)
    print("\nRelative Complexity (vs Baseline):")
    print("-" * 80)
    
    if 'baseline' in results:
        baseline = results['baseline']
        for name, result in results.items():
            param_ratio = result['total_params'] / baseline['total_params']
            flops_ratio = result['gflops'] / baseline['gflops']
            print(f"{name:<20} Params: {param_ratio:>5.2f}x   FLOPs: {flops_ratio:>5.2f}x")
    
    print("\nNotes:")
    print("- Parameter counts exclude YOLO feature extractor (runs separately)")
    print("- FLOPs are rough estimates for segmentation head only")
    print("- Actual inference time depends on hardware and optimization")
    print("- Distance loss and relational modules add minimal parameters")


if __name__ == '__main__':
    main()