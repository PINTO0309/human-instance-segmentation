#!/usr/bin/env python3
"""
Comprehensive analysis of YOLOv9 ONNX models and their intermediate feature outputs.
This script provides detailed information about what features are available for extraction.
"""

import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_model_outputs(model_path: str) -> Dict:
    """Analyze a single ONNX model and its outputs."""
    model_name = Path(model_path).name
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()
        
        # Test with dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        outputs = session.run(None, {input_info.name: dummy_input})
        
        # Organize output information
        outputs_data = []
        for i, (out_info, out_data) in enumerate(zip(output_info, outputs)):
            outputs_data.append({
                'index': i,
                'name': out_info.name,
                'shape': list(out_data.shape),
                'channels': out_data.shape[1],
                'spatial_size': f"{out_data.shape[2]}x{out_data.shape[3]}",
                'total_params': int(np.prod(out_data.shape[1:])),  # Exclude batch dimension
                'receptive_field_estimate': estimate_receptive_field(out_data.shape[2:])
            })
        
        return {
            'model_name': model_name,
            'input_shape': [1, 3, 640, 640],
            'num_outputs': len(outputs),
            'outputs': outputs_data,
            'success': True
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'success': False,
            'error': str(e)
        }

def estimate_receptive_field(spatial_shape: Tuple[int, int]) -> str:
    """Estimate receptive field based on output spatial dimensions."""
    h, w = spatial_shape
    if h >= 320:  # Very high resolution
        return "Small (local details)"
    elif h >= 160:  # High resolution  
        return "Medium-small (fine features)"
    elif h >= 80:   # Medium resolution
        return "Medium (balanced)"
    elif h >= 40:   # Low resolution
        return "Medium-large (context)"
    else:           # Very low resolution
        return "Large (global context)"

def categorize_outputs_by_resolution(all_results: List[Dict]) -> Dict:
    """Categorize outputs by their spatial resolution for easy comparison."""
    resolution_categories = {}
    
    for result in all_results:
        if not result['success']:
            continue
            
        model_name = result['model_name'].split('_')[1]  # Extract variant (n, t, s, e, c)
        
        for output in result['outputs']:
            spatial_size = output['spatial_size']
            
            if spatial_size not in resolution_categories:
                resolution_categories[spatial_size] = {}
            
            if model_name not in resolution_categories[spatial_size]:
                resolution_categories[spatial_size][model_name] = []
            
            resolution_categories[spatial_size][model_name].append({
                'name': output['name'],
                'channels': output['channels'],
                'index': output['index']
            })
    
    return resolution_categories

def analyze_feature_progression():
    """Analyze how features progress across different model variants."""
    print("YOLOv9 ONNX Models: Intermediate Feature Analysis")
    print("=" * 80)
    
    # Models to analyze
    models = [
        'ext_extractor/yolov9_n_wholebody25_Nx3x640x640_featext_optimized.onnx',  # Nano
        'ext_extractor/yolov9_t_wholebody25_Nx3x640x640_featext_optimized.onnx',  # Tiny  
        'ext_extractor/yolov9_s_wholebody25_Nx3x640x640_featext_optimized.onnx',  # Small
        'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',  # Efficient
        'ext_extractor/yolov9_c_wholebody25_Nx3x640x640_featext_optimized.onnx',  # Compact
    ]
    
    # Analyze each model
    all_results = []
    for model_path in models:
        if Path(model_path).exists():
            result = analyze_model_outputs(model_path)
            all_results.append(result)
            
            print(f"\n{result['model_name']}:")
            if result['success']:
                print(f"  ✓ {result['num_outputs']} intermediate outputs available")
                for output in result['outputs']:
                    print(f"    [{output['index']}] {output['name']}")
                    print(f"        Shape: {output['shape']} | Channels: {output['channels']} | Size: {output['spatial_size']}")
                    print(f"        Receptive field: {output['receptive_field_estimate']} | Params: {output['total_params']:,}")
            else:
                print(f"  ✗ Error: {result['error']}")
    
    # Categorize by resolution
    print(f"\n\nFeature Maps by Spatial Resolution:")
    print("=" * 50)
    
    categories = categorize_outputs_by_resolution(all_results)
    
    # Sort by resolution (largest to smallest)
    sorted_resolutions = sorted(categories.keys(), key=lambda x: int(x.split('x')[0]), reverse=True)
    
    for resolution in sorted_resolutions:
        print(f"\n{resolution} - {estimate_receptive_field(tuple(map(int, resolution.split('x'))))}:")
        
        models_at_resolution = categories[resolution]
        for variant in ['n', 't', 's', 'e', 'c']:
            if variant in models_at_resolution:
                outputs = models_at_resolution[variant]
                channels_list = [str(out['channels']) for out in outputs]
                print(f"  {variant.upper()}: {len(outputs)} outputs, channels: {', '.join(channels_list)}")
    
    # Feature extraction recommendations
    print(f"\n\nFeature Extraction Recommendations:")
    print("=" * 40)
    
    # Current usage analysis
    print("Current implementation uses:")
    print("- Target layer: 'segmentation_model_34_Concat_output_0'")
    print("- This corresponds to the highest-channel 80x80 feature map")
    print("- Provides 1024 channels for YOLOv9-e model")
    
    print("\nAlternative feature extraction options:")
    
    # Find the 80x80 outputs across models
    for result in all_results:
        if not result['success']:
            continue
        model_variant = result['model_name'].split('_')[1]
        
        print(f"\n{model_variant.upper()} Model ({result['model_name']}):")
        
        # Group by spatial resolution
        resolution_groups = {}
        for output in result['outputs']:
            spatial = output['spatial_size']
            if spatial not in resolution_groups:
                resolution_groups[spatial] = []
            resolution_groups[spatial].append(output)
        
        for spatial in sorted(resolution_groups.keys(), key=lambda x: int(x.split('x')[0]), reverse=True):
            outputs_at_res = resolution_groups[spatial]
            if len(outputs_at_res) == 1:
                out = outputs_at_res[0]
                print(f"  {spatial}: {out['channels']} channels ('{out['name']}')")
            else:
                channels = [str(out['channels']) for out in outputs_at_res]
                print(f"  {spatial}: {len(outputs_at_res)} outputs, {', '.join(channels)} channels")
    
    # Multi-scale extraction recommendations
    print(f"\n\nMulti-Scale Feature Extraction Options:")
    print("=" * 45)
    print("For enhanced feature extraction, consider using multiple resolution layers:")
    print("1. High-res (160x160): Fine-grained details, edges, small objects")
    print("2. Medium-res (80x80): Balanced features, current default")
    print("3. Low-res (40x40 or 20x20): Global context, large object relationships")
    print("\nThis would enable feature pyramid-style processing in the decoder.")
    
    # Save results to JSON
    output_file = "feature_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nDetailed results saved to: {output_file}")

def find_best_feature_layer_for_model(model_variant: str) -> str:
    """Find the best feature extraction layer for a given model variant."""
    model_path = f'ext_extractor/yolov9_{model_variant}_wholebody25_Nx3x640x640_featext_optimized.onnx'
    
    if not Path(model_path).exists():
        return None
    
    result = analyze_model_outputs(model_path)
    if not result['success']:
        return None
    
    # Find the layer with 80x80 spatial resolution and highest channels
    best_layer = None
    best_channels = 0
    
    for output in result['outputs']:
        if output['spatial_size'] == '80x80' and output['channels'] > best_channels:
            best_channels = output['channels']
            best_layer = output['name']
    
    return best_layer

if __name__ == "__main__":
    analyze_feature_progression()
    
    # Test finding best layers for each variant
    print(f"\n\nBest Feature Layers by Model Variant:")
    print("=" * 40)
    for variant in ['n', 't', 's', 'e', 'c']:
        best_layer = find_best_feature_layer_for_model(variant)
        if best_layer:
            print(f"{variant.upper()}: {best_layer}")
        else:
            print(f"{variant.upper()}: Model not found or error")