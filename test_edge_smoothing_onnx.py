#!/usr/bin/env python3
"""
Test and demonstrate ONNX edge smoothing models with real segmentation masks.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def create_test_mask(size=(256, 256), num_objects=3):
    """Create a synthetic binary mask with rough edges for testing."""
    mask = np.zeros(size, dtype=np.float32)
    
    for i in range(num_objects):
        # Random ellipse parameters
        center = (np.random.randint(50, size[0]-50), 
                 np.random.randint(50, size[1]-50))
        axes = (np.random.randint(20, 60), np.random.randint(20, 60))
        angle = np.random.randint(0, 360)
        
        # Draw filled ellipse
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
    
    # Add some noise to create rough edges
    noise = np.random.randn(*size) * 0.1
    mask = np.clip(mask + noise, 0, 1)
    mask = (mask > 0.5).astype(np.float32)
    
    return mask


def apply_edge_smoothing(mask, model_path, model_type='basic'):
    """Apply edge smoothing using ONNX model."""
    
    # Create inference session
    session = ort.InferenceSession(model_path)
    
    # Prepare input
    if mask.ndim == 2:
        mask = mask[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
    
    # Prepare inputs based on model type
    if model_type == 'adaptive':
        inputs = {
            'mask': mask.astype(np.float32),
            'blur_strength': np.array([3.0], dtype=np.float32),
            'edge_sensitivity': np.array([1.0], dtype=np.float32),
            'final_threshold': np.array([0.5], dtype=np.float32)
        }
    elif model_type == 'fp16':
        inputs = {'mask': mask.astype(np.float16)}
    else:
        inputs = {'mask': mask.astype(np.float32)}
    
    # Run inference
    output = session.run(None, inputs)[0]
    
    # Remove batch and channel dimensions
    return output[0, 0]


def visualize_results(original_mask, smoothed_masks, model_names):
    """Visualize original and smoothed masks."""
    
    n_models = len(smoothed_masks)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))
    
    # Original mask
    axes[0].imshow(original_mask, cmap='gray')
    axes[0].set_title('Original Mask')
    axes[0].axis('off')
    
    # Smoothed masks
    for i, (mask, name) in enumerate(zip(smoothed_masks, model_names)):
        axes[i + 1].imshow(mask, cmap='gray')
        axes[i + 1].set_title(name)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    return fig


def compare_edge_quality(original_mask, smoothed_mask):
    """Compare edge quality between original and smoothed masks."""
    
    # Compute edges using Canny
    original_edges = cv2.Canny((original_mask * 255).astype(np.uint8), 50, 150)
    smoothed_edges = cv2.Canny((smoothed_mask * 255).astype(np.uint8), 50, 150)
    
    # Count edge pixels
    original_edge_pixels = np.sum(original_edges > 0)
    smoothed_edge_pixels = np.sum(smoothed_edges > 0)
    
    # Compute edge smoothness (lower is smoother)
    # Using the variance of edge normal directions as a proxy
    def compute_edge_smoothness(edges):
        # Find edge pixels
        edge_points = np.argwhere(edges > 0)
        if len(edge_points) < 10:
            return 0
        
        # Compute local gradients
        sobelx = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute angle variance at edge points
        angles = []
        for y, x in edge_points:
            if sobelx[y, x] != 0 or sobely[y, x] != 0:
                angle = np.arctan2(sobely[y, x], sobelx[y, x])
                angles.append(angle)
        
        if angles:
            return np.var(angles)
        return 0
    
    original_smoothness = compute_edge_smoothness(original_edges)
    smoothed_smoothness = compute_edge_smoothness(smoothed_edges)
    
    return {
        'original_edge_pixels': original_edge_pixels,
        'smoothed_edge_pixels': smoothed_edge_pixels,
        'edge_reduction': (original_edge_pixels - smoothed_edge_pixels) / original_edge_pixels * 100,
        'original_smoothness': original_smoothness,
        'smoothed_smoothness': smoothed_smoothness,
        'smoothness_improvement': (original_smoothness - smoothed_smoothness) / original_smoothness * 100
    }


def main():
    parser = argparse.ArgumentParser(description="Test ONNX edge smoothing models")
    parser.add_argument('--model-dir', default='onnx_models', help='Directory containing ONNX models')
    parser.add_argument('--mask-path', help='Path to input mask image (optional)')
    parser.add_argument('--output', help='Output path for visualization')
    parser.add_argument('--size', type=int, default=256, help='Size of test mask if not provided')
    
    args = parser.parse_args()
    
    # Load or create test mask
    if args.mask_path:
        original_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        original_mask = (original_mask > 127).astype(np.float32)
    else:
        print("Creating synthetic test mask...")
        original_mask = create_test_mask((args.size, args.size))
    
    # Define models to test
    models = [
        ('basic_edge_smoothing.onnx', 'Basic', 'basic'),
        ('directional_edge_smoothing.onnx', 'Directional', 'basic'),
        ('adaptive_edge_smoothing.onnx', 'Adaptive', 'adaptive'),
        ('optimized_edge_smoothing_fp32.onnx', 'Optimized', 'basic'),
    ]
    
    smoothed_masks = []
    model_names = []
    
    # Apply each model
    print("\nApplying edge smoothing models...")
    for model_file, name, model_type in models:
        model_path = Path(args.model_dir) / model_file
        if model_path.exists():
            print(f"  - {name}...")
            try:
                smoothed = apply_edge_smoothing(original_mask, str(model_path), model_type)
                smoothed_masks.append(smoothed)
                model_names.append(name)
                
                # Compute metrics
                metrics = compare_edge_quality(original_mask, smoothed)
                print(f"    Edge pixels: {metrics['original_edge_pixels']} -> {metrics['smoothed_edge_pixels']} "
                      f"({metrics['edge_reduction']:.1f}% reduction)")
                if metrics['smoothness_improvement'] > 0:
                    print(f"    Smoothness improved by {metrics['smoothness_improvement']:.1f}%")
                
            except Exception as e:
                print(f"    Error: {e}")
    
    # Visualize results
    if smoothed_masks:
        print("\nGenerating visualization...")
        fig = visualize_results(original_mask, smoothed_masks, model_names)
        
        if args.output:
            fig.savefig(args.output, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {args.output}")
        else:
            plt.show()
    else:
        print("No models were successfully applied.")


if __name__ == "__main__":
    main()