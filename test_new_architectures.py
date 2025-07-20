#!/usr/bin/env python3
"""Test script for new architectures (hierarchical_segmentation and class_specific_decoder)."""

import subprocess
import sys

def test_architecture(config_name: str, epochs: int = 0):
    """Test a specific architecture configuration."""
    print(f"\n{'='*60}")
    print(f"Testing {config_name} architecture")
    print(f"{'='*60}")
    
    cmd = [
        'uv', 'run', 'python', 'run_experiments.py',
        '--configs', config_name,
        '--epochs', str(epochs),
        '--export_onnx'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✓ {config_name} export successful!")
    else:
        print(f"✗ {config_name} export failed with code {result.returncode}")
        
    return result.returncode == 0

def main():
    """Test both new architectures."""
    print("Testing new architectures for ONNX export...")
    
    # Test hierarchical segmentation
    hierarchical_success = test_architecture('hierarchical_segmentation', epochs=0)
    
    # Test class-specific decoder
    class_specific_success = test_architecture('class_specific_decoder', epochs=0)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Hierarchical Segmentation: {'✓ Success' if hierarchical_success else '✗ Failed'}")
    print(f"  Class-Specific Decoder: {'✓ Success' if class_specific_success else '✗ Failed'}")
    print(f"{'='*60}")
    
    # Return 0 if all tests passed
    return 0 if (hierarchical_success and class_specific_success) else 1

if __name__ == "__main__":
    sys.exit(main())