"""Test script for RGB-based hierarchical model."""

import torch
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import create_rgb_hierarchical_model
from src.human_edge_detection.experiments.config_manager import ConfigManager


def test_rgb_hierarchical_model():
    """Test RGB hierarchical model functionality."""
    print("Testing RGB Hierarchical Model...")
    
    # Load configuration
    config = ConfigManager.get_config('rgb_hierarchical_unet_v2')
    print(f"Config: {config.name}")
    print(f"Description: {config.description}")
    
    # Create model
    model = create_rgb_hierarchical_model(
        roi_size=config.model.roi_size,
        mask_size=config.model.mask_size,
        multi_scale=False
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    num_rois = 4
    
    # Create dummy inputs
    images = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # Create ROIs (batch_idx, x1, y1, x2, y2)
    rois = []
    for i in range(batch_size):
        for j in range(num_rois // batch_size):
            # Random ROI within image bounds
            x1 = torch.randint(0, 500, (1,)).item()
            y1 = torch.randint(0, 500, (1,)).item()
            x2 = x1 + torch.randint(50, 140, (1,)).item()
            y2 = y1 + torch.randint(50, 140, (1,)).item()
            rois.append([i, x1, y1, x2, y2])
    
    rois = torch.tensor(rois, dtype=torch.float32).to(device)
    
    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  ROIs: {rois.shape}")
    
    # Forward pass
    with torch.no_grad():
        predictions, aux_outputs = model(images, rois)
    
    print(f"\nOutput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Expected: (num_rois={num_rois}, num_classes=3, mask_size={config.model.mask_size}, mask_size={config.model.mask_size})")
    
    print(f"\nAuxiliary outputs:")
    for key, value in aux_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")
    
    # Test multi-scale version
    print("\n" + "="*50)
    print("Testing Multi-scale RGB Hierarchical Model...")
    
    # Load multi-scale configuration
    config_ms = ConfigManager.get_config('rgb_hierarchical_unet_v2_multiscale')
    
    # Create multi-scale model
    model_ms = create_rgb_hierarchical_model(
        roi_size=config_ms.model.roi_size,
        mask_size=config_ms.model.mask_size,
        multi_scale=True,
        roi_sizes=config_ms.model.variable_roi_sizes,
        fusion_method=config_ms.multiscale.fusion_method
    )
    
    model_ms = model_ms.to(device)
    
    # Forward pass
    with torch.no_grad():
        predictions_ms, aux_outputs_ms = model_ms(images, rois)
    
    print(f"\nMulti-scale output shapes:")
    print(f"  Predictions: {predictions_ms.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_ms = sum(p.numel() for p in model_ms.parameters())
    
    print(f"\nModel parameters:")
    print(f"  Single-scale: {total_params:,}")
    print(f"  Multi-scale: {total_params_ms:,}")
    
    print("\nTest completed successfully!")


if __name__ == "__main__":
    test_rgb_hierarchical_model()