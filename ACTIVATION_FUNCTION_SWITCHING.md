# Activation Function Switching in RGB Hierarchical UNet V2

This document describes how to switch activation functions in the RGB Hierarchical UNet V2 model.

## Overview

The `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg` configuration now supports switching activation functions in layers after the pre-trained UNet. This allows for experimentation with different activation functions to potentially improve model performance.

## Supported Activation Functions

- **relu** (default): Standard ReLU activation
- **swish**: Swish activation (also known as SiLU in PyTorch)
- **gelu**: Gaussian Error Linear Unit
- **silu**: Sigmoid Linear Unit (equivalent to Swish with beta=1.0)

## Configuration

To change the activation function, modify the `ModelConfig` parameters:

```python
ModelConfig(
    # ... other parameters ...
    activation_function='swish',  # Options: 'relu', 'swish', 'gelu', 'silu'
    activation_beta=1.0,         # Beta parameter for Swish (default: 1.0)
)
```

## Usage Examples

### 1. Using ReLU (Default)

```bash
# No modification needed - ReLU is the default
python train_advanced.py \
    --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware \
    --epochs 10
```

### 2. Using Swish Activation

```bash
# Use config_modifications to set Swish activation
python train_advanced.py \
    --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware \
    --config_modifications '{"model.activation_function": "swish"}' \
    --epochs 10
```

### 3. Using GELU Activation

```bash
python train_advanced.py \
    --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware \
    --config_modifications '{"model.activation_function": "gelu"}' \
    --epochs 10
```

## Implementation Details

The activation function switching has been successfully implemented across all major components of the RGB Hierarchical UNet V2 model. When you set `activation_function='swish'`, all 50 activation functions after the pre-trained UNet will switch from ReLU to Swish.

The implementation covers:
1. **RGB Feature Extractor**: All activation functions in the feature extraction layers
2. **Hierarchical Segmentation Heads**: Both base and refined versions fully support switching
3. **All Auxiliary Modules**: Contour detection, distance transform, boundary refinement
4. **Residual Blocks**: All instances throughout the network support activation switching

### Affected Layers

The activation function switching has been fully implemented for all layers after the pre-trained UNet:

- **RGB Feature Extractor**: ✅ Fully supports activation switching (10 activations)
- **Shared Features**: ✅ Fully supports activation switching (5 activations)
- **Background vs Foreground UNet**: ✅ Fully supports activation switching (18 activations)
- **Target vs Non-target Branch**: ✅ Fully supports activation switching (all 6 activations)
- **Upsampling Layers**: ✅ Fully supports activation switching
- **Gating Mechanisms**: ✅ Fully supports activation switching
- **Contour Detection Branch**: ✅ Fully supports activation switching
- **Distance Transform Decoder**: ✅ Fully supports activation switching
- **Residual Blocks**: ✅ All ResidualBlocks support activation switching
- **Attention Modules**: ✅ ChannelAttentionModule and AttentionGate fully support activation switching
- **Pre-trained UNet layers**: ✅ NOT affected (correctly remain as ReLU to preserve pre-trained weights)

## Performance Considerations

- **ReLU**: Fastest, most memory efficient, standard choice
- **Swish/SiLU**: Smoother gradient flow, potentially better performance, slightly slower
- **GELU**: Similar benefits to Swish, used in many transformer models

## Testing

To verify activation function switching:

```bash
python test_activation_switch.py
```

This will:
1. Test default ReLU activation
2. Test Swish activation
3. Verify forward pass works correctly
4. Display which layers use which activation functions

## Notes

- The pre-trained UNet weights are not affected by activation function changes
- Only layers after the pre-trained UNet are modified
- Some auxiliary branches may still use ReLU for stability
- The `activation_beta` parameter is only used for Swish activation