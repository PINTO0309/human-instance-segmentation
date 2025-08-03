# Mixed Activation Functions in Target/Non-Target Branch

This document describes the enhanced activation function configuration that allows using different activation functions for downsampling and upsampling phases in the target/non-target separation layers.

## Overview

The RGB Hierarchical UNet V2 model now supports using different activation functions for:
- **Downsampling phase**: Feature extraction and processing before upsampling
- **Upsampling phase**: Feature reconstruction after upsampling

This allows for optimized performance by using activation functions suited to each phase.

## Configuration

A new configuration has been added: `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware_swish_relu_up`

This configuration uses:
- **Swish** for downsampling (feature extraction)
- **ReLU** for upsampling (reconstruction)

## Usage

### Using the Pre-defined Configuration

```bash
python train_advanced.py \
    --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware_swish_relu_up \
    --epochs 10
```

### Custom Configuration

You can create custom configurations by specifying both activation functions:

```python
ModelConfig(
    # ... other parameters ...
    activation_function='swish',  # For downsampling
    activation_beta=1.0,
    upsampling_activation_function='relu',  # For upsampling
    upsampling_activation_beta=1.0,
)
```

Or via command line:

```bash
python train_advanced.py \
    --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware \
    --config_modifications '{"model.activation_function": "swish", "model.upsampling_activation_function": "relu"}' \
    --epochs 10
```

## Architecture Details

In the target/non-target branch, the activation functions are applied as follows:

```
Input Features
      ↓
[ResidualBlock with Swish]    ← Downsampling phase
      ↓
[SpatialAttention]
      ↓
[ConvTranspose2d]             ← Upsampling operation
      ↓
[Normalization + ReLU]        ← Upsampling phase begins
      ↓
[ChannelAttention with ReLU]  ← Upsampling phase
      ↓
[ResidualBlock with ReLU]     ← Upsampling phase
      ↓
Output (2 channels)
```

## Benefits

1. **Swish for downsampling**: 
   - Smoother gradients for better feature extraction
   - Non-monotonic properties help capture complex patterns
   - Better performance in deep networks

2. **ReLU for upsampling**:
   - Computational efficiency
   - Proven performance for reconstruction tasks
   - Faster inference

## Implementation Notes

- If `upsampling_activation_function` is not specified, it defaults to using the same activation as `activation_function`
- This feature is currently implemented for the target/non-target branch in hierarchical segmentation models
- The background/foreground branch continues to use the primary `activation_function` throughout

## Test Results

Testing confirms the correct activation placement:
- Downsampling components: 2 Swish activations
- Upsampling components: 4 ReLU activations
- Forward pass: Successfully processes batched inputs

## Available Activation Options

Both `activation_function` and `upsampling_activation_function` support:
- `'relu'`: Standard ReLU
- `'swish'` or `'silu'`: Swish/SiLU activation
- `'gelu'`: Gaussian Error Linear Unit

Choose the combination that works best for your specific use case!