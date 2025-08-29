# ONNX Edge Smoothing Models

This directory contains ONNX models for binary mask edge smoothing, exported from PyTorch implementations.

## Available Models

### 1. Basic Edge Smoothing (`basic_edge_smoothing.onnx`)
- Simple edge detection and smoothing using Laplacian and Gaussian filters
- Fixed parameters: threshold=0.5, blur_strength=3.0
- Input: mask [batch, 1, height, width]
- Output: smoothed_mask [batch, 1, height, width]

### 2. Directional Edge Smoothing (`directional_edge_smoothing.onnx`)
- Edge smoothing that considers edge direction
- Uses Sobel filters for edge direction detection
- Applies directional blur based on edge orientation
- Input: mask [batch, 1, height, width]
- Output: smoothed_mask [batch, 1, height, width]

### 3. Adaptive Edge Smoothing (`adaptive_edge_smoothing.onnx`)
- Dynamically adjustable parameters
- Inputs:
  - mask [batch, 1, height, width]
  - blur_strength [1] - Blur intensity (1.0-5.0)
  - edge_sensitivity [1] - Edge detection sensitivity (0.5-2.0)
  - final_threshold [1] - Final binarization threshold (0.3-0.7)
- Output: smoothed_mask [batch, 1, height, width]

### 4. Optimized Edge Smoothing FP32 (`optimized_edge_smoothing_fp32.onnx`)
- Performance-optimized implementation
- Uses separable convolutions for efficient Gaussian blur
- Input: mask [batch, 1, height, width]
- Output: smoothed_mask [batch, 1, height, width]

### 5. Optimized Edge Smoothing FP16 (`optimized_edge_smoothing_fp16.onnx`)
- Same as FP32 version but with half-precision
- Faster inference on GPUs with FP16 support
- Input: mask [batch, 1, height, width] (FP16)
- Output: smoothed_mask [batch, 1, height, width] (FP16)

## Usage Example

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("basic_edge_smoothing.onnx")

# Prepare input
mask = np.random.rand(1, 1, 256, 256).astype(np.float32)

# Run inference
output = session.run(None, {"mask": mask})[0]

# For adaptive model
adaptive_session = ort.InferenceSession("adaptive_edge_smoothing.onnx")
output = adaptive_session.run(None, {
    "mask": mask,
    "blur_strength": np.array([3.0], dtype=np.float32),
    "edge_sensitivity": np.array([1.0], dtype=np.float32),
    "final_threshold": np.array([0.5], dtype=np.float32)
})[0]
```

## Model Details

All models support:
- Dynamic batch size
- Dynamic spatial dimensions (height, width)
- ONNX opset version 11

The models are optimized for inference with:
- Constant folding
- Graph optimization
- Efficient memory usage

## Source

These models are exported from PyTorch implementations defined in `Refine_the_Binary_Mask.md`.
To regenerate the models, run:

```bash
python export_edge_smoothing_onnx.py --fp16
```