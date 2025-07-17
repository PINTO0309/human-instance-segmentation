# ROI-based Instance Segmentation for Human Detection

This repository implements a lightweight ROI-based instance segmentation model for human detection using YOLOv9 features and a custom segmentation decoder.

## Overview

The model uses a 3-class segmentation approach:
- **Class 0**: Background
- **Class 1**: Target mask (primary instance in ROI)
- **Class 2**: Non-target mask (other instances in ROI)

This formulation helps the model better distinguish between multiple person instances in crowded scenes.

## Quick Start

1. **Environment Setup**
   ```bash
   uv sync
   ```

2. **Test Pipeline**
   ```bash
   uv run python test_pipeline.py
   ```

3. **Train Model (minimal dataset)**
   ```bash
   uv run python main.py --epochs 10
   ```

## Model Architecture

### Overview

The model consists of three main components:

1. **Feature Extractor**: Pre-trained YOLOv9 model (ONNX format)
   - Input: 640×640×3 RGB images
   - Output: 1024×80×80 intermediate features
   - Feature stride: 8 (640÷80)

2. **ROI Segmentation Head**: Custom lightweight decoder
   - Input: YOLOv9 features + ROI coordinates
   - Processing: DynamicRoIAlign → Conv blocks → Upsampling
   - Output: 56×56×3 segmentation masks per ROI

3. **Loss Function**: Combined weighted loss
   - Weighted CrossEntropy for all classes
   - Dice loss specifically for target class (class 1)
   - Separation-aware weights to handle class imbalance

### Architecture Details

```
YOLOv9 Features (1024×80×80)
       ↓
DynamicRoIAlign (→ 1024×28×28)  ← Improved: 2x larger ROI
       ↓
Conv 1×1 + LayerNorm + ReLU (→ 256×28×28)
       ↓
Residual Block 1 (→ 256×28×28)
       ↓
Residual Block 2 (→ 256×28×28)
       ↓
ConvTranspose + Refine (→ 256×56×56)
       ↓
ConvTranspose + Refine (→ 128×112×112)
       ↓
Multi-scale Fusion (→ 128×56×56)
       ↓
Conv 1×1 (→ 3×56×56)
```

### ROI Processing Details

#### DynamicRoIAlign Operation

The DynamicRoIAlign module extracts fixed-size features from variable-sized ROIs:

1. **Input Coordinates**: ROIs are specified as `[batch_idx, x1, y1, x2, y2]` in original image space (640×640)

2. **Coordinate Transformation**:
   - Image space coordinates are scaled to feature space using `spatial_scale = 1/8`
   - Example: ROI [0, 160, 160, 320, 320] → feature space [0, 20, 20, 40, 40]

3. **28×28 Grid Sampling**:
   - The ROI region is divided into a 28×28 grid
   - Each grid point represents a sampling location within the ROI
   - Bilinear interpolation is used to extract features at each point

#### Grid Interpretation

The 28×28 output grid provides a spatially normalized representation of each ROI with improved resolution:

```
28×28 Grid Mapping for Human Detection:
┌─────────────────────────────┐
│ Row 0-9:    Head/Shoulders  │  Each cell (i,j) represents:
│ Row 10-18:  Torso/Arms      │  - (0,0): Top-left of ROI
│ Row 19-27:  Legs/Lower body │  - (0,27): Top-right of ROI
├─────────────────────────────┤  - (27,0): Bottom-left of ROI
│ Col 0-9:    Left side       │  - (27,27): Bottom-right of ROI
│ Col 10-18:  Center          │  - (13,13): Center of ROI
│ Col 19-27:  Right side      │
└─────────────────────────────┘
```

#### Non-Square ROI Handling

For non-square ROIs (e.g., tall person bounding box):
- Horizontal sampling: width ÷ 28 pixels per sample
- Vertical sampling: height ÷ 28 pixels per sample
- The 28×28 grid adapts to the ROI aspect ratio while maintaining consistent output size

Example with 200×100 ROI:
```
Original ROI (200×100 pixels)     →    Normalized Grid (28×28)
┌────────────────────┐                 ┌─────────────┐
│ Sparse vertical    │                 │ ● ● ● ● ● ● │
│ sampling (~3.5px)  │        →        │ ● ● ● ● ● ● │
│ Dense horizontal   │                 │ ● ● ● ● ● ● │
│ sampling (~7px)    │                 └─────────────┘
└────────────────────┘
```

### Key Design Decisions

1. **LayerNorm instead of BatchNorm/InstanceNorm**: Better ONNX compatibility and stable inference

2. **Enhanced 28×28 ROI size**: Improved spatial resolution for better detail capture

3. **3-class formulation**: Explicitly modeling non-target instances improves separation in crowded scenes

4. **DynamicRoIAlign**: Custom implementation for better ONNX export with opset 16

## Mask Output Interpretation and Overlay

### Understanding the Output

The model outputs masks with shape `[num_rois, 3, 56, 56]`:
- `num_rois`: Number of ROIs processed
- `3`: Three classes (background, target, non-target)
- `56×56`: Fixed mask resolution

### Steps to Overlay Masks on Original Image

#### 1. Extract Class Predictions

```python
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Model output: masks [N, 3, 56, 56]
# Apply softmax to get probabilities
mask_probs = F.softmax(masks, dim=1)  # [N, 3, 56, 56]

# Get target mask (class 1) probabilities
target_masks = mask_probs[:, 1, :, :]  # [N, 56, 56]

# Optionally, get predicted class per pixel
predicted_classes = torch.argmax(mask_probs, dim=1)  # [N, 56, 56]
```

#### 2. Resize Masks to ROI Dimensions

Each 56×56 mask must be resized to match its corresponding ROI size:

```python
# ROIs format: [batch_idx, x1, y1, x2, y2]
resized_masks = []

for i, roi in enumerate(rois):
    x1, y1, x2, y2 = roi[1:].int()
    roi_width = x2 - x1
    roi_height = y2 - y1

    # Resize 56×56 mask to ROI size
    mask = target_masks[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 56, 56]
    resized_mask = F.interpolate(
        mask,
        size=(roi_height, roi_width),
        mode='bilinear',
        align_corners=False
    )
    resized_masks.append(resized_mask.squeeze())
```

#### 3. Place Masks on Full Image

Create a full-resolution mask and place each ROI mask at its correct position:

```python
# Create empty mask for full image
full_mask = torch.zeros((image_height, image_width))

for i, roi in enumerate(rois):
    x1, y1, x2, y2 = roi[1:].int()

    # Place resized mask in correct position
    full_mask[y1:y2, x1:x2] = resized_masks[i]
```

#### 4. Apply Threshold and Overlay

```python
# Apply threshold to get binary mask
threshold = 0.5
binary_mask = (full_mask > threshold).float()

# Convert to numpy for visualization
mask_np = binary_mask.cpu().numpy()
image_np = np.array(image)  # Assuming image is PIL Image

# Create colored overlay
overlay = image_np.copy()
mask_color = [255, 0, 0]  # Red for target instances

# Apply mask with transparency
alpha = 0.5
overlay[mask_np > 0] = (
    alpha * np.array(mask_color) +
    (1 - alpha) * image_np[mask_np > 0]
).astype(np.uint8)
```

### Complete Example Function

```python
def overlay_masks_on_image(image, masks, rois, threshold=0.5, alpha=0.5):
    """
    Overlay segmentation masks on original image.

    Args:
        image: PIL Image or numpy array (H, W, 3)
        masks: Model output tensor [N, 3, 56, 56]
        rois: ROI coordinates tensor [N, 5]
        threshold: Confidence threshold for binary mask
        alpha: Transparency for overlay

    Returns:
        PIL Image with masks overlaid
    """
    # Convert image to numpy
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    h, w = image_np.shape[:2]

    # Get mask probabilities
    mask_probs = F.softmax(masks, dim=1)

    # Create full resolution masks for each class
    full_masks = {
        'background': torch.zeros((h, w)),
        'target': torch.zeros((h, w)),
        'non_target': torch.zeros((h, w))
    }

    # Process each ROI
    for i, roi in enumerate(rois):
        _, x1, y1, x2, y2 = roi.int()
        roi_w = x2 - x1
        roi_h = y2 - y1

        # Resize masks for this ROI
        roi_masks = F.interpolate(
            mask_probs[i].unsqueeze(0),  # [1, 3, 56, 56]
            size=(roi_h, roi_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [3, roi_h, roi_w]

        # Place in full image (using maximum for overlapping ROIs)
        full_masks['background'][y1:y2, x1:x2] = torch.maximum(
            full_masks['background'][y1:y2, x1:x2],
            roi_masks[0]
        )
        full_masks['target'][y1:y2, x1:x2] = torch.maximum(
            full_masks['target'][y1:y2, x1:x2],
            roi_masks[1]
        )
        full_masks['non_target'][y1:y2, x1:x2] = torch.maximum(
            full_masks['non_target'][y1:y2, x1:x2],
            roi_masks[2]
        )

    # Create colored overlay
    overlay = image_np.copy()

    # Apply target masks (red)
    target_mask = (full_masks['target'] > threshold).cpu().numpy()
    overlay[target_mask] = (
        alpha * np.array([255, 0, 0]) +
        (1 - alpha) * image_np[target_mask]
    ).astype(np.uint8)

    # Apply non-target masks (blue)
    non_target_mask = (full_masks['non_target'] > threshold).cpu().numpy()
    overlay[non_target_mask] = (
        alpha * np.array([0, 0, 255]) +
        (1 - alpha) * image_np[non_target_mask]
    ).astype(np.uint8)

    return Image.fromarray(overlay)
```

### Handling Multiple Overlapping ROIs

When multiple ROIs overlap, you need to decide how to combine their masks:

1. **Maximum**: Use the highest confidence value (recommended)
   ```python
   full_mask[y1:y2, x1:x2] = torch.maximum(
       full_mask[y1:y2, x1:x2],
       resized_mask
   )
   ```

2. **Average**: Average the overlapping predictions
   ```python
   full_mask[y1:y2, x1:x2] = (
       full_mask[y1:y2, x1:x2] + resized_mask
   ) / 2
   ```

3. **Priority-based**: Process ROIs in order of confidence
   ```python
   # Sort ROIs by confidence first
   roi_confidences = get_roi_confidences()  # From detection model
   sorted_indices = torch.argsort(roi_confidences, descending=True)

   # Process in order
   for idx in sorted_indices:
       # Higher confidence ROIs overwrite lower ones
       full_mask[y1:y2, x1:x2] = resized_masks[idx]
   ```

### Important Considerations

1. **Coordinate Systems**: Ensure ROI coordinates match the image coordinate system (0-indexed, exclusive end coordinates)

2. **Interpolation Mode**: Use `bilinear` with `align_corners=False` for smooth mask edges

3. **Memory Efficiency**: For large batch processing, process ROIs in chunks to avoid memory issues

4. **Edge Artifacts**: Consider applying Gaussian blur to mask edges for smoother transitions:
   ```python
   from scipy.ndimage import gaussian_filter
   smoothed_mask = gaussian_filter(mask_np, sigma=1.0)
   ```

## Project Structure

```
├── src/human_edge_detection/
│   ├── dataset.py          # COCO dataset loader
│   ├── feature_extractor.py # YOLO feature extraction
│   ├── model.py            # Segmentation model
│   ├── losses.py           # Loss functions
│   ├── train.py            # Training pipeline
│   ├── visualize.py        # Validation visualization
│   └── export_onnx.py      # ONNX export
├── data/
│   ├── annotations/        # COCO format annotations
│   └── images/            # Training/validation images
├── ext_extractor/         # YOLOv9 ONNX models
├── main.py               # Main training script
└── test_pipeline.py      # Component verification

```

## Training

### Training Features

- **Resume Training**: Continue training from any checkpoint
- **Flexible Epoch Control**: Specify additional epochs when resuming with `--resume_epochs`
- **Dynamic Progress Bars**: Automatically adjust to terminal width
- **Checkpoint Management**: Save models at specified intervals
- **Best Model Tracking**: Automatically save the best model based on validation mIoU

### Recommended Training Parameters

Based on extensive experimentation, here are the recommended parameters for different training scenarios:

#### Quick Test Run (100 images)
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd_100.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd_100.json \
--epochs 10 \
--batch_size 8 \
--lr 1e-4 \
--validate_every 1 \
--save_every 1
```

#### Small Dataset Training (500 images)
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd_500.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd_500.json \
--epochs 50 \
--batch_size 8 \
--lr 5e-4 \
--scheduler cosine \
--min_lr 1e-6 \
--validate_every 2 \
--save_every 5
```

#### Full Dataset Training (Recommended)
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--data_stats data_analyze_full.json \
--epochs 100 \
--batch_size 16 \
--lr 1e-3 \
--optimizer adamw \
--weight_decay 1e-4 \
--scheduler cosine \
--min_lr 1e-6 \
--gradient_clip 5.0 \
--num_workers 8 \
--validate_every 1 \
--save_every 1
```

#### High-Performance Training (Multi-GPU)
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--data_stats data_analyze_full.json \
--epochs 150 \
--batch_size 32 \
--lr 2e-3 \
--optimizer adamw \
--weight_decay 5e-5 \
--scheduler cosine \
--min_lr 1e-7 \
--gradient_clip 10.0 \
--num_workers 16 \
--ce_weight 1.0 \
--dice_weight 2.0 \
--validate_every 1 \
--save_every 1
```

#### Fine-tuning from Checkpoint
```bash
# Resume training from checkpoint until original epoch count
uv run python main.py \
--resume checkpoints/best_model.pth \
--epochs 50 \
--batch_size 8 \
--lr 1e-4 \
--scheduler cosine \
--min_lr 1e-7 \
--dice_weight 3.0 \
--validate_every 1 \
--save_every 1

# Train for additional epochs from checkpoint
uv run python main.py \
--resume checkpoints/checkpoint_epoch_0010_640x640_0850.pth \
--resume_epochs 20 \
--batch_size 8 \
--lr 5e-5 \
--scheduler cosine \
--min_lr 1e-7 \
--dice_weight 3.0
```

### Parameter Selection Guidelines

#### Learning Rate
- **Small dataset (100-500 images)**: 1e-4 to 5e-4
- **Medium dataset (500-5000 images)**: 5e-4 to 1e-3
- **Large dataset (5000+ images)**: 1e-3 to 2e-3
- **Fine-tuning**: 1e-5 to 1e-4

#### Batch Size
- **GPU Memory < 8GB**: 4-8
- **GPU Memory 8-16GB**: 8-16
- **GPU Memory > 16GB**: 16-32
- Larger batch sizes generally lead to more stable training

#### Optimizer Choice
- **AdamW** (default): Best for most cases, good convergence
- **Adam**: Slightly faster, but less regularization
- **SGD**: More stable but slower convergence, good for fine-tuning

#### Scheduler Strategy
- **Cosine** (recommended): Smooth decay, works well for most cases
- **Step**: Good when you know specific epochs where learning should drop
- **Exponential**: Continuous decay, good for long training
- **None**: Fixed learning rate, only for short experiments

#### Loss Weights
- **CE weight**: 1.0 (default) - Controls classification accuracy
- **Dice weight**: 1.0-3.0 - Higher values improve mask quality
  - Start with 1.0 for balanced training
  - Increase to 2.0-3.0 if masks are not sharp enough
  - Use 3.0+ for fine-tuning to refine mask boundaries

#### Gradient Clipping
- **0** (disabled): For stable datasets
- **5.0**: Recommended for most training
- **10.0**: For larger batch sizes or unstable gradients
- **1.0**: For very unstable training (rare)

#### Data Augmentation (built-in)
The dataset automatically applies:
- Random horizontal flipping
- Random slight scaling
- Color jittering (brightness, contrast, saturation)

### Memory Optimization

If you encounter CUDA out of memory errors:

1. **Reduce batch size**: Halve the batch size
2. **Enable gradient accumulation** (modify code):
   ```bash
   --batch_size 4 --gradient_accumulation 4  # Effective batch size 16
   ```
3. **Reduce number of workers**: `--num_workers 2`
4. **Use mixed precision training** (requires code modification)

### Training Monitoring

Monitor these metrics during training:
- **Training Loss**: Should decrease steadily
- **Validation mIoU**: Main metric, should increase
- **Learning Rate**: Check it's decaying as expected
- **CE vs Dice Loss**: Both should decrease, but at different rates

Use TensorBoard to monitor:
```bash
tensorboard --logdir logs
```

### Common Issues and Solutions

1. **Loss becomes NaN**
   - Reduce learning rate
   - Enable gradient clipping
   - Check for corrupted images in dataset

2. **Validation mIoU not improving**
   - Reduce learning rate
   - Increase dice weight
   - Check if model is overfitting (train loss << val loss)

3. **Training too slow**
   - Increase batch size if GPU memory allows
   - Use more workers: `--num_workers 8`
   - Ensure data is on SSD, not HDD

4. **Masks too blurry**
   - Increase dice weight: `--dice_weight 3.0`
   - Train for more epochs
   - Reduce learning rate for fine details

### Advanced Training Strategies

#### Progressive Training
Start with small dataset, then expand:
```bash
# Stage 1: 100 images, 10 epochs
uv run python main.py --epochs 10 [... other params]

# Stage 2: 500 images, resume from stage 1
uv run python main.py --resume checkpoints/best_model.pth \
  --train_ann [..._500.json] --epochs 30

# Stage 3: Full dataset
uv run python main.py --resume checkpoints/best_model.pth \
  --train_ann [..._full.json] --epochs 100
```

#### Ensemble Training
Train multiple models with different seeds:
```bash
for seed in 42 123 456; do
  uv run python main.py --seed $seed \
    --checkpoint_dir checkpoints/seed_$seed \
    [... other params]
done
```

See `CLAUDE.md` for additional training instructions and command examples.

## Validation

### Standalone Validation

You can run validation on trained checkpoints without running the full training pipeline using the `validate.py` script.

#### Validate a Single Checkpoint

```bash
# Validate best model
uv run python validate.py checkpoints/best_model.pth

# Validate specific checkpoint with custom settings
uv run python validate.py checkpoints/checkpoint_epoch_0050_640x640_0850.pth \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --data_stats data_analyze_full.json \
  --batch_size 16 \
  --num_workers 8
```

#### Validate Multiple Checkpoints

```bash
# Validate all checkpoints in directory
uv run python validate.py "checkpoints/*.pth" --multiple

# Validate checkpoints matching pattern
uv run python validate.py "checkpoints/checkpoint_epoch_00*.pth" --multiple \
  --no_visualization  # Skip visualization generation for faster validation
```

#### Command Line Arguments

- `checkpoint`: Path to checkpoint file or glob pattern (with --multiple)
- `--val_ann`: Validation annotation file (default: 100 image subset)
- `--val_img_dir`: Validation images directory (default: data/images/val2017)
- `--onnx_model`: YOLO ONNX model path
- `--data_stats`: Data statistics file for class weights
- `--batch_size`: Batch size for validation (default: 8)
- `--num_workers`: Number of data loader workers (default: 4)
- `--device`: Device to use - cuda/cpu (default: cuda)
- `--no_visualization`: Skip generating visualization images
- `--val_output_dir`: Output directory for visualizations
- `--multiple`: Enable validation of multiple checkpoints using glob pattern

#### Output

The validation script provides:
- Detailed metrics for each checkpoint (Loss, CE Loss, Dice Loss, mIoU)
- Comparison table when validating multiple checkpoints
- Optional visualization images (same as during training)
- Best checkpoint identification based on mIoU

### Validation During Training

Validation is automatically performed during training based on the `--validate_every` parameter. To run validation only without training:

```bash
# Using main.py with test_only flag
uv run python main.py --test_only --resume checkpoints/best_model.pth
```

## GPU Acceleration and TensorRT Support

### Installation

The project supports GPU acceleration through ONNX Runtime GPU and TensorRT:

```bash
# Already included in the environment
uv add onnxruntime-gpu tensorrt
```

### Performance Comparison

Based on inference tests with the segmentation model:

| Provider | Average Inference Time | Speedup |
|----------|----------------------|---------|
| CPU      | 74.57 ms            | 1.0x    |
| CUDA     | 10.07 ms            | 7.4x    |
| TensorRT (first run) | 19.6 s* | - |
| **TensorRT (cached)** | **4.04 ms** | **18.4x** |

*First run includes engine building time. Subsequent runs use cached engine.

### Usage

The system automatically detects and uses the best available provider:

1. **TensorRT** (fastest) - If available, with engine caching and FP16 optimization
2. **CUDA** (fast) - If CUDA is available
3. **CPU** (fallback) - Always available

**TensorRT Features:**
- **Engine Caching**: Engines are cached in the same directory as the ONNX model
- **FP16 Optimization**: Automatic mixed precision for better performance
- **First Run**: Takes time to build and optimize the engine
- **Subsequent Runs**: Ultra-fast inference using cached engine

### Provider Selection

You can manually specify providers when needed:

```python
# Feature extraction with specific providers
extractor = YOLOv9FeatureExtractor(
    'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
    device='cuda',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)

# ONNX Runtime session with TensorRT
import onnxruntime as ort
session = ort.InferenceSession(
    'test_model.onnx',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### Troubleshooting

If TensorRT is not working:

1. **Check GPU compatibility**: Ensure your GPU supports TensorRT
2. **CUDA version**: Make sure CUDA is properly installed
3. **Memory**: TensorRT requires additional GPU memory for optimization
4. **Fallback**: The system will automatically fall back to CUDA or CPU

## License

MIT License - See LICENSE file for details.