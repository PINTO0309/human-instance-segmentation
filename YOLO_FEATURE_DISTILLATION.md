# YOLOv9 Feature Distillation Pipeline

## Overview

This document describes the knowledge distillation pipeline that combines YOLO feature alignment with UNet teacher-student distillation for human segmentation.

## Architecture Overview

### Training Pipeline

```
Input Image (640×640, ImageNet normalized)
        ├─────────────────┬─────────────────┐
        │                 │                 │
        v                 v                 v
   Denormalize      Teacher UNet B3    Student UNet B0
   to [0,1]         (frozen)           (trainable decoder)
        │                 │                 │
        v                 │                 v
   YOLOv9 ONNX           │            Feature@80×80
   (frozen)              │            (40 channels)
        │                 │                 │
        v                 v                 v
  Features@80×80    Segmentation      Projection Layer
  (1024 channels)   Output             (40→768→1024)
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                          v
                    Loss Computation
```

### Inference Pipeline

```
Input Image (640×640)
        │
        v
  Student UNet B0
  (without projection)
        │
        v
  Segmentation Output
```

## Components

### 1. Teacher Model: UNet with EfficientNet-B3

- **Architecture**: Standard UNet with EfficientNet-B3 encoder
- **Weights**: Pre-trained on human segmentation task
- **Role**: Provides segmentation supervision
- **State**: Frozen during training
- **Input**: 640×640 images (ImageNet normalized)
- **Output**: Binary segmentation mask (1×640×640)

### 2. Student Model: UNet with EfficientNet-B0

- **Architecture**: Lightweight UNet with EfficientNet-B0 encoder
- **Encoder**: Frozen (ImageNet pre-trained)
- **Decoder**: Trainable
- **Feature extraction point**: Layer 3 (40×80×80)
- **Input**: 640×640 images (ImageNet normalized)
- **Output**: Binary segmentation mask (1×640×640)

### 3. YOLOv9 Feature Extractor

- **Model**: YOLOv9-E whole body detector
- **Format**: ONNX optimized
- **Target layer**: `segmentation_model_34_Concat_output_0`
- **Output shape**: 1024×80×80
- **Input normalization**: Simple [0,1] range (no ImageNet normalization)
- **Role**: Provides rich human-aware features for distillation

### 4. Feature Projection Layer

- **Architecture**: 
  ```
  Conv2d(40, 768, kernel_size=1)
  BatchNorm2d(768)
  ReLU()
  Conv2d(768, 1024, kernel_size=1)
  ```
- **Purpose**: Aligns student features with YOLO features
- **Training**: Included in optimization
- **Inference**: Removed (not needed)

## Loss Functions

### 1. Output Distillation Losses

#### KL Divergence Loss
- Measures distribution difference between teacher and student outputs
- Formula for binary segmentation:
  ```
  KL = p * log(p/q) + (1-p) * log((1-p)/(1-q))
  where p = teacher probability, q = student probability
  ```
- Temperature scaling: T=3.0
- Weight: 1.0

#### MSE Loss
- Direct output matching between teacher and student
- Weight: 0.5

#### BCE Loss
- Binary cross-entropy with ground truth
- Weight: 0.5

#### Dice Loss
- Overlap measurement with ground truth
- Weight: 1.0

### 2. Feature Alignment Loss

- **Type**: MSE between YOLO features and projected student features
- **Resolution**: 80×80
- **Channels**: 1024
- **Weight**: 0.5 (configurable)

### Total Loss
```python
total_loss = kl_weight * kl_loss + 
             mse_weight * mse_loss + 
             bce_weight * bce_loss + 
             dice_weight * dice_loss + 
             feature_weight * feature_loss
```

## Data Flow

### Training Phase

1. **Input Processing**
   - Load image (640×640)
   - Apply augmentations (if enabled)
   - Normalize with ImageNet statistics

2. **Feature Extraction**
   - Denormalize for YOLO (remove ImageNet normalization)
   - Extract YOLO features (1024×80×80)
   - Extract student features at layer 3 (40×80×80)
   - Project student features to 1024 channels

3. **Forward Pass**
   - Teacher generates segmentation mask
   - Student generates segmentation mask
   - Compute all loss components

4. **Backward Pass**
   - Update only student decoder parameters
   - Update projection layer parameters
   - Keep teacher and encoder frozen

### Inference Phase

1. **Simplified Pipeline**
   - Input image → Student UNet B0 → Segmentation
   - No YOLO model needed
   - No projection layer needed
   - No teacher model needed

2. **Performance Benefits**
   - Reduced model size (B0 vs B3)
   - Faster inference
   - Lower memory usage

## Configuration

### Model Configuration
```python
{
  "student_encoder": "timm-efficientnet-b0",
  "teacher_checkpoint": "ext_extractor/2020-09-23a.pth",
  "yolo_onnx_path": "ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx",
  "yolo_target_layer": "segmentation_model_34_Concat_output_0",
  "projection_hidden_dim": 768
}
```

### Training Configuration
```python
{
  "learning_rate": 1e-4,
  "batch_size": 4,
  "epochs": 50,
  "optimizer": "AdamW",
  "scheduler": "cosine",
  "temperature": 3.0
}
```

### Loss Weights
```python
{
  "kl_weight": 1.0,
  "mse_weight": 0.5,
  "bce_weight": 0.5,
  "dice_weight": 1.0,
  "feature_weight": 0.5
}
```

## Key Design Decisions

### 1. Feature Resolution Matching
- Student features extracted at 80×80 resolution (layer 3)
- Matches YOLO feature resolution exactly
- Avoids upsampling/downsampling artifacts

### 2. Normalization Handling
- Models use different normalizations:
  - UNets: ImageNet normalization
  - YOLO: Simple [0,1] normalization
- Denormalization applied before YOLO inference

### 3. Projection Layer Design
- Two-stage projection with hidden dimension
- BatchNorm for training stability
- Removed during inference for efficiency

### 4. Teacher-Student Architecture
- Teacher: B3 (larger, more accurate)
- Student: B0 (smaller, faster)
- Balance between accuracy and efficiency

## Training Command

```bash
# Full dataset training
uv run python train_yolo_feature_distillation.py \
  --config rgb_unet_yolo_feature_distillation_b0_from_b3 \
  --epochs 50 \
  --batch_size 4 \
  --feature_weight 0.5

# Quick test with small dataset
uv run python train_yolo_feature_distillation.py \
  --config rgb_unet_yolo_feature_distillation_b0_from_b3 \
  --epochs 1 \
  --batch_size 2
```

## Expected Results

### Metrics
- Student mIoU: ~0.65-0.70 (approaching teacher's 0.89)
- Agreement with teacher: >85%
- Feature loss: <1.0 after convergence
- KL divergence: 0.1-1.0 range

### Model Size
- Teacher (B3): ~40MB
- Student (B0): ~20MB
- Reduction: ~50%

### Inference Speed
- Teacher: ~50ms/image
- Student: ~20ms/image
- Speedup: ~2.5x

## Advantages

1. **Knowledge Transfer**: Student learns from both teacher outputs and YOLO's human-specific features
2. **Efficiency**: Inference requires only the lightweight student model
3. **Accuracy**: YOLO features provide additional human detection knowledge
4. **Flexibility**: Can adjust feature weight to balance distillation sources

## Limitations

1. **Training Complexity**: Requires managing three models during training
2. **Memory Usage**: Training requires loading all three models
3. **Feature Alignment**: Fixed projection architecture may not be optimal for all cases

## Temperature Scheduling

Progressive distillation is supported, allowing temperature to gradually decrease during training:

### Configuration
Add these parameters to `feature_match_layers` in config_manager.py:
```python
feature_match_layers=[
    "segmentation_model_34_Concat_output_0",  # YOLO target layer
    "ext_extractor/yolov9_e_wholebody25.onnx",  # ONNX path
    "mse",    # Feature loss type
    "0.5",    # Feature loss weight
    "768",    # Projection hidden dimension
    "true",   # Enable temperature scheduling
    "3.0",    # Initial temperature
    "1.0",    # Final temperature
    "cosine", # Schedule type (linear/cosine/exponential)
]
```

### Schedule Types
- **linear**: Linear interpolation from initial to final temperature
- **cosine**: Cosine annealing for smooth decay curve
- **exponential**: Exponential decay (rapid initial decrease, then gradual)

### Benefits
- **Early training**: Learn from soft targets with high temperature
- **Late training**: Approach hard targets with low temperature
- **Result**: More stable convergence and improved final accuracy

## Future Improvements

1. **Dynamic Feature Selection**: Learn which YOLO layers to use
2. **Attention Mechanisms**: Weight important spatial regions
3. **Multi-Scale Features**: Use multiple YOLO layers at different resolutions
4. **Feature Adaptation**: Learn task-specific transformations of YOLO features