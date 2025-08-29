# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Human edge detection using ROI-based lightweight instance segmentation models with hierarchical RGB architecture. The system employs knowledge distillation from larger teacher models (B7) to smaller student models (B0/B1) with temperature progression techniques.

## Key Architecture Components

### Model Hierarchy
- **B0, B1, B7 Architectures**: EfficientNet-based encoders with increasing complexity
  - B0: Most lightweight, ~71MB ONNX
  - B1: Medium complexity, ~81MB ONNX  
  - B7: Largest teacher model, ~90MB ONNX
- **Hierarchical Segmentation**: Two-stage approach
  1. Pretrained UNet for coarse binary segmentation (frozen during training)
  2. ROI-based refinement for instance segmentation
- **Knowledge Distillation**: Temperature progression (10→1) for smooth knowledge transfer

### Core Pipeline
1. **RGB Input**: Direct processing without separate feature extraction
2. **Pretrained UNet**: Generates binary foreground/background masks
3. **ROI Extraction**: Uses COCO bounding boxes
4. **Instance Segmentation**: 3-class output (background, target, non-target)
5. **Post-processing**: Optional dilation and edge smoothing

## Common Development Commands

### Training
```bash
# Minimal dataset (100 images) 
uv run python train_advanced.py --config <config_name> --epochs 10

# Full dataset training
uv run python train_advanced.py \
  --config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
  --train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --epochs 100

# Resume from checkpoint
uv run python train_advanced.py --config <config_name> --resume <checkpoint.pth> --epochs 20

# Knowledge distillation
uv run python train_distillation_staged.py \
  --teacher_config <teacher_config> \
  --student_config <student_config> \
  --teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth
```

### Validation
```bash
# Single checkpoint
uv run python validate_advanced.py <checkpoint.pth>

# Multiple checkpoints
uv run python validate_advanced.py "experiments/*/checkpoints/best_model*.pth" --multiple

# With custom settings
uv run python validate_advanced.py <checkpoint.pth> \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --batch_size 16
```

### ONNX Export
```bash
# Export hierarchical model (auto-detects architecture)
uv run python export_hierarchical_instance_peopleseg_onnx.py \
  <checkpoint.pth> \
  --output <output.onnx> \
  --image_size 640,640 \
  --dilation_pixels 0

# Export with 2-pixel dilation
uv run python export_hierarchical_instance_peopleseg_onnx.py \
  <checkpoint.pth> \
  --output <output_dil2.onnx> \
  --dilation_pixels 2

# Test ONNX inference
uv run python test_hierarchical_instance_peopleseg_onnx.py \
  --onnx <model.onnx> \
  --annotations <annotations.json> \
  --num_images 5 \
  --binary_mode  # Optional: visualize binary masks as green overlay
```

### Linting and Testing
```bash
# Run linting
uv run ruff check .

# Run type checking  
uv run python -m mypy src/

# Test pipeline components
uv run python test_pipeline.py
```

## Configuration System

Configurations in `src/human_edge_detection/experiments/`:
- Use `ConfigManager.list_configs()` to list available configs
- Use `ConfigManager.get_config(config_name)` to load config
- Config naming pattern: `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r{roi}m{mask}_disttrans_contdet_baware_from_{arch}`

Key configuration parameters:
- ROI sizes: r64x48 (B0), r80x60 (B1), r128x96 (B7)
- Mask sizes: m128x96 (B0), m160x120 (B1), m256x192 (B7)
- Loss weights, learning rates, augmentation settings

## Dataset Structure

- **Annotations**: `data/annotations/` - COCO format, person-only, no crowds
  - Full: `instances_{train|val}2017_person_only_no_crowd.json`
  - Subsets: 100, 500 image versions for development
- **Images**: `data/images/{train|val}2017/` - COCO dataset images
- **Pretrained Models**: `ext_extractor/best_model_{b0|b1|b7}_*.pth`

## Model I/O Specifications

### Training Input
- **Images**: `[B, 3, H, W]` - RGB in [0, 1] range
- **ROIs**: `[N, 5]` - `[batch_idx, x1, y1, x2, y2]` normalized coordinates

### ONNX Model
- **Inputs**:
  - `images`: `[B, 3, H, W]` - RGB images
  - `rois`: `[N, 5]` - ROI coordinates
- **Outputs**:
  - `masks`: `[N, 3, mask_h, mask_w]` - Segmentation logits
  - `binary_masks`: `[B, 1, H, W]` - Binary foreground masks

## Key Technical Details

- **3-Class Segmentation**: Background, target instance, non-target instances
- **ImageNet Normalization**: Applied for pretrained encoders during distillation
- **Temperature Progression**: Gradual reduction (10→1) during distillation training
- **Loss Functions**: Weighted CrossEntropy + Dice with separation-aware weights
- **Class Weights** (from full dataset):
  - Background: 0.538
  - Target: 0.750  
  - Non-target: 1.712 (1.2x boosted for better separation)
- **Checkpoint Format**: `checkpoint_epoch_{epoch:04d}_640x640_{miou:04d}.pth`
- **Primary Metric**: mIoU (mean Intersection over Union)