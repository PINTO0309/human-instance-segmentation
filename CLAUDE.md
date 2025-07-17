# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a human edge detection project that appears to be in its initial stages. The repository contains:
- COCO dataset images and annotations filtered for person instances (no crowd annotations)
- YOLOv9 ONNX models for whole body detection with feature extraction capabilities

## Repository Structure

### Data Organization
- `data/annotations/`: Contains COCO format JSON files with person-only annotations
  - Full datasets: `instances_train2017_person_only_no_crowd.json`, `instances_val2017_person_only_no_crowd.json`
  - Smaller subsets: 100 and 500 image versions for development/testing
- `data/images/`: Contains train2017 and val2017 image directories from COCO dataset
- `ext_extractor/`: Contains YOLOv9 ONNX models
  - `yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx`: Efficient model variant
  - `yolov9_n_wholebody25_Nx3x640x640_featext_optimized.onnx`: Nano model variant

## Implementation Status

The repository now contains a complete ROI-based instance segmentation training pipeline:

### Core Components
- `src/human_edge_detection/dataset.py`: COCO dataset loader with 3-class mask generation
- `src/human_edge_detection/feature_extractor.py`: YOLO ONNX feature extraction
- `src/human_edge_detection/model.py`: ROIAlign-based segmentation decoder
- `src/human_edge_detection/losses.py`: Weighted CrossEntropy + Dice loss
- `src/human_edge_detection/train.py`: Training pipeline with checkpoint saving
- `src/human_edge_detection/visualize.py`: Validation visualization generator
- `src/human_edge_detection/export_onnx.py`: ONNX export functionality

### Main Scripts
- `main.py`: Main training script
- `test_pipeline.py`: Component verification script
- `src/human_edge_detection/analyze_data.py`: Data statistics analyzer

## Common Development Tasks

### Training with minimal dataset (100 images)
```bash
uv run python main.py --epochs 10
```

### Training with different datasets
```bash
# 500 images
uv run python main.py \
  --train_ann data/annotations/instances_train2017_person_only_no_crowd_500.json \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd_500.json \
  --data_stats data_analyze.json \
  --epochs 50

# Full dataset
uv run python main.py \
  --train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --data_stats data_analyze.json \
  --epochs 100
```

### Resume training from checkpoint
```bash
uv run python main.py --resume checkpoints/checkpoint_epoch_0010_640x640_0850.pth --epochs 20
```

### Run validation only
```bash
# Using main.py
uv run python main.py --test_only --resume checkpoints/best_model.pth

# Using standalone validation script
uv run python validate.py checkpoints/best_model.pth

# Validate multiple checkpoints
uv run python validate.py "checkpoints/*.pth" --multiple --no_visualization

# Validate with custom settings
uv run python validate.py checkpoints/checkpoint_epoch_0050_640x640_0850.pth \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --data_stats data_analyze_full.json \
  --batch_size 16
```

### Export model to ONNX
```bash
uv run python -m src.human_edge_detection.export_onnx \
  --checkpoint checkpoints/best_model.pth \
  --output checkpoints/model.onnx
```

### Run data analysis
```bash
# Analyze full dataset
uv run python src/human_edge_detection/analyze_data_full.py
```

## Model Architecture

The model uses an enhanced ROIAlign-based architecture:
1. YOLOv9 extracts features (1024x80x80) from 640x640 images
2. DynamicRoIAlign extracts 28x28 features for each ROI
3. Enhanced decoder with residual blocks and multi-scale fusion produces 56x56 masks:
   - Progressive upsampling: 28x28 → 56x56 → 112x112 → 56x56
   - Residual connections for better gradient flow
   - Multi-scale feature fusion for detail preservation
4. Output: 3-class segmentation masks
   - Class 0: Background
   - Class 1: Target mask (primary instance)
   - Class 2: Non-target mask (other instances in ROI)

**Key improvements**:
- 2x larger initial ROI size (28x28) for better detail capture
- Residual blocks for deeper feature extraction
- Progressive upsampling reduces blocky artifacts
- LayerNorm2d for ONNX compatibility and stable inference

## Training Details

- Loss: Weighted CrossEntropy + Dice loss (only for target class)
- Class weights use separation-aware weights from `data_analyze_full.json` which boost non-target class weight for better instance separation
- Default weights (from full dataset analysis):
  - Background: 0.538
  - Target: 0.750
  - Non-target: 1.712 (boosted by 1.2x for instance separation)
- Checkpoints saved every epoch with format: `checkpoint_epoch_{epoch:04d}_640x640_{miou:04d}.pth`
- Validation visualization generated for specific test images every epoch
- TensorBoard logs saved in `logs/` directory

## Key Design Decisions

1. **3-class formulation**: Explicitly modeling non-target instances helps with attention and instance separation
2. **ROIAlign approach**: More efficient than full-image segmentation, focuses computation on relevant regions
3. **Separate feature extraction**: YOLO features are extracted once and reused, making training more efficient
4. **Log-scaled class weights**: More stable training compared to inverse frequency weights