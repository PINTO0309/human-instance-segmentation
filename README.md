# Human Instance Segmentation

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PINTO0309/human-instance-segmentation)

English / [日本語](./README_jp.md)

A lightweight ROI-based hierarchical instance segmentation model for human detection with knowledge distillation from EfficientNet-based teacher models. The model achieves efficient real-time performance through a two-stage hierarchical architecture and temperature progression distillation techniques.

- Instance Segmentation Mode

  <img width="640" height="424" alt="image" src="https://github.com/user-attachments/assets/10431a3c-0bba-422f-9d98-67af8e77b777" />

- Binary Mask Mode

  <img width="427" height="640" alt="000000229849_binary" src="https://github.com/user-attachments/assets/f73ef1b2-36d8-4eb0-b167-06764e05ebe3" />

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Architecture Details](#architecture-details)
- [Architecture Diagram](#architecture-diagram)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Refinement Mechanism](#refinement-mechanism)
- [Dataset Structure](#dataset-structure)
- [Environment Setup](#environment-setup)
- [UNet Distillation Commands](#unet-distillation-commands)
- [ROI-Based Hierarchical Training](#roi-based-hierarchical-training)
- [ONNX Export](#onnx-export)
- [Test Inference](#test-inference)
- [License](#license)
- [Citations and Acknowledgments](#citations-and-acknowledgments)

## Architecture Overview

The Human Instance Segmentation model employs a sophisticated hierarchical segmentation approach that combines:
- **Two-Stage Architecture**: Coarse binary segmentation followed by ROI-based instance refinement
- **Multi-Architecture Support**: B0 (lightweight), B1 (balanced), and B7 (high-accuracy) variants
- **Knowledge Distillation**: Temperature progression (10→1) for efficient knowledge transfer
- **Real-time Processing**: Optimized for edge devices with ONNX/TensorRT deployment

### Key Features
- Direct RGB input processing without separate feature extraction
- Pre-trained UNet for robust binary foreground/background segmentation
- ROI-based refinement for precise instance separation
- 3-class output system (background, target instance, non-target instances)
- Optional post-processing with dilation and edge smoothing

## Architecture Details

### Model Hierarchy

#### B0 Architecture (Lightweight)
- **Encoder**: EfficientNet-B0 based (timm-efficientnet-b0)
- **Parameters**: ~5.3M
- **ONNX Size**: ~71MB
- **ROI Size**: 64×48 (standard), 80×60 (enhanced)
- **Mask Size**: 128×96 (standard), 160×120 (enhanced)
- **Use Case**: Real-time edge deployment, mobile devices

#### B1 Architecture (Balanced)
- **Encoder**: EfficientNet-B1 based (timm-efficientnet-b1)
- **Parameters**: ~7.8M
- **ONNX Size**: ~81MB
- **ROI Size**: 64×48 (standard), 80×60 (enhanced)
- **Mask Size**: 128×96 (standard), 160×120 (enhanced)
- **Use Case**: Balanced performance/accuracy trade-off

#### B7 Architecture (High-Accuracy)
- **Encoder**: EfficientNet-B7 based (timm-efficientnet-b7)
- **Parameters**: ~66M
- **ONNX Size**: ~90MB
- **ROI Size**: 64×48 (standard), 80×60 (enhanced), 128×96 (ultra)
- **Mask Size**: 128×96 (standard), 160×120 (enhanced), 256×192 (ultra)
- **Use Case**: Maximum accuracy, server deployment

### Core Components

#### 1. Pretrained UNet Module
- **Architecture**: Enhanced UNet with residual blocks
- **Normalization**: LayerNorm2D for stable training
- **Activation**: ReLU/SiLU configurable
- **Output**: Binary foreground/background mask
- **Training**: Frozen during instance segmentation training

#### 2. ROI Extraction Module
- **Input**: COCO bounding boxes
- **Normalization**: Coordinates normalized to [0, 1]
- **Pooling**: Dynamic RoI Align with configurable output sizes
- **Batch Processing**: Efficient multi-instance handling

#### 3. Instance Segmentation Head
- **Architecture**: Hierarchical UNet V2 with attention modules
- **Classes**: 3-class segmentation (background, target, non-target)
- **Features**:
  - Residual blocks for feature refinement
  - Attention gating for focus on person boundaries
  - Distance-aware loss for better instance separation
  - Contour detection auxiliary task

#### 4. Loss Functions
- **Primary Loss**: Weighted CrossEntropy + Dice Loss
- **Class Weights**:
  - Background: 0.538
  - Target: 0.750
  - Non-target: 1.712 (1.2× boosted)
- **Auxiliary Losses**:
  - Distance transform loss for boundary awareness
  - Contour detection loss for edge refinement
  - Separation-aware weighting for instance distinction

## Architecture Diagram

```
             ┌─────────────────────────────┐           ┌──────────────────────────────┐
             │       Input RGB Image       │           │             ROIs             │
             │        [B, 3, H, W]         │           │            [N, 5]            │
             └──────────────┬──────────────┘           │ [batch_idx, x1, y1, x2, y2]  │
                            │                          │ (0-1 normalized coordinates) │
                            │                          └──────────────┬───────────────┘
                            │                                         │
             ┌──────────────▼──────────────┐                          │
             │   Pretrained UNet Module    │                          │
             │    (Frozen during training) │                          │
             │   Output: Binary FG/BG      │                          │
             └──────────────┬──────────────┘                          │
                            │                                         │
              ┌─────────────┴─────────────┐                           │
              │                           │                           │
  ┌───────────▼───────────┐   ┌───────────▼──────────┐                │
  │  Binary Mask Output   │   │   Feature Maps       │                │
  │   [B, 1, H, W]        │   │   for ROI Pooling    │                │
  └───────────┬───────────┘   └───────────┬──────────┘                │
              │                           │                           │
              └─────────────┬─────────────┘                           │
                            │◀────────────────────────────────────────┘
            ┌───────────────▼───────────────┐
            │   Dynamic RoI Align           │
            │  Output: [N, C, H_roi, W_roi] │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
 ┌────────────▼───────────┐   ┌───────────▼────────────┐
 │      EfficientNet      │   │  Pretrained UNet Mask  │
 │      Encoder           │   │  (for each ROI)        │
 │      (B0/B1/B7)        │   │  [N, 1, H_roi, W_roi]  │
 └────────────┬───────────┘   └───────────┬────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Instance Segmentation    │
              │  Head (UNet V2)           │
              │  - Attention Modules      │
              │  - Residual Blocks        │
              │  - Distance-Aware Loss    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   3-Class Output Logits   │
              │   [N, 3, mask_h, mask_w]  │
              │   Classes:                │
              │   0: Background           │
              │   1: Target Instance      │
              │   2: Non-target Instances │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Post-Processing         │
              │   (Optional)              │
              │   - Mask Dilation         │
              │   - Edge Smoothing        │
              └───────────────────────────┘
```

## Model Architecture

<details><summary>Click to expand</summary>

<img width="481" height="450" alt="image" src="https://github.com/user-attachments/assets/300efc73-8652-4e80-b75e-3771b102e39d" />

<img alt="best_model_b1_80x60_0 8551" src="https://github.com/user-attachments/assets/4f9a8a19-a81a-4b10-a72a-cdb9e099151d" />

</details>

## Training Pipeline

### Knowledge Distillation Pipeline
1. **Teacher Model Training**: Train B7 architecture to high accuracy
2. **Temperature Progression**: Gradual temperature reduction (10→1)
3. **Student Training**: Distill to B0/B1 with feature and logit matching
4. **Fine-tuning**: Optional direct training on target dataset

### Training Stages
1. **Stage 1: UNet Pre-training**
   - Binary person segmentation on COCO dataset
   - Frozen after pre-training for all subsequent stages

2. **Stage 2: Knowledge Distillation**
   - Teacher model (B7) provides soft targets
   - Temperature progression for smooth knowledge transfer
   - Feature matching at multiple decoder levels

      <img width="868" height="552" alt="image" src="https://github.com/user-attachments/assets/1204d50b-d034-427a-9238-b0e604eb33d8" />

3. **Stage 3: Instance Segmentation Training**
   - ROI-based training with 3-class outputs
   - Distance-aware loss for instance separation
   - Auxiliary tasks for boundary refinement

## Refinement Mechanism

### Hierarchical Refinement Process
1. **Coarse Segmentation**: Pretrained UNet provides initial binary mask
2. **ROI Extraction**: Extract regions around detected persons
3. **Feature Enhancement**: Process ROIs through EfficientNet encoder
4. **Instance Refinement**:
   - Apply attention-gated refinement
   - Use binary mask as prior for background suppression
   - Separate overlapping instances via distance transform

### Key Refinement Techniques
- **Attention Gating**: Focus processing on person boundaries
- **Distance Transform**: Encode spatial relationships for better separation
- **Contour Detection**: Auxiliary task for edge preservation
- **Separation-Aware Weighting**: Boost non-target class for clearer boundaries

## Dataset Structure

### Directory Layout
```
data/
├── annotations/
│   ├── instances_train2017_person_only_no_crowd.json  # Full training set
│   ├── instances_val2017_person_only_no_crowd.json    # Full validation set
│   ├── instances_train2017_person_only_no_crowd_100imgs.json  # Dev subset
│   └── instances_val2017_person_only_no_crowd_100imgs.json    # Dev subset
├── images/
│   ├── train2017/  # COCO training images
│   └── val2017/    # COCO validation images
└── pretrained/
    ├── best_model_b0_*.pth  # Pretrained B0 models
    ├── best_model_b1_*.pth  # Pretrained B1 models
    └── best_model_b7_*.pth  # Pretrained B7 models
```

### Annotation Format
- **Format**: COCO JSON format
- **Categories**: Person only (no crowd annotations)
- **Content**: Bounding boxes and segmentation polygons
- **Filtering**: Crowd instances removed for cleaner training

### Dataset Statistics
- **Full Dataset**: ~64K training, ~2.7K validation images
- **Development Subsets**: 100, 500 image versions
- **Class Distribution**:
  - Background: ~53.8% pixels
  - Target instances: ~33.3% pixels
  - Non-target instances: ~12.9% pixels

## Environment Setup

### Prerequisites
- Python 3.10
- CUDA 11.8+ (for GPU support)
- uv package manager

### Installation with uv

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv

# Activate environment
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r pyproject.toml

# Install development dependencies (optional)
uv pip install -e ".[dev]"
```

### Verify Installation
```bash
# Check PyTorch and CUDA
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check ONNX Runtime
uv run python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}')"
```

## UNet Distillation Commands

### Distillation Configuration Files
- `rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog`: B7→B0 distillation
- `rgb_hierarchical_unet_v2_distillation_b1_from_b7_temp_prog`: B7→B1 distillation
- `rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog`: B7 self-distillation

### Basic Distillation Training
```bash
# B7 to B0 distillation with temperature progression
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--epochs 100 \
--batch_size 16

# B7 to B1 distillation
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b1_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--epochs 100 \
--batch_size 12
```

### Advanced Distillation Options
```bash
# With custom temperature schedule
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--initial_temperature 10.0 \
--final_temperature 1.0 \
--temperature_decay_epochs 50 \
--epochs 100

# Resume from checkpoint
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--resume checkpoints/distillation_epoch_050.pth \
--epochs 100
```

## ROI-Based Hierarchical Training

### Standard Configuration Files
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B0`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B1`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B7`

### Enhanced Configuration Files
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B0_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B1_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B7_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r128x96m256x192_disttrans_contdet_baware_from_B7_enhanced`

### Basic Training Commands

```bash
# Train B0 model with standard ROI size (development dataset)
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--epochs 10 \
--batch_size 8

# Train B1 model with enhanced ROI size (full dataset)
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1_enhanced \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--epochs 100 \
--batch_size 6

# Train B7 model with ultra ROI size
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r128x96m256x192_disttrans_contdet_baware_from_B7_enhanced \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--epochs 100 \
--batch_size 4
```

### Advanced Training Options

```bash
# Resume training from checkpoint
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--resume experiments/*/checkpoints/checkpoint_epoch_0050_640x640_0750.pth \
--epochs 100

# Fine-tuning with smaller learning rate
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--pretrained_checkpoint experiments/*/checkpoints/best_model_*.pth \
--learning_rate 1e-5 \
--epochs 20
```

### Validation Commands

```bash
# Validate single checkpoint
uv run python validate_advanced.py \
experiments/*/checkpoints/best_model_epoch_*_640x640_*.pth \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--batch_size 16

# Validate multiple checkpoints
uv run python validate_advanced.py \
"experiments/*/checkpoints/best_model*.pth" \
--multiple \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json

# Validation with visualization
uv run python validate_advanced.py \
experiments/*/checkpoints/best_model_*.pth \
--visualize \
--num_visualize 20 \
--output_dir validation_results
```

## ONNX Export

### Export Scripts
- `export_peopleseg_onnx.py`: Export pretrained UNet models
- `export_hierarchical_instance_peopleseg_onnx.py`: Export full hierarchical models
- `export_bilateral_filter.py`: Export bilateral filter post-processing
- `export_edge_smoothing_onnx.py`: Export edge smoothing modules

### Pre-trained weights
https://github.com/PINTO0309/human-instance-segmentation/releases/tag/weights

<img width="747" height="442" alt="image" src="https://github.com/user-attachments/assets/2278a435-355d-4cd3-b307-11b6fcd6b3e4" />

### Basic Export Commands

```bash
# Export B0 model to ONNX
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b0_*.pth \
--output models/b0_model.onnx \
--image_size 640,640

# Export B1 model with 1-pixel dilation
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b1_*.pth \
--output models/b1_model_dil2.onnx \
--image_size 640,640 \
--dilation_pixels 1

# Export B7 model with custom ROI size
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b7_*.pth \
--output models/b7_model_ultra.onnx \
--image_size 1024,1024
```

### Export Post-Processing Modules

```bash
# Export edge smoothing module
uv run python export_edge_smoothing_onnx.py
```
<img width="843" height="777" alt="image" src="https://github.com/user-attachments/assets/fea811f4-2aa5-436d-9845-c714b511831f" />

```bash
# Export bilateral filter
uv run python export_bilateral_filter.py
```
<img width="1245" height="1297" alt="image" src="https://github.com/user-attachments/assets/c49c82e7-08ba-49be-abb1-17dddefb16b4" />

### ONNX Optimization

```bash
# Optimize ONNX model with onnxsim
uv run python -m onnxsim models/b0_model.onnx models/b0_model_opt.onnx

# Verify optimized model
uv run python -c "import onnx; model = onnx.load('models/b0_model_opt.onnx'); onnx.checker.check_model(model); print('Model is valid')"
```

## Test Inference

### Test Script: `test_hierarchical_instance_peopleseg_onnx.py`

### Basic Testing

```bash
# Test ONNX model on validation images
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd_100imgs.json \
--images_dir data/images/val2017 \
--num_images 5 \
--output_dir test_outputs

# Test with CUDA provider
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b1_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--provider cuda \
--num_images 10 \
--output_dir test_outputs_cuda
```

### Advanced Testing Options

```bash
# Test with binary mask visualization (green overlay)
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 20 \
--binary_mode \
--alpha 0.7 \
--output_dir test_binary_masks

# Test with custom score threshold
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b7_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 15 \
--score_threshold 0.5 \
--save_masks \
--output_dir test_high_confidence

# Batch processing test
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 100 \
--batch_size 8 \
--output_dir batch_test_outputs
```

### Performance Benchmarking

```bash
# Benchmark inference speed
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 50 \
--provider cuda

# Compare different model variants
for model in b0 b1 b7; do
  echo "Testing $model model..."
  uv run python test_hierarchical_instance_peopleseg_onnx.py \
    --onnx models/${model}_model_opt.onnx \
    --annotations data/annotations/instances_val2017_person_only_no_crowd_100imgs.json \
    --num_images 20 \
    --output_dir benchmark_${model}
done
```

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Katsuya Hyodo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Citations and Acknowledgments

This project builds upon several excellent works in the computer vision community:

### People Segmentation
We gratefully acknowledge the work by Vladimir Iglovikov (Ternaus) on people segmentation:
- Repository: [https://github.com/ternaus/people_segmentation](https://github.com/ternaus/people_segmentation)
- Paper: "TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation"

### EfficientNet
```bibtex
@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

### COCO Dataset
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### U-Net Architecture
```bibtex
@inproceedings{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### Knowledge Distillation
```bibtex
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

### Special Thanks
- The PyTorch team for the excellent deep learning framework
- The ONNX community for cross-platform model deployment tools
- The Albumentations team for powerful augmentation pipelines
- The Segmentation Models PyTorch contributors for pre-trained encoders
