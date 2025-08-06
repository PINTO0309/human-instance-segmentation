# Knowledge Distillation Implementation

## Overview

This implementation provides knowledge distillation from a teacher model (EfficientNet-B3) to a student model (EfficientNet-B0) for hierarchical segmentation tasks. The `run_experiments.py` script fully supports knowledge distillation training, allowing you to train student models with teacher guidance while leveraging all experiment management features.

## Key Components

### 1. Student Model Architecture
- **Encoder**: `timm-efficientnet-b0` (smaller, faster)
- **Parameters**: ~6.25M (2.1x compression from teacher's 13.16M parameters)
- **Architecture**: Same UNet structure as teacher, different encoder

### 2. Teacher Model
- **Encoder**: `timm-efficientnet-b3` (larger, more accurate)
- **Checkpoint**: `experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware/checkpoints/checkpoint_epoch_0048.pth`
- **Frozen**: Teacher weights are not updated during training

### 3. Distillation Configuration

```python
distillation:
    enabled: True
    teacher_checkpoint: "path/to/teacher/checkpoint.pth"
    temperature: 4.0  # Softens probability distributions
    alpha: 0.7  # 70% distillation loss, 30% ground truth loss
    distill_logits: True  # Distill output predictions
    freeze_teacher: True  # Don't update teacher weights
    student_encoder: "timm-efficientnet-b0"
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--teacher_checkpoint` | Path to teacher model checkpoint | None |
| `--distillation_temperature` | Temperature for softening distributions | 4.0 (from config) |
| `--distillation_alpha` | Weight for distillation loss (0-1) | 0.7 (from config) |
| `--epochs` | Number of training epochs | 100 |
| `--batch_size` | Batch size for training | 2 (from config) |
| `--config` | Base configuration to use | `rgb_hierarchical_unet_v2_distillation_b0_from_b3` |
| `--resume` | Resume from checkpoint | None |

## Training Examples

### Using train_advanced.py

```bash
# Specify custom teacher checkpoint
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint path/to/your/teacher.pth \
--batch_size 2 \
--epochs 20

# Override all distillation parameters
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint path/to/teacher.pth \
--distillation_temperature 5.0 \
--distillation_alpha 0.6 \
--batch_size 16 \
--epochs 50

# Enable distillation on any configuration
uv run python train_advanced.py \
--config baseline \
--teacher_checkpoint path/to/teacher.pth \
--epochs 100
```

### Using run_experiments.py

```bash
# Run distillation with default configuration
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware/checkpoints/checkpoint_epoch_0048.pth \
--batch_size 2 \
--epochs 20

# Override temperature and alpha
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint path/to/teacher.pth \
--distillation_temperature 5.0 \
--distillation_alpha 0.6 \
--epochs 50 \
--batch_size 4

# Use distillation with baseline configuration
uv run python run_experiments.py \
--configs baseline \
--teacher_checkpoint experiments/baseline/checkpoints/best_model.pth \
--distillation_temperature 4.0 \
--distillation_alpha 0.7 \
--epochs 100
```

### Compare Multiple Configurations

```bash
# Compare teacher, student without distillation, and student with distillation
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware/checkpoints/checkpoint_epoch_0048.pth \
--epochs 50
```

### Resume Training

```bash
# Resume using train_advanced.py
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--resume experiments/rgb_hierarchical_unet_v2_distillation_b0_from_b3/checkpoints/checkpoint_epoch_0050.pth \
--teacher_checkpoint path/to/teacher.pth \
--epochs 100

# Resume using run_experiments.py
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--resume experiments/rgb_hierarchical_unet_v2_distillation_b0_from_b3/checkpoints/checkpoint_epoch_0025.pth \
--teacher_checkpoint path/to/teacher.pth \
--total_epochs 100
```

### Export ONNX Models

```bash
# Export trained distilled models to ONNX
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--export_onnx \
--skip_training
```

### Using the Training Script

```bash
# Basic usage with defaults
./run_distillation_training.sh

# With custom teacher checkpoint
./run_distillation_training.sh \
--teacher_checkpoint experiments/your_teacher/checkpoints/best_model.pth

# With all parameters
./run_distillation_training.sh \
--teacher_checkpoint path/to/teacher.pth \
--epochs 50 \
--temperature 5.0 \
--alpha 0.6 \
--config rgb_hierarchical_unet_v2_distillation_b0_from_b3
```

## Loss Function

The distillation loss combines:
1. **Base Loss** (30%): Standard segmentation loss (CE + Dice) against ground truth
2. **Distillation Loss** (70%):
   - Main logits KL divergence
   - Background/foreground auxiliary predictions
   - Target/non-target auxiliary predictions

Formula: `L_total = α * L_distill + (1-α) * L_base`

## Experiment Output Structure

When running distillation experiments, the following structure is created:

```
experiments/
└── rgb_hierarchical_unet_v2_distillation_b0_from_b3/
    ├── checkpoints/
    │   ├── untrained_model.pth      # Initial student model
    │   ├── checkpoint_epoch_*.pth   # Training checkpoints
    │   ├── best_model.pth          # Best student model
    │   └── best_model.onnx         # Exported ONNX model
    ├── logs/
    │   ├── events.out.tfevents.*   # TensorBoard logs
    │   └── training_log.txt        # Text log
    ├── visualizations/
    │   └── epoch_*/                # Validation visualizations
    └── config.json                 # Experiment configuration
```

## Monitoring Training

### TensorBoard

```bash
# Monitor training progress
tensorboard --logdir experiments/rgb_hierarchical_unet_v2_distillation_b0_from_b3/logs
```

### Metrics Comparison

After training multiple configurations:

```bash
# View comparison results
cat experiment_runs.json
cat experiment_comparison.csv
```

## Implementation Files

- **`src/human_edge_detection/advanced/knowledge_distillation.py`**: Core distillation logic
- **`src/human_edge_detection/advanced/hierarchical_segmentation_unet.py`**: Updated to support different encoders
- **`src/human_edge_detection/advanced/hierarchical_segmentation_rgb.py`**: Passes encoder name to UNet models
- **`train_advanced.py`**: Training logic with distillation support
- **`run_experiments.py`**: Experiment management with full distillation support
- **`src/human_edge_detection/experiments/config_manager.py`**: Distillation configuration

## Validation & Export

- **Validation**: Uses student model for evaluation metrics
- **Visualization**: Student model predictions are visualized
- **ONNX Export**: Exports student model only
- **Checkpoints**: Save student model state dict

## Key Features

1. **Flexible Teacher Loading**: Specify teacher checkpoint via command-line or config
2. **Frozen Teacher**: Teacher weights remain fixed during training
3. **Compatible Architecture**: Student uses same architecture with smaller encoder
4. **Full Pipeline Support**: Works with validation, visualization, and ONNX export
5. **Resume Support**: Can resume training from checkpoints
6. **Runtime Configuration**: Override distillation parameters at runtime

## Performance

- **Model Size**: 2.1x compression (13.16M → 6.25M parameters)
- **Expected Accuracy**: ~95% of teacher performance with proper training
- **Inference Speed**: ~2x faster than teacher model

## Best Practices

1. **Teacher Selection**: Use a well-trained teacher model (typically with high mIoU)
2. **Temperature Tuning**: Start with T=4.0, increase for softer distributions
3. **Alpha Balance**: Use α=0.7 (70% distillation) as starting point
4. **Batch Size**: Use same or larger batch size as teacher training
5. **Learning Rate**: Often lower than training from scratch (e.g., 1e-4)
6. **Epochs**: Student typically needs fewer epochs than teacher

## Troubleshooting

### Teacher Checkpoint Not Found

Ensure the path is correct and the file exists:
```bash
ls -la experiments/*/checkpoints/*.pth
```

### Memory Issues

Reduce batch size or use gradient accumulation:
```bash
uv run python run_experiments.py \
--configs rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
--teacher_checkpoint path/to/teacher.pth \
--batch_size 2 \
--epochs 100
```

### Slow Convergence

Adjust distillation parameters:
```bash
# Increase temperature for softer targets
--distillation_temperature 6.0

# Increase distillation weight
--distillation_alpha 0.8
```

## Testing

Verify distillation support:
```bash
uv run python test_run_experiments_simple.py
```