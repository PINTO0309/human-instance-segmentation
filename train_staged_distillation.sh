#!/bin/bash

# UNet encoder-only distillation training script
# This script trains a UNet with B0 encoder using knowledge distillation from B3 teacher

echo "=========================================="
echo "UNet Encoder-Only Distillation Training"
echo "=========================================="
echo ""
echo "Teacher: timm-efficientnet-b3 (ext_extractor/2020-09-23a.pth)"
echo "Student: timm-efficientnet-b0"
echo ""

# Default configuration
CONFIG="rgb_hierarchical_unet_v2_distillation_b0_from_b3"
DEVICE="cuda"
EPOCHS=50
BATCH_SIZE=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --resume)
      RESUME="$2"
      shift 2
      ;;
    --mixed-precision)
      MIXED_PRECISION="--mixed_precision"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Build command
CMD="uv run python train_distillation_staged.py"
CMD="$CMD --config $CONFIG"
CMD="$CMD --device $DEVICE"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --epochs $EPOCHS"

if [ ! -z "$RESUME" ]; then
  CMD="$CMD --resume $RESUME"
fi

if [ ! -z "$MIXED_PRECISION" ]; then
  CMD="$CMD $MIXED_PRECISION"
fi

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
if [ ! -z "$RESUME" ]; then
  echo "  Resume from: $RESUME"
fi
if [ ! -z "$MIXED_PRECISION" ]; then
  echo "  Mixed precision: enabled"
fi

echo ""
echo "Command: $CMD"
echo ""
echo "=========================================="
echo ""

# Run training
$CMD