#!/bin/bash

# Script to run knowledge distillation training from B3 teacher to B0 student

# Default values
TEACHER_CHECKPOINT=""
EPOCHS=100
TEMPERATURE=""
ALPHA=""
CONFIG="rgb_hierarchical_unet_v2_distillation_b0_from_b3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --teacher_checkpoint)
      TEACHER_CHECKPOINT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--teacher_checkpoint PATH] [--epochs N] [--temperature T] [--alpha A] [--config CONFIG]"
      exit 1
      ;;
  esac
done

echo "Starting knowledge distillation training..."
echo "Configuration: $CONFIG"
if [ -n "$TEACHER_CHECKPOINT" ]; then
  echo "Teacher checkpoint: $TEACHER_CHECKPOINT"
fi
echo "Epochs: $EPOCHS"
if [ -n "$TEMPERATURE" ]; then
  echo "Temperature: $TEMPERATURE"
fi
if [ -n "$ALPHA" ]; then
  echo "Alpha: $ALPHA"
fi
echo ""

# Build command
CMD="uv run python train_advanced.py --config $CONFIG --epochs $EPOCHS"

if [ -n "$TEACHER_CHECKPOINT" ]; then
  CMD="$CMD --teacher_checkpoint $TEACHER_CHECKPOINT"
fi

if [ -n "$TEMPERATURE" ]; then
  CMD="$CMD --distillation_temperature $TEMPERATURE"
fi

if [ -n "$ALPHA" ]; then
  CMD="$CMD --distillation_alpha $ALPHA"
fi

# Run training
echo "Running: $CMD"
echo ""
$CMD

echo ""
echo "Training complete. Check experiments/$CONFIG/ for results."