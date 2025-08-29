#\!/bin/bash

# Improved UNet encoder-only distillation training script
# With fixes for IoU stagnation issues

echo "=========================================="
echo "Improved UNet Distillation Training"
echo "=========================================="
echo ""
echo "Key improvements:"
echo "1. Balanced loss function (BCE + Distillation)"
echo "2. Adjusted learning rate (5e-4)"
echo "3. Proper temperature scaling (T=4.0)"
echo "4. More weight on ground truth (task_weight=0.3)"
echo ""

# Kill any existing training processes
echo "Cleaning up previous runs..."
pkill -f train_distillation_staged.py 2>/dev/null

# Clear cache
echo "Clearing cache..."
rm -rf ~/.cache/torch/hub/checkpoints/* 2>/dev/null

echo ""
echo "Starting improved training..."
echo ""

# Run with improved settings
uv run python train_distillation_staged.py \
    --config rgb_hierarchical_unet_v2_distillation_b0_from_b3 \
    --batch_size 8 \
    --epochs 100 \
    --device cuda \
    --mixed_precision

echo ""
echo "Training complete\!"
echo "Check results in: experiments/rgb_hierarchical_unet_v2_distillation_b0_from_b3/"
