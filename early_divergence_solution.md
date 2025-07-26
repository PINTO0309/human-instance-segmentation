# Solution for Early Divergence Issue

## Problem Summary
The refined model diverges to NaN at epoch 4, with very high loss values (3.6-3.9) that don't improve.

## Implemented Solutions

### 1. Created Stable Configuration (✅ Done)
New configuration: `rgb_hierarchical_unet_v2_attention_r64m64_refined_stable`

**Key changes:**
- Only 2 refinement modules enabled:
  - `use_boundary_refinement=True`
  - `use_boundary_aware_loss=True`
- Disabled unstable modules:
  - `use_progressive_upsampling=False` (can be unstable with small ROI)
  - `use_active_contour_loss=False`
  - `use_contour_detection=False`
  - `use_distance_transform=False`

**Training parameters:**
- Learning rate: 5e-6 (reduced from 1e-5)
- Warmup epochs: 15 (increased from 10)
- Gradient clipping: 0.5 (reduced from 1.0)
- Mixed precision: False (disabled for stability)
- Weight decay: 1e-5 (reduced from 1e-4)

### 2. Added Loss Clamping (✅ Done)
All refinement losses now clamped to max=10.0:
- Active contour loss
- Boundary aware loss
- Contour detection loss
- Distance transform loss

### 3. Improved Numerical Stability (✅ Done)
- Edge normalization with near-zero checks
- Gradient clamping in active contour loss
- Conservative weight initialization

## Usage Instructions

### Step 1: Train with Stable Configuration
```bash
uv run python run_experiments.py \
  --configs rgb_hierarchical_unet_v2_attention_r64m64_refined_stable \
  --epochs 20 \
  --batch_size 2
```

### Step 2: Monitor Training
Watch for:
- Gradual loss decrease (should start around 1.5-2.0, not 3.6)
- Stable gradient norms (< 10)
- No NaN warnings

### Step 3: Progressive Refinement
After stable training, gradually enable more refinements:

1. **Phase 1** (current stable config): boundary_refinement + boundary_aware_loss
2. **Phase 2**: Add active_contour_loss
3. **Phase 3**: Add contour_detection
4. **Phase 4**: Add distance_transform

## Additional Recommendations

### If Still Unstable:
1. Further reduce learning rate to 1e-6
2. Increase batch size to 4 (if GPU memory allows)
3. Use gradient accumulation if batch size can't be increased
4. Try Adam optimizer instead of AdamW

### Debugging Tips:
1. Monitor individual loss components in tensorboard
2. Check which refinement module causes instability
3. Visualize gradients of specific layers
4. Save checkpoints frequently for analysis

## Expected Behavior
With the stable configuration:
- Initial loss: 1.5-2.0 (not 3.6)
- Steady improvement over epochs
- mIoU improvement from 0.18 to 0.50+ 
- No NaN losses throughout training

## Next Steps
1. Run training with stable config
2. If successful, create configs for phases 2-4
3. Fine-tune the fully refined model with very low learning rate (1e-6)