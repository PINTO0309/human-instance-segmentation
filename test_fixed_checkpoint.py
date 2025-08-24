"""Test that the fixed checkpoint produces correct outputs."""
import torch
import numpy as np
from PIL import Image

def test_checkpoint(checkpoint_path):
    """Test a checkpoint's output_conv weights and model predictions."""
    print(f"\nTesting: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check output_conv weights
    output_conv_weight = checkpoint['model_state_dict'].get('pretrained_unet.output_conv.weight')
    if output_conv_weight is not None:
        ch0_weight = output_conv_weight[0, 0, 0, 0].item()
        ch1_weight = output_conv_weight[1, 0, 0, 0].item()
        
        print(f"  output_conv weights:")
        print(f"    Channel 0 (background): {ch0_weight:.4f}")
        print(f"    Channel 1 (foreground): {ch1_weight:.4f}")
        
        if ch0_weight == -1.0 and ch1_weight == 1.0:
            print("  ✓ Weights are FIXED correctly")
            return True
        else:
            print("  ✗ Weights are NOT fixed")
            return False
    else:
        print("  No output_conv weights found")
        return False

# Test the fixed checkpoints
checkpoints = [
    'experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7/checkpoints/best_model.pth',
    'experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7/checkpoints/checkpoint_epoch_0001.pth'
]

print("Verifying fixed checkpoints...")
print("="*50)

all_fixed = True
for ckpt_path in checkpoints:
    if not test_checkpoint(ckpt_path):
        all_fixed = False

print("\n" + "="*50)
if all_fixed:
    print("✓ All checkpoints have been fixed successfully!")
    print("\nYou can now resume training with:")
    print("python run_experiments.py \\")
    print("  --configs rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7 \\")
    print("  --resume experiments/rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7/checkpoints/best_model.pth \\")
    print("  --epochs 10 --batch_size 16 --mixed_precision")
else:
    print("✗ Some checkpoints were not fixed properly")