#!/usr/bin/env python3
"""Fix output_conv weights in existing checkpoints to ensure consistent channel interpretation."""

import torch
import argparse
import glob
from pathlib import Path

def fix_checkpoint(checkpoint_path, backup=True):
    """Fix output_conv weights in a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        backup: Whether to create a backup of the original file
    """
    print(f"\nProcessing: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Check if output_conv weights exist
    output_conv_weight_key = 'pretrained_unet.output_conv.weight'
    output_conv_bias_key = 'pretrained_unet.output_conv.bias'
    
    if output_conv_weight_key not in checkpoint['model_state_dict']:
        print("  No output_conv weights found - skipping")
        return False
    
    # Get current values
    current_weight = checkpoint['model_state_dict'][output_conv_weight_key]
    print(f"  Current values:")
    print(f"    Channel 0 (background): {current_weight[0, 0, 0, 0].item():.4f}")
    print(f"    Channel 1 (foreground): {current_weight[1, 0, 0, 0].item():.4f}")
    
    # Check if already fixed
    if (current_weight[0, 0, 0, 0].item() == -1.0 and 
        current_weight[1, 0, 0, 0].item() == 1.0):
        print("  Already using fixed values - skipping")
        return False
    
    # Create backup if requested
    if backup:
        backup_path = checkpoint_path.replace('.pth', '_backup.pth')
        torch.save(checkpoint, backup_path)
        print(f"  Created backup: {backup_path}")
    
    # Fix the weights
    checkpoint['model_state_dict'][output_conv_weight_key][0, 0, 0, 0] = -1.0
    checkpoint['model_state_dict'][output_conv_weight_key][1, 0, 0, 0] = 1.0
    checkpoint['model_state_dict'][output_conv_bias_key].zero_()
    
    # Save fixed checkpoint
    torch.save(checkpoint, checkpoint_path)
    print(f"  Fixed values:")
    print(f"    Channel 0 (background): -1.0")
    print(f"    Channel 1 (foreground): +1.0")
    print(f"  ✓ Checkpoint fixed and saved")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Fix output_conv weights in checkpoints')
    parser.add_argument('pattern', help='Pattern to match checkpoint files (e.g., "experiments/*/checkpoints/*.pth")')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backups')
    parser.add_argument('--specific', type=str, help='Fix a specific checkpoint file')
    
    args = parser.parse_args()
    
    if args.specific:
        # Fix specific checkpoint
        if Path(args.specific).exists():
            fixed = fix_checkpoint(args.specific, backup=not args.no_backup)
            if fixed:
                print(f"\n✓ Successfully fixed 1 checkpoint")
        else:
            print(f"Error: File not found: {args.specific}")
    else:
        # Find all matching checkpoints
        checkpoints = glob.glob(args.pattern, recursive=True)
        
        if not checkpoints:
            print(f"No checkpoints found matching pattern: {args.pattern}")
            return
        
        print(f"Found {len(checkpoints)} checkpoint(s) to process")
        
        fixed_count = 0
        for checkpoint_path in checkpoints:
            if fix_checkpoint(checkpoint_path, backup=not args.no_backup):
                fixed_count += 1
        
        print(f"\n✓ Successfully fixed {fixed_count} checkpoint(s)")

if __name__ == '__main__':
    main()