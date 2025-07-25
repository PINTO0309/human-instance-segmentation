#!/usr/bin/env python3
"""Monitor hierarchical training progress and dynamic weight changes."""

import json
from pathlib import Path
import sys
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


def load_tensorboard_data(log_dir: Path) -> Dict[str, List[float]]:
    """Load training metrics from TensorBoard logs."""
    # This is a simplified version - in practice you'd use tensorboard API
    # For now, we'll parse from the experiment's checkpoint JSON files
    data = {
        'epoch': [],
        'train/total_loss': [],
        'val/total_loss': [],
        'val/miou': [],
        'bg_weight': [],
        'fg_weight': [],
        'target_weight': [],
        'nontarget_weight': []
    }
    
    # Look for checkpoint JSON files
    checkpoint_dir = log_dir.parent / 'checkpoints'
    if checkpoint_dir.exists():
        for json_file in sorted(checkpoint_dir.glob('*.json')):
            if 'epoch_' in json_file.stem:
                with open(json_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    epoch = checkpoint_data.get('epoch', -1)
                    if epoch >= 0:
                        data['epoch'].append(epoch)
                        # Extract metrics if available
                        metrics = checkpoint_data.get('metrics', {})
                        data['train/total_loss'].append(metrics.get('train_loss', 0))
                        data['val/total_loss'].append(metrics.get('val_loss', 0))
                        data['val/miou'].append(metrics.get('val_miou', 0))
    
    return data


def plot_training_progress(data: Dict[str, List[float]], output_path: Path):
    """Create plots showing training progress and weight dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    if data['epoch']:
        ax1.plot(data['epoch'], data['train/total_loss'], label='Train Loss', marker='o')
        ax1.plot(data['epoch'], data['val/total_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
    
    # Plot 2: mIoU
    ax2 = axes[0, 1]
    if data['epoch'] and data['val/miou']:
        ax2.plot(data['epoch'], data['val/miou'], label='Val mIoU', marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.set_title('Validation mIoU')
        ax2.legend()
        ax2.grid(True)
    
    # Plot 3: Dynamic weights (bg/fg)
    ax3 = axes[1, 0]
    if data['bg_weight']:
        ax3.plot(data['epoch'][:len(data['bg_weight'])], data['bg_weight'], label='BG Weight', marker='o')
        ax3.plot(data['epoch'][:len(data['fg_weight'])], data['fg_weight'], label='FG Weight', marker='s')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Weight')
        ax3.set_title('Background/Foreground Dynamic Weights')
        ax3.legend()
        ax3.grid(True)
    
    # Plot 4: Dynamic weights (target/non-target)
    ax4 = axes[1, 1]
    if data['target_weight']:
        ax4.plot(data['epoch'][:len(data['target_weight'])], data['target_weight'], label='Target Weight', marker='o')
        ax4.plot(data['epoch'][:len(data['nontarget_weight'])], data['nontarget_weight'], label='Non-target Weight', marker='s')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Weight')
        ax4.set_title('Target/Non-target Dynamic Weights')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved training progress plot to {output_path}")


def analyze_weight_stability(data: Dict[str, List[float]]):
    """Analyze the stability of dynamic weights."""
    print("\n=== Weight Stability Analysis ===")
    
    weight_keys = ['bg_weight', 'fg_weight', 'target_weight', 'nontarget_weight']
    
    for key in weight_keys:
        if key in data and len(data[key]) > 1:
            weights = np.array(data[key])
            mean_weight = np.mean(weights)
            std_weight = np.std(weights)
            cv = std_weight / mean_weight if mean_weight > 0 else 0
            
            print(f"\n{key}:")
            print(f"  Mean: {mean_weight:.3f}")
            print(f"  Std: {std_weight:.3f}")
            print(f"  CV (Coefficient of Variation): {cv:.3f}")
            
            # Check for instability
            if cv > 0.5:
                print(f"  WARNING: High variation detected! Weights are unstable.")
            elif cv > 0.2:
                print(f"  CAUTION: Moderate variation in weights.")
            else:
                print(f"  OK: Weights are relatively stable.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python monitor_hierarchical_training.py <experiment_dir>")
        sys.exit(1)
    
    experiment_dir = Path(sys.argv[1])
    if not experiment_dir.exists():
        print(f"Error: Experiment directory {experiment_dir} does not exist")
        sys.exit(1)
    
    log_dir = experiment_dir / 'logs'
    
    print(f"Monitoring experiment: {experiment_dir.name}")
    
    # Load data
    data = load_tensorboard_data(log_dir)
    
    # Create plots
    plot_path = experiment_dir / 'training_progress.png'
    plot_training_progress(data, plot_path)
    
    # Analyze weight stability
    analyze_weight_stability(data)
    
    # Print latest metrics
    if data['epoch']:
        latest_epoch = data['epoch'][-1]
        print(f"\n=== Latest Metrics (Epoch {latest_epoch}) ===")
        print(f"Train Loss: {data['train/total_loss'][-1]:.4f}" if data['train/total_loss'] else "N/A")
        print(f"Val Loss: {data['val/total_loss'][-1]:.4f}" if data['val/total_loss'] else "N/A")
        print(f"Val mIoU: {data['val/miou'][-1]:.4f}" if data['val/miou'] else "N/A")


if __name__ == "__main__":
    main()