"""Analyze hierarchical training dynamics from log files."""

import re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def parse_training_log(log_path):
    """Parse training log file and extract metrics."""
    with open(log_path, 'r') as f:
        content = f.read()
    
    epochs = []
    train_losses = []
    val_losses = []
    val_iou_0 = []
    val_iou_1 = []
    val_iou_2 = []
    val_miou = []
    learning_rates = []
    
    # Find all epoch summaries
    epoch_pattern = r'\[.*?\] Epoch (\d+) Summary.*?Learning Rate: ([\d.]+).*?Training Metrics:.*?total_loss: ([\d.]+).*?Validation Metrics:.*?iou_class_0: ([\d.]+).*?iou_class_1: ([\d.]+).*?iou_class_2: ([\d.]+).*?miou: ([\d.]+).*?total_loss: ([\d.]+)'
    
    matches = re.finditer(epoch_pattern, content, re.DOTALL)
    
    for match in matches:
        epochs.append(int(match.group(1)))
        learning_rates.append(float(match.group(2)))
        train_losses.append(float(match.group(3)))
        val_iou_0.append(float(match.group(4)))
        val_iou_1.append(float(match.group(5)))
        val_iou_2.append(float(match.group(6)))
        val_miou.append(float(match.group(7)))
        val_losses.append(float(match.group(8)))
    
    return {
        'epochs': epochs,
        'learning_rates': learning_rates,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_iou_0': val_iou_0,
        'val_iou_1': val_iou_1,
        'val_iou_2': val_iou_2,
        'val_miou': val_miou
    }


def plot_training_dynamics(data, save_path):
    """Create comprehensive training dynamics plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Hierarchical Training Dynamics Analysis', fontsize=16)
    
    # Plot 1: Training vs Validation Loss
    ax = axes[0, 0]
    ax.plot(data['epochs'], data['train_losses'], 'b-', label='Train Loss')
    ax.plot(data['epochs'], data['val_losses'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training vs Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Individual Class IoUs
    ax = axes[0, 1]
    ax.plot(data['epochs'], data['val_iou_0'], 'g-', label='Background (Class 0)')
    ax.plot(data['epochs'], data['val_iou_1'], 'b-', label='Target (Class 1)')
    ax.plot(data['epochs'], data['val_iou_2'], 'r-', label='Non-target (Class 2)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.set_title('Individual Class IoUs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: mIoU vs Validation Loss
    ax = axes[0, 2]
    ax2 = ax.twinx()
    ax.plot(data['epochs'], data['val_miou'], 'b-', label='mIoU')
    ax2.plot(data['epochs'], data['val_losses'], 'r-', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU', color='b')
    ax2.set_ylabel('Val Loss', color='r')
    ax.set_title('mIoU vs Validation Loss')
    ax.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Loss ratio (Val/Train)
    ax = axes[1, 0]
    loss_ratio = np.array(data['val_losses']) / np.array(data['train_losses'])
    ax.plot(data['epochs'], loss_ratio, 'm-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Val Loss / Train Loss')
    ax.set_title('Overfitting Indicator (Val/Train Loss Ratio)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    
    # Plot 5: Learning Rate
    ax = axes[1, 1]
    ax.plot(data['epochs'], data['learning_rates'], 'k-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Class IoU correlations
    ax = axes[1, 2]
    # Calculate correlation between background IoU and val loss
    from scipy.stats import pearsonr
    corr_bg_loss, _ = pearsonr(data['val_iou_0'], data['val_losses'])
    corr_tgt_loss, _ = pearsonr(data['val_iou_1'], data['val_losses'])
    corr_ntgt_loss, _ = pearsonr(data['val_iou_2'], data['val_losses'])
    
    correlations = [corr_bg_loss, corr_tgt_loss, corr_ntgt_loss]
    labels = ['Background', 'Target', 'Non-target']
    colors = ['g', 'b', 'r']
    
    bars = ax.bar(labels, correlations, color=colors, alpha=0.7)
    ax.set_ylabel('Correlation with Val Loss')
    ax.set_title('Class IoU vs Val Loss Correlations')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylim(-1, 1)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Analysis plot saved to: {save_path}")
    
    # Print analysis summary
    print("\n=== Training Analysis Summary ===")
    print(f"Total epochs analyzed: {len(data['epochs'])}")
    print(f"Final training loss: {data['train_losses'][-1]:.4f}")
    print(f"Final validation loss: {data['val_losses'][-1]:.4f}")
    print(f"Overfitting ratio: {data['val_losses'][-1] / data['train_losses'][-1]:.2f}x")
    print(f"\nFinal IoUs:")
    print(f"  Background: {data['val_iou_0'][-1]:.4f}")
    print(f"  Target: {data['val_iou_1'][-1]:.4f}")
    print(f"  Non-target: {data['val_iou_2'][-1]:.4f}")
    print(f"  mIoU: {data['val_miou'][-1]:.4f}")
    print(f"\nCorrelations with val loss:")
    print(f"  Background IoU: {corr_bg_loss:.3f} (should be negative)")
    print(f"  Target IoU: {corr_tgt_loss:.3f} (should be negative)")
    print(f"  Non-target IoU: {corr_ntgt_loss:.3f} (should be negative)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_hierarchical_training.py <log_file_path> [output_path]")
        sys.exit(1)
    
    log_path = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    else:
        output_path = log_path.parent / f"training_analysis_{log_path.stem}.png"
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)
    
    print(f"Analyzing log file: {log_path}")
    data = parse_training_log(log_path)
    
    if not data['epochs']:
        print("Error: No training data found in log file")
        sys.exit(1)
    
    plot_training_dynamics(data, output_path)