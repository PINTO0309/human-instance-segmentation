import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import seaborn as sns

# TensorBoardログファイルの読み込み
log_dir = "logs/"
log_file = "events.out.tfevents.1752772231.132-226-80-87.4960.0"

# EventAccumulatorを使用してログを読み込む
event_acc = EventAccumulator(os.path.join(log_dir, log_file))
event_acc.Reload()

# 利用可能なタグを確認
print("Available scalar tags:")
for tag in event_acc.Tags()['scalars']:
    print(f"  - {tag}")

# データの収集
metrics = {}
for tag in event_acc.Tags()['scalars']:
    events = event_acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    metrics[tag] = pd.DataFrame({'step': steps, 'value': values})

# プロット作成
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
fig.suptitle('Training Metrics Analysis', fontsize=16)

# Loss plots
if 'train/loss' in metrics and 'val/loss' in metrics:
    ax = axes[0, 0]
    metrics['train/loss'].plot(x='step', y='value', ax=ax, label='Train Loss')
    metrics['val/loss'].plot(x='step', y='value', ax=ax, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True)

# mIoU plot
if 'val/mIoU' in metrics:
    ax = axes[0, 1]
    metrics['val/mIoU'].plot(x='step', y='value', ax=ax)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mIoU')
    ax.set_title('Validation mIoU')
    ax.grid(True)

# Class-wise IoU plots
iou_classes = ['val/IoU_class_0', 'val/IoU_class_1', 'val/IoU_class_2']
colors = ['blue', 'green', 'red']
labels = ['Background', 'Target', 'Non-target']

for i, (iou_tag, color, label) in enumerate(zip(iou_classes, colors, labels)):
    if iou_tag in metrics:
        ax = axes[1, i%2]
        metrics[iou_tag].plot(x='step', y='value', ax=ax, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title(f'{label} IoU')
        ax.grid(True)
        
        # 統計情報を追加
        values = metrics[iou_tag]['value'].values
        print(f"\n{label} IoU statistics:")
        print(f"  - Initial: {values[0]:.4f}")
        print(f"  - Final: {values[-1]:.4f}")
        print(f"  - Max: {values.max():.4f} (at epoch {values.argmax()})")
        print(f"  - Change: {values[-1] - values[0]:.4f}")

# Combined IoU plot
ax = axes[2, 1]
for iou_tag, color, label in zip(iou_classes, colors, labels):
    if iou_tag in metrics:
        metrics[iou_tag].plot(x='step', y='value', ax=ax, color=color, label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('IoU')
ax.set_title('All Classes IoU')
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.savefig('training_analysis.png', dpi=150)
print("\nPlot saved as training_analysis.png")

# 学習率の確認
if 'train/lr' in metrics:
    print("\nLearning rate statistics:")
    lr_values = metrics['train/lr']['value'].values
    print(f"  - Initial: {lr_values[0]:.6f}")
    print(f"  - Final: {lr_values[-1]:.6f}")
    print(f"  - Min: {lr_values.min():.6f}")