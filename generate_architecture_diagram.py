"""Generate modern architecture diagram for ROI-based instance segmentation model."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Set up the figure with high DPI for quality
fig, ax = plt.subplots(1, 1, figsize=(14.65, 10), dpi=150)
ax.set_xlim(0, 14.65)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
colors = {
    'input': '#E8F4F8',
    'yolo': '#FFE5CC',
    'feature': '#D4E6F1',
    'roi': '#E8DAEF',
    'decoder': '#D5F4E6',
    'output': '#FADBD8',
    'arrow': '#34495E',
    'text': '#2C3E50',
    'border': '#95A5A6'
}

# Helper function to draw rounded rectangle with text
def draw_block(ax, x, y, w, h, text, color, fontsize=11, subtext=None):
    # Main block
    fancy_box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=colors['border'],
        linewidth=2,
        zorder=10
    )
    ax.add_patch(fancy_box)
    
    # Main text
    ax.text(x + w/2, y + h/2 + (0.1 if subtext else 0), text,
            ha='center', va='center', fontsize=fontsize, 
            fontweight='bold', color=colors['text'], zorder=11)
    
    # Subtext
    if subtext:
        ax.text(x + w/2, y + h/2 - 0.2, subtext,
                ha='center', va='center', fontsize=fontsize-2, 
                color=colors['text'], zorder=11)
    
    return fancy_box

# Helper function to draw arrow
def draw_arrow(ax, x1, y1, x2, y2, label='', curved=False):
    if curved:
        style = "arc3,rad=0.3"
    else:
        style = "arc3,rad=0"
    
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        connectionstyle=style,
        arrowstyle='-|>',
        mutation_scale=20,
        linewidth=2,
        color=colors['arrow'],
        zorder=5
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.1, label,
                ha='center', va='bottom', fontsize=9,
                color=colors['text'], 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='none', alpha=0.8))

# Title
ax.text(7.325, 9.5, 'ROI-based Instance Segmentation Architecture', 
        ha='center', va='center', fontsize=21, fontweight='bold', color=colors['text'])

# 1. Input Image (same height as Conv 1x1, aligned with Conv 1x1 left edge)
draw_block(ax, 1.5, 7.8, 2, 0.7, 'Input Image', colors['input'], 11, '640×640×3')

# 2. YOLOv9 Feature Extractor (same width as Input Image)
draw_block(ax, 5, 7.8, 2, 0.7, 'YOLOv9\nFeature Extractor', colors['yolo'], 11, 'ONNX Model')

# 3. Feature Maps (same width as Input Image)
draw_block(ax, 9, 7.8, 2, 0.7, 'Feature Maps', colors['feature'], 11, '1024×80×80')

# 4. ROI Coordinates (moved up, aligned with Conv 1x1 left edge)
draw_block(ax, 1.5, 6.2, 2, 0.7, 'ROI Coordinates', colors['input'], 11, '[x1, y1, x2, y2]')

# 5. DynamicRoIAlign (same width as Input Image)
draw_block(ax, 9, 6.2, 2, 0.7, 'DynamicRoIAlign', colors['roi'], 11, '28×28 Grid')

# 6. ROI Features (same width as Input Image)
draw_block(ax, 9, 4.8, 2, 0.7, 'ROI Features', colors['feature'], 10, '1024×28×28')

# 7. Decoder Network Components
decoder_y = 3.3
# Conv + LayerNorm (shifted left)
draw_block(ax, 1.5, decoder_y, 2, 0.7, 'Conv 1×1\n+ LayerNorm', colors['decoder'], 10, '256×28×28')
# Residual Blocks (shifted left)
draw_block(ax, 4, decoder_y, 2, 0.7, 'Residual\nBlocks ×2', colors['decoder'], 10, '256×28×28')
# Upsampling (shifted left)
draw_block(ax, 6.5, decoder_y, 2, 0.7, 'Progressive\nUpsampling', colors['decoder'], 10, '56×56')
# Multi-scale Fusion (shifted left)
draw_block(ax, 9, decoder_y, 2, 0.7, 'Multi-scale\nFusion', colors['decoder'], 10)

# 8. Output (shifted left)
draw_block(ax, 12, decoder_y, 2, 0.7, '3-Class Masks', colors['output'], 11, '3×56×56')

# Draw arrows (adjusted for new positions, shortened to stay within block boundaries)
# Input to YOLO
draw_arrow(ax, 3.5, 8.15, 4.95, 8.15)
# YOLO to Features
draw_arrow(ax, 7.05, 8.15, 8.95, 8.15)
# Features to ROIAlign
draw_arrow(ax, 10, 7.75, 10, 6.95)
# ROI coords to ROIAlign
draw_arrow(ax, 3.5, 6.55, 8.95, 6.55, 'ROI coords')
# ROIAlign to ROI Features (straight down)
draw_arrow(ax, 10, 6.15, 10, 5.55)
# ROI Features to Conv 1x1 (single straight line)
draw_arrow(ax, 8.95, 5.15, 3.55, 4.0)
# Decoder chain
draw_arrow(ax, 3.55, 3.65, 3.95, 3.65)
draw_arrow(ax, 6.05, 3.65, 6.45, 3.65)
draw_arrow(ax, 8.55, 3.65, 8.95, 3.65)
draw_arrow(ax, 11.05, 3.65, 11.95, 3.65)

# Add classes box (same width and position as 3-Class Masks)
classes_box = FancyBboxPatch(
    (12, 0.7), 2, 1.8,  # Same x=12 and width=2 as 3-Class Masks
    boxstyle="round,pad=0.1",
    facecolor='#F8F9F9',
    edgecolor=colors['border'],
    linewidth=1.5,
    zorder=1  # Lower z-order so text appears on top
)
ax.add_patch(classes_box)
# Add class labels (centered in the box)
ax.text(13, 2.2, 'Classes', ha='center', fontsize=12, fontweight='bold', zorder=15)
ax.text(12.2, 1.9, '• 0: Background', ha='left', fontsize=10, color='#E74C3C', zorder=15)
ax.text(12.2, 1.65, '• 1: Target', ha='left', fontsize=10, color='#27AE60', zorder=15)
ax.text(12.2, 1.4, '• 2: Non-target', ha='left', fontsize=10, color='#3498DB', zorder=15)

# Add key features box (aligned with Conv 1x1 left and Residual Blocks x2 right)
features_box = FancyBboxPatch(
    (1.5, 0.7), 4.5, 1.8,
    boxstyle="round,pad=0.1",
    facecolor='#F8F9F9',
    edgecolor=colors['border'],
    linewidth=1.5,
    zorder=1  # Lower z-order so text appears on top
)
ax.add_patch(features_box)
ax.text(3.75, 2.2, 'Key Features', ha='center', fontsize=12, fontweight='bold', zorder=15)
features_text = [
    '• Enhanced 28×28 ROI size',
    '• Residual blocks with LayerNorm',
    '• Progressive upsampling',
    '• Multi-scale feature fusion',
    '• 3-class segmentation'
]
for i, text in enumerate(features_text):
    ax.text(1.7, 1.9 - i*0.25, text, ha='left', fontsize=10, color=colors['text'], zorder=15)

# Add model specs box (aligned with Progressive Upsampling left and Multi-scale Fusion right)
specs_box = FancyBboxPatch(
    (6.5, 0.7), 4.5, 1.8,
    boxstyle="round,pad=0.1",
    facecolor='#F8F9F9',
    edgecolor=colors['border'],
    linewidth=1.5,
    zorder=1  # Lower z-order so text appears on top
)
ax.add_patch(specs_box)
ax.text(8.75, 2.2, 'Model Specifications', ha='center', fontsize=12, fontweight='bold', zorder=15)
specs_text = [
    '• Input: 640×640 RGB',
    '• Feature stride: 8',
    '• ROI output: 56×56',
    '• ONNX compatible',
    '• GPU accelerated'
]
for i, text in enumerate(specs_text):
    ax.text(6.7, 1.9 - i*0.25, text, ha='left', fontsize=10, color=colors['text'], zorder=15)

# Add visual flow indicators (adjusted for new positions)
# Feature extraction flow (moved closer to YOLOv9 block)
ax.annotate('Feature\nExtraction', xy=(6, 8.6), xytext=(6, 9.0),
            ha='center', va='center', fontsize=10, color=colors['text'],
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1),
            zorder=15)

# ROI processing flow (shortened arrow to end at ROI Features right edge)
ax.annotate('ROI\nProcessing', xy=(11.1, 5.15), xytext=(11.7, 5.15),
            ha='center', va='center', fontsize=10, color=colors['text'],
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1),
            zorder=15)

# Decoder flow (adjusted for shifted Conv 1x1)
ax.annotate('Segmentation\nDecoder', xy=(1.4, 3.65), xytext=(0.65, 3.65),
            ha='center', va='center', fontsize=10, color=colors['text'],
            arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1),
            zorder=15)

# Save the figure
plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Architecture diagram saved as 'architecture_diagram.png'")