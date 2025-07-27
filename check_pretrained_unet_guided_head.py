"""Check PretrainedUNetGuidedSegmentationHead behavior."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    PretrainedUNetGuidedSegmentationHead,
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)

def check_segmentation_head():
    """Check if the segmentation head processes features correctly."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple test case
    batch_size = 2
    in_channels = 256
    roi_height, roi_width = 64, 48
    mask_height, mask_width = 64, 48
    
    # Create segmentation head
    head = PretrainedUNetGuidedSegmentationHead(
        in_channels=in_channels,
        mid_channels=256,
        num_classes=3,
        mask_size=(mask_height, mask_width),
        use_attention_module=True
    ).to(device)
    
    # Create test inputs
    # Features from RGB extractor
    features = torch.randn(batch_size, in_channels, roi_height, roi_width).to(device)
    
    # Binary mask from pre-trained UNet (simulating perfect foreground mask)
    # Create a clear foreground region in the center
    bg_fg_mask = torch.zeros(batch_size, 1, roi_height, roi_width).to(device)
    # Set center region to high values (will be foreground after sigmoid)
    h_start, h_end = roi_height // 4, 3 * roi_height // 4
    w_start, w_end = roi_width // 4, 3 * roi_width // 4
    bg_fg_mask[:, :, h_start:h_end, w_start:w_end] = 5.0  # High logit value
    
    print(f"Input shapes:")
    print(f"  Features: {features.shape}")
    print(f"  BG/FG mask: {bg_fg_mask.shape}")
    print(f"  BG/FG mask range: [{bg_fg_mask.min():.3f}, {bg_fg_mask.max():.3f}]")
    
    # Forward pass
    with torch.no_grad():
        final_logits, aux_outputs = head(features, bg_fg_mask)
    
    print(f"\nOutput shapes:")
    print(f"  Final logits: {final_logits.shape}")
    print(f"  Final logits range: [{final_logits.min():.3f}, {final_logits.max():.3f}]")
    
    # Check intermediate values
    fg_prob = torch.sigmoid(bg_fg_mask)
    print(f"\nForeground probability:")
    print(f"  Shape: {fg_prob.shape}")
    print(f"  Range: [{fg_prob.min():.3f}, {fg_prob.max():.3f}]")
    print(f"  Center region mean: {fg_prob[:, :, h_start:h_end, w_start:w_end].mean():.3f}")
    print(f"  Background region mean: {fg_prob[:, :, :h_start, :].mean():.3f}")
    
    # Check final probabilities
    final_probs = torch.exp(final_logits)  # Since final_logits = log(probs)
    print(f"\nFinal probabilities (per class):")
    for i in range(3):
        print(f"  Class {i}: range [{final_probs[:, i].min():.3f}, {final_probs[:, i].max():.3f}]")
        print(f"    Center region mean: {final_probs[:, i, h_start:h_end, w_start:w_end].mean():.3f}")
        print(f"    Background region mean: {final_probs[:, i, :h_start, :].mean():.3f}")
    
    # Check if probabilities sum to 1
    prob_sum = final_probs.sum(dim=1)
    print(f"\nProbability sum check:")
    print(f"  Range: [{prob_sum.min():.3f}, {prob_sum.max():.3f}]")
    print(f"  Mean: {prob_sum.mean():.3f}")
    
    # Check auxiliary outputs
    print(f"\nAuxiliary outputs:")
    for key, value in aux_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}, range [{value.min():.3f}, {value.max():.3f}]")

def check_gradient_flow():
    """Check if gradients flow properly through the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create segmentation head
    head = PretrainedUNetGuidedSegmentationHead(
        in_channels=256,
        mid_channels=256,
        num_classes=3,
        mask_size=(64, 48),
        use_attention_module=True
    ).to(device)
    
    # Create inputs
    features = torch.randn(2, 256, 64, 48, requires_grad=True).to(device)
    bg_fg_mask = torch.randn(2, 1, 64, 48).to(device)
    
    # Create target (simulate perfect segmentation)
    target = torch.zeros(2, 64, 48, dtype=torch.long).to(device)
    # Set center to class 1 (target)
    target[:, 16:48, 12:36] = 1
    
    # Forward pass
    final_logits, _ = head(features, bg_fg_mask)
    
    # Compute loss
    loss = F.cross_entropy(final_logits, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print(f"\nGradient check:")
    print(f"  Input features gradient: {features.grad is not None}")
    if features.grad is not None:
        print(f"    Gradient norm: {features.grad.norm():.6f}")
        print(f"    Gradient range: [{features.grad.min():.6f}, {features.grad.max():.6f}]")
    
    # Check parameter gradients
    print(f"\nParameter gradients:")
    for name, param in head.named_parameters():
        if param.grad is not None:
            print(f"  {name}: norm={param.grad.norm():.6f}, range=[{param.grad.min():.6f}, {param.grad.max():.6f}]")
        else:
            print(f"  {name}: No gradient!")

def main():
    print("=== Checking PretrainedUNetGuidedSegmentationHead ===\n")
    
    print("1. Testing forward pass behavior:")
    print("-" * 50)
    check_segmentation_head()
    
    print("\n\n2. Testing gradient flow:")
    print("-" * 50)
    check_gradient_flow()

if __name__ == "__main__":
    main()