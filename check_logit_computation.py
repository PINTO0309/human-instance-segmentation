"""Check the logit computation method in PretrainedUNetGuidedSegmentationHead."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    PretrainedUNetGuidedSegmentationHead
)

def visualize_outputs():
    """Visualize how the outputs are computed."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test case
    batch_size = 1
    in_channels = 256
    roi_height, roi_width = 8, 8  # Small for visualization
    
    # Create segmentation head
    head = PretrainedUNetGuidedSegmentationHead(
        in_channels=in_channels,
        mid_channels=256,
        num_classes=3,
        mask_size=(roi_height, roi_width),
        use_attention_module=True
    ).to(device)
    
    # Create simple test inputs
    features = torch.randn(batch_size, in_channels, roi_height, roi_width).to(device)
    
    # Create a clear binary mask
    bg_fg_mask = torch.zeros(batch_size, 1, roi_height, roi_width).to(device)
    # Top-left quadrant: background (logit = -5)
    bg_fg_mask[:, :, :4, :4] = -5.0
    # Top-right quadrant: strong foreground (logit = 5)
    bg_fg_mask[:, :, :4, 4:] = 5.0
    # Bottom half: moderate foreground (logit = 2)
    bg_fg_mask[:, :, 4:, :] = 2.0
    
    print("Binary mask (logits):")
    print(bg_fg_mask[0, 0].cpu().numpy())
    print("\nBinary mask (probabilities):")
    print(torch.sigmoid(bg_fg_mask)[0, 0].cpu().numpy())
    
    # Forward pass
    with torch.no_grad():
        final_logits, aux_outputs = head(features, bg_fg_mask)
    
    # Convert to probabilities for visualization
    final_probs = F.softmax(final_logits, dim=1)
    
    print("\nFinal probabilities:")
    for i in range(3):
        class_name = ['Background', 'Target', 'Non-target'][i]
        print(f"\n{class_name} (Class {i}):")
        print(final_probs[0, i].cpu().numpy())
    
    # Check if probabilities sum to 1
    prob_sum = final_probs.sum(dim=1)
    print(f"\nProbability sum (should be 1.0):")
    print(prob_sum[0].cpu().numpy())
    
    # Check target/non-target logits
    print("\nTarget/Non-target logits (before masking):")
    print("Target:", aux_outputs['target_nontarget_logits'][0, 0].cpu().numpy())
    print("Non-target:", aux_outputs['target_nontarget_logits'][0, 1].cpu().numpy())
    
    # Analyze different regions
    print("\n=== Regional Analysis ===")
    regions = [
        ("Background (top-left)", slice(0, 4), slice(0, 4)),
        ("Strong FG (top-right)", slice(0, 4), slice(4, 8)),
        ("Moderate FG (bottom)", slice(4, 8), slice(0, 8))
    ]
    
    for name, row_slice, col_slice in regions:
        print(f"\n{name}:")
        fg_prob = torch.sigmoid(bg_fg_mask[:, :, row_slice, col_slice]).mean().item()
        print(f"  Avg FG probability: {fg_prob:.3f}")
        
        for i in range(3):
            class_name = ['BG', 'Target', 'Non-target'][i]
            avg_prob = final_probs[:, i, row_slice, col_slice].mean().item()
            print(f"  Avg {class_name} prob: {avg_prob:.3f}")

def test_gradient_masking():
    """Test if gradients are properly masked in background regions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create segmentation head
    head = PretrainedUNetGuidedSegmentationHead(
        in_channels=256,
        mid_channels=256,
        num_classes=3,
        mask_size=8,
        use_attention_module=True
    ).to(device)
    
    # Create inputs
    features = torch.randn(1, 256, 8, 8, requires_grad=True).to(device)
    
    # Create binary mask with clear regions
    bg_fg_mask = torch.zeros(1, 1, 8, 8).to(device)
    bg_fg_mask[:, :, :4, :] = -5.0  # Top half: background
    bg_fg_mask[:, :, 4:, :] = 5.0   # Bottom half: foreground
    
    # Create target that matches the mask
    target = torch.zeros(1, 8, 8, dtype=torch.long).to(device)
    target[:, 4:, :4] = 1  # Target in bottom-left
    target[:, 4:, 4:] = 2  # Non-target in bottom-right
    
    # Forward pass
    final_logits, _ = head(features, bg_fg_mask)
    
    # Compute loss only in foreground regions
    fg_mask = torch.sigmoid(bg_fg_mask).squeeze(1) > 0.5
    loss = F.cross_entropy(final_logits, target, reduction='none')
    masked_loss = (loss * fg_mask).sum() / fg_mask.sum()
    
    print(f"Masked loss: {masked_loss.item():.4f}")
    
    # Backward pass
    masked_loss.backward()
    
    # Check if target/non-target branch receives gradients
    print("\nGradient norms for target/non-target branch:")
    for name, param in head.target_vs_nontarget_branch.named_parameters():
        if param.grad is not None:
            print(f"  {name}: {param.grad.norm().item():.6f}")

def main():
    print("=== Testing Logit Computation ===\n")
    
    print("1. Visualizing output computation:")
    print("-" * 50)
    visualize_outputs()
    
    print("\n\n2. Testing gradient masking:")
    print("-" * 50)
    test_gradient_masking()

if __name__ == "__main__":
    main()