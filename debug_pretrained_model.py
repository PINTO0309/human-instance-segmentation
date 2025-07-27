"""Debug the pretrained model architecture and data flow."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.human_edge_detection.advanced.hierarchical_segmentation_rgb import (
    HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet
)

def visualize_activations():
    """Visualize activations at different stages of the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    
    # Create synthetic test data
    batch_size = 1
    img_height, img_width = 512, 512
    images = torch.randn(batch_size, 3, img_height, img_width).to(device)
    
    # Create two ROIs - one with clear person, one with overlapping persons
    rois = torch.tensor([
        [0, 100, 100, 250, 300],  # ROI 1
        [0, 200, 150, 350, 350],  # ROI 2
    ], dtype=torch.float32).to(device)
    
    # Hook to capture intermediate activations
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks
    hooks = []
    hooks.append(model.pretrained_unet.register_forward_hook(get_activation('full_image_mask')))
    hooks.append(model.roi_align_mask.register_forward_hook(get_activation('roi_masks')))
    hooks.append(model.roi_align_rgb.register_forward_hook(get_activation('roi_rgb')))
    hooks.append(model.rgb_feature_extractor.register_forward_hook(get_activation('rgb_features')))
    
    # Also hook into the segmentation head
    if hasattr(model.segmentation_head, 'input_adjust'):
        hooks.append(model.segmentation_head.input_adjust.register_forward_hook(get_activation('adjusted_features')))
    if hasattr(model.segmentation_head, 'feature_processor'):
        hooks.append(model.segmentation_head.feature_processor.register_forward_hook(get_activation('processed_features')))
    if hasattr(model.segmentation_head, 'target_vs_nontarget_branch'):
        hooks.append(model.segmentation_head.target_vs_nontarget_branch.register_forward_hook(get_activation('target_nontarget')))
    
    # Forward pass
    with torch.no_grad():
        predictions, aux_outputs = model(images, rois)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print shapes and statistics
    print("=== Activation Shapes and Statistics ===")
    for name, act in activations.items():
        print(f"\n{name}:")
        print(f"  Shape: {act.shape}")
        print(f"  Range: [{act.min():.3f}, {act.max():.3f}]")
        print(f"  Mean: {act.mean():.3f}, Std: {act.std():.3f}")
    
    # Analyze the flow
    print("\n=== Data Flow Analysis ===")
    
    # Check full image mask
    full_mask = activations.get('full_image_mask')
    if full_mask is not None:
        fg_prob_full = torch.sigmoid(full_mask)
        print(f"Full image foreground probability - Mean: {fg_prob_full.mean():.3f}")
    
    # Check ROI masks
    roi_masks = activations.get('roi_masks')
    if roi_masks is not None:
        fg_prob_roi = torch.sigmoid(roi_masks)
        print(f"\nROI foreground probabilities:")
        for i in range(roi_masks.shape[0]):
            print(f"  ROI {i}: Mean={fg_prob_roi[i].mean():.3f}, Max={fg_prob_roi[i].max():.3f}")
    
    # Check RGB features
    rgb_features = activations.get('rgb_features')
    if rgb_features is not None:
        print(f"\nRGB features norm by ROI:")
        for i in range(rgb_features.shape[0]):
            print(f"  ROI {i}: {rgb_features[i].norm():.3f}")
    
    # Check target/non-target outputs
    target_nontarget = activations.get('target_nontarget')
    if target_nontarget is not None:
        tn_probs = F.softmax(target_nontarget, dim=1)
        print(f"\nTarget/Non-target probabilities by ROI:")
        for i in range(target_nontarget.shape[0]):
            target_prob = tn_probs[i, 0].mean()
            nontarget_prob = tn_probs[i, 1].mean()
            print(f"  ROI {i}: Target={target_prob:.3f}, Non-target={nontarget_prob:.3f}")
    
    # Check final predictions
    final_probs = F.softmax(predictions, dim=1)
    print(f"\nFinal prediction probabilities by ROI:")
    for i in range(predictions.shape[0]):
        bg_prob = final_probs[i, 0].mean()
        target_prob = final_probs[i, 1].mean()
        nontarget_prob = final_probs[i, 2].mean()
        print(f"  ROI {i}: BG={bg_prob:.3f}, Target={target_prob:.3f}, Non-target={nontarget_prob:.3f}")
    
    # Visualize if we have 2 ROIs
    if predictions.shape[0] == 2:
        fig, axes = plt.subplots(3, 2, figsize=(10, 12))
        
        for i in range(2):
            # ROI mask (from pretrained UNet)
            if roi_masks is not None:
                mask = torch.sigmoid(roi_masks[i, 0]).cpu().numpy()
                axes[0, i].imshow(mask, cmap='hot')
                axes[0, i].set_title(f'ROI {i}: Binary Mask from UNet')
            
            # Target probability
            target_prob = final_probs[i, 1].cpu().numpy()
            axes[1, i].imshow(target_prob, cmap='hot')
            axes[1, i].set_title(f'ROI {i}: Target Probability')
            
            # Non-target probability
            nontarget_prob = final_probs[i, 2].cpu().numpy()
            axes[2, i].imshow(nontarget_prob, cmap='hot')
            axes[2, i].set_title(f'ROI {i}: Non-target Probability')
        
        plt.tight_layout()
        plt.savefig('debug_activations.png')
        print("\nVisualization saved as 'debug_activations.png'")

def check_gradient_flow():
    """Check gradient flow through the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HierarchicalRGBSegmentationModelWithFullImagePretrainedUNet(
        roi_size=(64, 48),
        mask_size=(64, 48),
        pretrained_weights_path="ext_extractor/2020-09-23a.pth",
        use_attention_module=True,
        freeze_pretrained_weights=True,
    ).to(device)
    model.train()
    
    # Create synthetic data
    images = torch.randn(1, 3, 512, 512).to(device)
    rois = torch.tensor([[0, 100, 100, 200, 200]], dtype=torch.float32).to(device)
    
    # Create target with foreground object
    target = torch.ones(1, 64, 48, dtype=torch.long).to(device)  # All target class
    
    # Forward pass
    predictions, aux_outputs = model(images, rois)
    
    # Compute loss
    loss = F.cross_entropy(predictions, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients in different parts
    print("\n=== Gradient Analysis ===")
    
    # RGB feature extractor
    rgb_grad_norms = []
    for name, param in model.rgb_feature_extractor.named_parameters():
        if param.grad is not None:
            rgb_grad_norms.append(param.grad.norm().item())
    print(f"RGB feature extractor - Avg gradient norm: {np.mean(rgb_grad_norms):.6f}")
    
    # Segmentation head
    head_grad_norms = []
    for name, param in model.segmentation_head.named_parameters():
        if param.grad is not None:
            head_grad_norms.append(param.grad.norm().item())
            if 'target_vs_nontarget' in name:
                print(f"  {name}: {param.grad.norm().item():.6f}")
    print(f"Segmentation head - Avg gradient norm: {np.mean(head_grad_norms):.6f}")

def main():
    print("=== Debugging Pretrained Model Architecture ===\n")
    
    print("1. Visualizing activations:")
    print("-" * 50)
    visualize_activations()
    
    print("\n\n2. Checking gradient flow:")
    print("-" * 50)
    check_gradient_flow()

if __name__ == "__main__":
    main()