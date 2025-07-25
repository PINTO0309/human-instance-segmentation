"""Test the hierarchical segmentation fix to verify correct behavior."""

import torch
import torch.nn as nn
from src.human_edge_detection.advanced.hierarchical_segmentation import HierarchicalLoss


def test_hierarchical_loss():
    """Test that the hierarchical loss behaves correctly."""
    print("Testing Hierarchical Loss Fix...")
    
    # Create synthetic data
    batch_size = 2
    height = width = 56
    num_classes = 3
    
    # Create loss function
    loss_fn = HierarchicalLoss(
        bg_weight=1.0,
        fg_weight=1.0, 
        target_weight=1.0,
        consistency_weight=0.1,
        use_dynamic_weights=True
    )
    
    # Test case 1: Perfect background prediction
    print("\n1. Testing perfect background prediction:")
    predictions = torch.zeros(batch_size, num_classes, height, width)
    predictions[:, 0, :, :] = 10.0  # Strong background prediction
    predictions[:, 1:, :, :] = -10.0  # Weak foreground predictions
    
    targets = torch.zeros(batch_size, height, width, dtype=torch.long)  # All background
    
    aux_outputs = {
        'bg_fg_logits': torch.zeros(batch_size, 2, height, width),
        'target_nontarget_logits': torch.zeros(batch_size, 2, height, width),
        'fg_attention': torch.zeros(batch_size, 256, 28, 28)
    }
    aux_outputs['bg_fg_logits'][:, 0, :, :] = 10.0  # Strong background
    aux_outputs['bg_fg_logits'][:, 1, :, :] = -10.0  # Weak foreground
    
    loss, loss_dict = loss_fn(predictions, targets, aux_outputs)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss']:.4f}")
    print(f"  Should be low (near 0)")
    
    # Test case 2: Mixed prediction
    print("\n2. Testing mixed prediction (some background, some foreground):")
    targets_mixed = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets_mixed[:, :, height//2:] = 1  # Half target class
    
    loss, loss_dict = loss_fn(predictions, targets_mixed, aux_outputs)
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss']:.4f}")
    print(f"  Should be higher than case 1")
    
    # Test case 3: Check that ce_loss and dice_loss are in loss_dict
    print("\n3. Checking loss_dict contains required keys:")
    required_keys = ['ce_loss', 'dice_loss', 'total_loss', 'bg_fg_loss', 
                     'target_nontarget_loss', 'final_loss', 'consistency_loss']
    for key in required_keys:
        if key in loss_dict:
            print(f"  ✓ {key}: {loss_dict[key]:.4f}")
        else:
            print(f"  ✗ {key}: MISSING!")
    
    # Test case 4: Verify background IoU correlation
    print("\n4. Testing correlation between predictions and loss:")
    # Perfect prediction should have low loss
    perfect_pred = torch.zeros(batch_size, num_classes, height, width)
    perfect_pred[0, 0, :height//2, :] = 10.0  # Background where it should be
    perfect_pred[0, 1, height//2:, :] = 10.0   # Target where it should be
    perfect_pred[1, 0, :height//2, :] = 10.0  # Background where it should be
    perfect_pred[1, 1, height//2:, :] = 10.0   # Target where it should be
    
    perfect_aux = {
        'bg_fg_logits': torch.zeros(batch_size, 2, height, width),
        'target_nontarget_logits': torch.zeros(batch_size, 2, height, width),
        'fg_attention': torch.zeros(batch_size, 256, 28, 28)
    }
    perfect_aux['bg_fg_logits'][:, 0, :height//2, :] = 10.0  # Background correct
    perfect_aux['bg_fg_logits'][:, 1, height//2:, :] = 10.0   # Foreground correct
    perfect_aux['target_nontarget_logits'][:, 0, :, :] = 10.0  # All target in fg
    
    loss_perfect, _ = loss_fn(perfect_pred, targets_mixed, perfect_aux)
    
    # Bad prediction should have high loss
    bad_pred = torch.zeros(batch_size, num_classes, height, width)
    bad_pred[:, 0, :, :] = -10.0  # Weak background everywhere
    bad_pred[:, 1, :, :] = 10.0   # Strong target everywhere
    
    bad_aux = {
        'bg_fg_logits': torch.zeros(batch_size, 2, height, width),
        'target_nontarget_logits': torch.zeros(batch_size, 2, height, width),
        'fg_attention': torch.zeros(batch_size, 256, 28, 28)
    }
    bad_aux['bg_fg_logits'][:, 1, :, :] = 10.0  # All foreground (wrong)
    
    loss_bad, _ = loss_fn(bad_pred, targets_mixed, bad_aux)
    
    print(f"  Perfect prediction loss: {loss_perfect.item():.4f}")
    print(f"  Bad prediction loss: {loss_bad.item():.4f}")
    print(f"  Ratio (bad/perfect): {loss_bad.item()/loss_perfect.item():.2f}x")
    print(f"  Should be >> 1.0")
    
    # Test case 5: Dynamic weights
    print("\n5. Testing dynamic weight calculation:")
    # Create imbalanced data
    targets_imbalanced = torch.zeros(batch_size, height, width, dtype=torch.long)
    targets_imbalanced[:, :5, :5] = 1  # Very few target pixels
    
    _, loss_dict = loss_fn(predictions, targets_imbalanced, aux_outputs)
    print(f"  Background weight: {loss_dict.get('bg_weight', 'N/A'):.4f}")
    print(f"  Foreground weight: {loss_dict.get('fg_weight', 'N/A'):.4f}")
    print(f"  Target weight: {loss_dict.get('target_weight', 'N/A'):.4f}")
    print(f"  Non-target weight: {loss_dict.get('nontarget_weight', 'N/A'):.4f}")
    print("  Foreground weight should be much higher due to class imbalance")


def test_logit_combination():
    """Test the new logit combination method."""
    print("\n\nTesting Hierarchical Logit Combination...")
    
    from src.human_edge_detection.advanced.hierarchical_segmentation import HierarchicalSegmentationHead
    
    # Create model
    head = HierarchicalSegmentationHead(
        in_channels=256,
        mid_channels=256,
        num_classes=3,
        mask_size=56
    )
    
    # Test input
    batch_size = 2
    features = torch.randn(batch_size, 256, 28, 28)
    
    # Forward pass
    final_logits, aux_outputs = head(features)
    
    print(f"Final logits shape: {final_logits.shape}")
    print(f"Aux outputs keys: {list(aux_outputs.keys())}")
    
    # Verify the hierarchical structure
    bg_fg_logits = aux_outputs['bg_fg_logits']
    target_nontarget_logits = aux_outputs['target_nontarget_logits']
    
    # Check that background logit is preserved
    print("\nChecking hierarchical structure:")
    print(f"Background logits match: {torch.allclose(final_logits[:, 0], bg_fg_logits[:, 0])}")
    
    # Check that target and non-target are related to foreground
    fg_logit = bg_fg_logits[:, 1]
    scale_factor = 0.5
    expected_target = fg_logit + scale_factor * target_nontarget_logits[:, 0]
    expected_nontarget = fg_logit + scale_factor * target_nontarget_logits[:, 1]
    
    print(f"Target logits match expected: {torch.allclose(final_logits[:, 1], expected_target)}")
    print(f"Non-target logits match expected: {torch.allclose(final_logits[:, 2], expected_nontarget)}")
    
    # Test numerical stability
    print("\nTesting numerical stability:")
    # Large values shouldn't cause overflow
    features_large = torch.randn(batch_size, 256, 28, 28) * 100
    final_logits_large, _ = head(features_large)
    print(f"Max logit value: {final_logits_large.max().item():.2f}")
    print(f"Contains NaN: {torch.isnan(final_logits_large).any().item()}")
    print(f"Contains Inf: {torch.isinf(final_logits_large).any().item()}")


if __name__ == "__main__":
    test_hierarchical_loss()
    test_logit_combination()
    print("\n✓ All tests completed!")