#!/usr/bin/env python3
"""Test resume functionality when distillation is disabled."""

import torch
import tempfile
from pathlib import Path
from src.human_edge_detection.experiments.config_manager import ConfigManager
from src.human_edge_detection.advanced.unet_decoder_distillation import (
    create_unet_distillation_model, UNetDistillationLoss
)

def save_checkpoint(path, epoch, model, loss_fn, config):
    """Save a checkpoint with loss function state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.student.state_dict(),
        'loss_fn_state': {
            'performance_ratio': loss_fn.performance_ratio if hasattr(loss_fn, 'performance_ratio') else 1.0,
            'alpha': loss_fn.alpha if hasattr(loss_fn, 'alpha') else 0.0,
            'task_weight': loss_fn.task_weight if hasattr(loss_fn, 'task_weight') else 1.0,
            'temperature': loss_fn.temperature if hasattr(loss_fn, 'temperature') else 1.0,
            'distillation_eliminated': loss_fn.distillation_eliminated if hasattr(loss_fn, 'distillation_eliminated') else False,
        },
        'best_iou': 0.85,
        'teacher_miou_cache': None,  # No teacher for fine-tuning
        'config': config.__dict__,
    }
    torch.save(checkpoint, path)
    return checkpoint

def load_checkpoint(path, model, loss_fn):
    """Load checkpoint and restore loss function state."""
    checkpoint = torch.load(path, weights_only=False)
    
    # Load model
    model.student.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore loss function state
    if 'loss_fn_state' in checkpoint:
        loss_fn_state = checkpoint['loss_fn_state']
        if hasattr(loss_fn, 'performance_ratio'):
            loss_fn.performance_ratio = loss_fn_state.get('performance_ratio', 1.0)
        if hasattr(loss_fn, 'alpha'):
            loss_fn.alpha = loss_fn_state.get('alpha', 0.0)
        if hasattr(loss_fn, 'task_weight'):
            loss_fn.task_weight = loss_fn_state.get('task_weight', 1.0)
        if hasattr(loss_fn, 'temperature'):
            loss_fn.temperature = loss_fn_state.get('temperature', 1.0)
        if hasattr(loss_fn, 'distillation_eliminated'):
            loss_fn.distillation_eliminated = loss_fn_state.get('distillation_eliminated', False)
    
    return checkpoint

print("=" * 70)
print("TEST: Resume with Distillation Disabled (Pure Fine-tuning)")
print("=" * 70)

# Test 1: Create model with distillation disabled from start
print("\n1. Creating initial model with distillation disabled...")
config = ConfigManager.get_config('rgb_hierarchical_unet_v2_finetune_b7')

# Override for pure fine-tuning
if not config.distillation.enabled:
    config.distillation.teacher_checkpoint = None

model1, loss_fn1 = create_unet_distillation_model(
    student_encoder=config.distillation.student_encoder,
    teacher_encoder=config.distillation.teacher_encoder,
    teacher_checkpoint=config.distillation.teacher_checkpoint,
    device='cpu',
    progressive_unfreeze=False,
    adaptive_distillation=False
)

# Apply fine-tuning overrides
loss_fn1.alpha = 0.0
loss_fn1.initial_alpha = 0.0
loss_fn1.task_weight = 1.0
loss_fn1.initial_task_weight = 1.0
loss_fn1.adaptive_distillation = False
loss_fn1.distillation_eliminated = True

print(f"   Teacher is None: {model1.teacher is None}")
print(f"   Loss alpha: {loss_fn1.alpha}")
print(f"   Loss task_weight: {loss_fn1.task_weight}")
print(f"   Loss distillation_eliminated: {loss_fn1.distillation_eliminated}")

# Save checkpoint
with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    
    print("\n2. Saving checkpoint with distillation disabled state...")
    saved_state = save_checkpoint(checkpoint_path, epoch=10, model=model1, loss_fn=loss_fn1, config=config)
    print(f"   Saved loss_fn_state: {saved_state['loss_fn_state']}")
    
    # Create new model and loss function
    print("\n3. Creating new model for resume test...")
    model2, loss_fn2 = create_unet_distillation_model(
        student_encoder=config.distillation.student_encoder,
        teacher_encoder=config.distillation.teacher_encoder,
        teacher_checkpoint=config.distillation.teacher_checkpoint,
        device='cpu',
        progressive_unfreeze=False,
        adaptive_distillation=False
    )
    
    print(f"   Initial loss alpha: {loss_fn2.alpha}")
    print(f"   Initial loss task_weight: {loss_fn2.task_weight}")
    print(f"   Initial distillation_eliminated: {loss_fn2.distillation_eliminated}")
    
    # Load checkpoint
    print("\n4. Loading checkpoint and restoring state...")
    loaded_checkpoint = load_checkpoint(checkpoint_path, model2, loss_fn2)
    
    print(f"   Restored loss alpha: {loss_fn2.alpha}")
    print(f"   Restored loss task_weight: {loss_fn2.task_weight}")
    print(f"   Restored distillation_eliminated: {loss_fn2.distillation_eliminated}")
    print(f"   Epoch to resume from: {loaded_checkpoint['epoch'] + 1}")
    
    # Test forward pass
    print("\n5. Testing forward pass after resume...")
    x = torch.randn(1, 3, 64, 64)
    masks = torch.randint(0, 2, (1, 1, 64, 64)).float()
    
    student_out, teacher_out = model2(x)
    loss, loss_dict = loss_fn2(student_out, teacher_out, masks)
    
    print(f"   KL loss: {loss_dict['kl_loss']:.4f}")
    print(f"   MSE loss: {loss_dict['mse_loss']:.4f}")
    print(f"   BCE loss: {loss_dict['bce_loss']:.4f}")
    print(f"   Dice loss: {loss_dict['dice_loss']:.4f}")
    print(f"   Total loss: {loss_dict['total_loss']:.4f}")

print("\n" + "=" * 70)
print("TEST: Resume After Distillation Elimination")
print("=" * 70)

# Test 2: Model that had distillation but was eliminated
print("\n1. Creating model with distillation initially enabled...")
config2 = ConfigManager.get_config('rgb_hierarchical_unet_v2_distillation_b0_from_b3_temp_prog')

model3, loss_fn3 = create_unet_distillation_model(
    student_encoder=config2.distillation.student_encoder,
    teacher_encoder=config2.distillation.teacher_encoder,
    teacher_checkpoint=config2.distillation.teacher_checkpoint,
    device='cpu',
    progressive_unfreeze=False,
    adaptive_distillation=True
)

print(f"   Initial alpha: {loss_fn3.alpha}")
print(f"   Initial task_weight: {loss_fn3.task_weight}")

# Simulate distillation elimination
print("\n2. Simulating distillation elimination...")
loss_fn3.update_distillation_weight(
    student_iou=0.88,
    teacher_iou=0.82,
    zero_distillation_threshold=0.03
)

print(f"   Alpha after elimination: {loss_fn3.alpha}")
print(f"   Task weight after elimination: {loss_fn3.task_weight}")
print(f"   Distillation eliminated: {loss_fn3.distillation_eliminated}")

with tempfile.TemporaryDirectory() as tmpdir:
    checkpoint_path2 = Path(tmpdir) / "checkpoint_eliminated.pth"
    
    print("\n3. Saving checkpoint after elimination...")
    saved_state2 = save_checkpoint(checkpoint_path2, epoch=25, model=model3, loss_fn=loss_fn3, config=config2)
    print(f"   Saved loss_fn_state: {saved_state2['loss_fn_state']}")
    
    # Create new model with distillation enabled initially
    print("\n4. Creating new model with distillation enabled...")
    model4, loss_fn4 = create_unet_distillation_model(
        student_encoder=config2.distillation.student_encoder,
        teacher_encoder=config2.distillation.teacher_encoder,
        teacher_checkpoint=config2.distillation.teacher_checkpoint,
        device='cpu',
        progressive_unfreeze=False,
        adaptive_distillation=True
    )
    
    print(f"   Initial alpha: {loss_fn4.alpha}")
    print(f"   Initial distillation_eliminated: {loss_fn4.distillation_eliminated}")
    
    # Load checkpoint with eliminated state
    print("\n5. Loading checkpoint with eliminated state...")
    loaded_checkpoint2 = load_checkpoint(checkpoint_path2, model4, loss_fn4)
    
    print(f"   Restored alpha: {loss_fn4.alpha}")
    print(f"   Restored task_weight: {loss_fn4.task_weight}")
    print(f"   Restored distillation_eliminated: {loss_fn4.distillation_eliminated}")
    
    # Test that distillation stays eliminated
    print("\n6. Testing that distillation remains eliminated...")
    # Try to update weight (should stay at 0 if properly eliminated)
    loss_fn4.update_distillation_weight(
        student_iou=0.70,  # Lower than teacher
        teacher_iou=0.80,
        zero_distillation_threshold=0.03
    )
    
    print(f"   Alpha after update attempt: {loss_fn4.alpha}")
    print(f"   Still eliminated: {loss_fn4.distillation_eliminated}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Check results
success_count = 0
total_tests = 4

# Test 1: Pure fine-tuning resume
if loss_fn2.alpha == 0.0 and loss_fn2.task_weight == 1.0 and loss_fn2.distillation_eliminated:
    print("✓ Test 1 PASSED: Pure fine-tuning state correctly restored")
    success_count += 1
else:
    print("✗ Test 1 FAILED: Pure fine-tuning state not restored correctly")

# Test 2: KL/MSE are 0 after resume
if loss_dict['kl_loss'] == 0.0 and loss_dict['mse_loss'] == 0.0:
    print("✓ Test 2 PASSED: KL/MSE remain 0 after resume")
    success_count += 1
else:
    print("✗ Test 2 FAILED: KL/MSE are not 0 after resume")

# Test 3: Elimination state persists
if loss_fn4.distillation_eliminated and loss_fn4.alpha == 0.0:
    print("✓ Test 3 PASSED: Distillation elimination persists after resume")
    success_count += 1
else:
    print("✗ Test 3 FAILED: Distillation elimination not persistent")

# Test 4: Elimination cannot be reversed
if loss_fn4.alpha == 0.0 and loss_fn4.distillation_eliminated:
    print("✓ Test 4 PASSED: Distillation cannot be re-enabled after elimination")
    success_count += 1
else:
    print("✗ Test 4 FAILED: Distillation was incorrectly re-enabled")

print(f"\nTotal: {success_count}/{total_tests} tests passed")