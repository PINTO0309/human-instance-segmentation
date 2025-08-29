"""Staged training utilities for progressive model training."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StageConfig:
    """Configuration for a training stage."""
    name: str
    start_epoch: int
    end_epoch: Optional[int]  # None means until the end
    freeze_encoder: bool
    freeze_decoder: bool
    freeze_rgb_extractor: bool
    learning_rate_scale: float = 1.0


def get_default_staged_config() -> List[StageConfig]:
    """Get default staged training configuration.
    
    Returns:
        List of stage configurations
    """
    return [
        StageConfig(
            name="encoder_only",
            start_epoch=0,
            end_epoch=50,
            freeze_encoder=False,  # Train encoder
            freeze_decoder=True,    # Freeze decoder
            freeze_rgb_extractor=True,  # Freeze RGB extractor
            learning_rate_scale=1.0
        ),
        StageConfig(
            name="full_model",
            start_epoch=50,
            end_epoch=None,
            freeze_encoder=False,  # Train encoder
            freeze_decoder=False,  # Train decoder
            freeze_rgb_extractor=False,  # Train RGB extractor
            learning_rate_scale=0.5  # Reduce learning rate
        )
    ]


def get_current_stage(epoch: int, stages: List[StageConfig]) -> Optional[StageConfig]:
    """Get the current training stage based on epoch.
    
    Args:
        epoch: Current epoch number
        stages: List of stage configurations
        
    Returns:
        Current stage configuration or None
    """
    for stage in stages:
        if stage.start_epoch <= epoch:
            if stage.end_epoch is None or epoch < stage.end_epoch:
                return stage
    return None


def apply_stage_freezing(
    model: nn.Module,
    stage: StageConfig,
    verbose: bool = True
) -> Dict[str, int]:
    """Apply parameter freezing based on stage configuration.
    
    Args:
        model: Model to configure
        stage: Stage configuration
        verbose: Whether to print status
        
    Returns:
        Dictionary with counts of frozen/trainable parameters
    """
    stats = {
        'encoder_params': 0,
        'encoder_trainable': 0,
        'decoder_params': 0,
        'decoder_trainable': 0,
        'rgb_params': 0,
        'rgb_trainable': 0,
        'other_params': 0,
        'other_trainable': 0
    }
    
    # Handle distillation wrapper
    if hasattr(model, 'student'):
        actual_model = model.student
    else:
        actual_model = model
    
    # Process each module
    for name, module in actual_model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip non-leaf modules
            
        # Determine module type and apply freezing
        if 'pretrained_unet' in name:
            # Encoder part (UNet)
            for param in module.parameters():
                param.requires_grad = not stage.freeze_encoder
                stats['encoder_params'] += param.numel()
                if param.requires_grad:
                    stats['encoder_trainable'] += param.numel()
                    
        elif 'rgb_feature_extractor' in name:
            # RGB feature extractor
            for param in module.parameters():
                param.requires_grad = not stage.freeze_rgb_extractor
                stats['rgb_params'] += param.numel()
                if param.requires_grad:
                    stats['rgb_trainable'] += param.numel()
                    
        elif 'segmentation_head' in name or 'feature_combiner' in name:
            # Decoder part
            for param in module.parameters():
                param.requires_grad = not stage.freeze_decoder
                stats['decoder_params'] += param.numel()
                if param.requires_grad:
                    stats['decoder_trainable'] += param.numel()
                    
        else:
            # Other parameters (ROI align, etc.)
            for param in module.parameters():
                # Keep these always trainable
                param.requires_grad = True
                stats['other_params'] += param.numel()
                stats['other_trainable'] += param.numel()
    
    if verbose:
        print(f"\n=== Stage: {stage.name} ===")
        print(f"Encoder: {stats['encoder_trainable']:,}/{stats['encoder_params']:,} params trainable")
        print(f"RGB Extractor: {stats['rgb_trainable']:,}/{stats['rgb_params']:,} params trainable")
        print(f"Decoder: {stats['decoder_trainable']:,}/{stats['decoder_params']:,} params trainable")
        print(f"Other: {stats['other_trainable']:,}/{stats['other_params']:,} params trainable")
        
        total_params = sum([stats[k] for k in stats if k.endswith('_params')])
        total_trainable = sum([stats[k] for k in stats if k.endswith('_trainable')])
        print(f"Total: {total_trainable:,}/{total_params:,} params trainable ({100*total_trainable/total_params:.1f}%)")
    
    return stats


def update_optimizer_for_stage(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    stage: StageConfig,
    base_lr: float
) -> torch.optim.Optimizer:
    """Update optimizer parameters for the current stage.
    
    Args:
        optimizer: Current optimizer
        model: Model
        stage: Stage configuration
        base_lr: Base learning rate
        
    Returns:
        Updated optimizer
    """
    # Get only parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    # Update learning rate
    new_lr = base_lr * stage.learning_rate_scale
    
    # Create new optimizer with updated parameters
    if isinstance(optimizer, torch.optim.AdamW):
        new_optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=new_lr,
            weight_decay=optimizer.param_groups[0]['weight_decay']
        )
    elif isinstance(optimizer, torch.optim.Adam):
        new_optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=new_lr
        )
    else:
        # Default to Adam
        new_optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=new_lr
        )
    
    # Copy state from old optimizer where possible
    try:
        new_optimizer.load_state_dict(optimizer.state_dict())
    except:
        pass  # State might not be compatible after freezing changes
    
    return new_optimizer


def configure_staged_training(
    config_dict: Dict,
    enable: bool = True,
    encoder_only_epochs: int = 50,
    lr_scale_after_encoder: float = 0.5
) -> Dict:
    """Add staged training configuration to experiment config.
    
    Args:
        config_dict: Experiment configuration dictionary
        enable: Whether to enable staged training
        encoder_only_epochs: Number of epochs for encoder-only training
        lr_scale_after_encoder: Learning rate scale factor after encoder training
        
    Returns:
        Updated configuration dictionary
    """
    config_dict['staged_training'] = {
        'enabled': enable,
        'stages': [
            {
                'name': 'encoder_only',
                'start_epoch': 0,
                'end_epoch': encoder_only_epochs,
                'freeze_encoder': False,
                'freeze_decoder': True,
                'freeze_rgb_extractor': True,
                'learning_rate_scale': 1.0
            },
            {
                'name': 'full_model',
                'start_epoch': encoder_only_epochs,
                'end_epoch': None,
                'freeze_encoder': False,
                'freeze_decoder': False,
                'freeze_rgb_extractor': False,
                'learning_rate_scale': lr_scale_after_encoder
            }
        ]
    }
    
    return config_dict