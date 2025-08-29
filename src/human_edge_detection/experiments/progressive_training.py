"""Progressive training strategies for advanced models."""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json

from ..experiments.config_manager import ExperimentConfig, ConfigManager


class ProgressiveTrainingSchedule:
    """Schedule for progressive feature activation."""
    
    def __init__(
        self,
        base_epochs: int = 10,
        feature_schedule: Optional[Dict[str, int]] = None
    ):
        """Initialize progressive training schedule.
        
        Args:
            base_epochs: Epochs for baseline training
            feature_schedule: Dict mapping feature names to activation epochs
        """
        self.base_epochs = base_epochs
        
        # Default schedule
        if feature_schedule is None:
            self.feature_schedule = {
                'baseline': 0,
                'multiscale': base_epochs,
                'distance_loss': base_epochs * 2,
                'cascade': base_epochs * 3,
                'relational': base_epochs * 4
            }
        else:
            self.feature_schedule = feature_schedule
            
    def get_active_features(self, epoch: int) -> List[str]:
        """Get list of active features for given epoch."""
        active = []
        
        for feature, start_epoch in self.feature_schedule.items():
            if epoch >= start_epoch:
                active.append(feature)
                
        return active
    
    def should_activate(self, feature: str, epoch: int) -> bool:
        """Check if a feature should be activated."""
        return epoch >= self.feature_schedule.get(feature, float('inf'))
    
    def get_config_for_epoch(self, epoch: int, base_config: str = 'baseline') -> ExperimentConfig:
        """Get configuration for specific epoch."""
        config = ConfigManager.get_config(base_config)
        
        # Activate features based on schedule
        active_features = self.get_active_features(epoch)
        
        config.multiscale.enabled = 'multiscale' in active_features
        config.distance_loss.enabled = 'distance_loss' in active_features
        config.cascade.enabled = 'cascade' in active_features
        config.relational.enabled = 'relational' in active_features
        
        return config


class ProgressiveModelBuilder:
    """Build models progressively with feature transfer."""
    
    @staticmethod
    def transfer_weights(
        source_model: nn.Module,
        target_model: nn.Module,
        strict: bool = False
    ) -> Dict[str, bool]:
        """Transfer weights from source to target model.
        
        Args:
            source_model: Source model
            target_model: Target model
            strict: Whether to require exact match
            
        Returns:
            Dictionary indicating which weights were transferred
        """
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        transferred = {}
        
        for name, param in source_dict.items():
            if name in target_dict:
                if param.shape == target_dict[name].shape:
                    target_dict[name] = param
                    transferred[name] = True
                else:
                    print(f"Shape mismatch for {name}: {param.shape} vs {target_dict[name].shape}")
                    transferred[name] = False
            else:
                if not strict:
                    # Try to find matching layer with different name
                    for target_name in target_dict:
                        if ProgressiveModelBuilder._match_layer_names(name, target_name):
                            if param.shape == target_dict[target_name].shape:
                                target_dict[target_name] = param
                                transferred[name] = True
                                break
                            
        target_model.load_state_dict(target_dict)
        
        print(f"Transferred {sum(transferred.values())}/{len(source_dict)} parameters")
        
        return transferred
    
    @staticmethod
    def _match_layer_names(source_name: str, target_name: str) -> bool:
        """Check if layer names match (allowing for module prefix differences)."""
        # Remove common prefixes
        source_parts = source_name.split('.')
        target_parts = target_name.split('.')
        
        # Check if core parts match
        if len(source_parts) >= 2 and len(target_parts) >= 2:
            return source_parts[-2:] == target_parts[-2:]
            
        return False
    
    @staticmethod
    def adapt_optimizer(
        old_optimizer: torch.optim.Optimizer,
        new_model: nn.Module,
        new_params_lr_scale: float = 1.0
    ) -> torch.optim.Optimizer:
        """Adapt optimizer for new model architecture.
        
        Args:
            old_optimizer: Previous optimizer
            new_model: New model
            new_params_lr_scale: Learning rate scale for new parameters
            
        Returns:
            New optimizer
        """
        # Get optimizer class and base config
        optimizer_class = type(old_optimizer)
        old_state = old_optimizer.state_dict()
        base_lr = old_state['param_groups'][0]['lr']
        
        # Identify new parameters
        old_param_names = set()
        for group in old_state['param_groups']:
            old_param_names.update(group['params'])
            
        # Create parameter groups
        transferred_params = []
        new_params = []
        
        for name, param in new_model.named_parameters():
            if param.requires_grad:
                if any(name.startswith(old_name) for old_name in old_param_names):
                    transferred_params.append(param)
                else:
                    new_params.append(param)
                    
        # Create optimizer with different learning rates
        param_groups = [
            {'params': transferred_params, 'lr': base_lr}
        ]
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': base_lr * new_params_lr_scale
            })
            
        # Create new optimizer
        if optimizer_class == torch.optim.AdamW:
            new_optimizer = optimizer_class(
                param_groups,
                weight_decay=old_state['param_groups'][0].get('weight_decay', 0)
            )
        else:
            new_optimizer = optimizer_class(param_groups)
            
        return new_optimizer


class WarmupSchedule:
    """Warmup schedule for new features."""
    
    def __init__(
        self,
        warmup_epochs: int = 5,
        warmup_factor: float = 0.1
    ):
        """Initialize warmup schedule.
        
        Args:
            warmup_epochs: Number of warmup epochs
            warmup_factor: Initial learning rate factor
        """
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        
    def get_lr_scale(self, epoch: int, feature_start_epoch: int) -> float:
        """Get learning rate scale for warmup.
        
        Args:
            epoch: Current epoch
            feature_start_epoch: Epoch when feature was activated
            
        Returns:
            Learning rate scale factor
        """
        epochs_since_start = epoch - feature_start_epoch
        
        if epochs_since_start >= self.warmup_epochs:
            return 1.0
            
        # Linear warmup
        return self.warmup_factor + (1.0 - self.warmup_factor) * (
            epochs_since_start / self.warmup_epochs
        )


def save_progressive_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    config: ExperimentConfig,
    metrics: Dict[str, float],
    schedule: ProgressiveTrainingSchedule,
    checkpoint_dir: Path
) -> Path:
    """Save checkpoint with progressive training info."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'config': config.to_dict(),
        'metrics': metrics,
        'progressive_schedule': {
            'base_epochs': schedule.base_epochs,
            'feature_schedule': schedule.feature_schedule,
            'active_features': schedule.get_active_features(epoch)
        }
    }
    
    # Save with feature info in filename
    active_features = schedule.get_active_features(epoch)
    feature_str = '_'.join(active_features) if active_features else 'baseline'
    
    checkpoint_path = checkpoint_dir / f'progressive_epoch_{epoch:04d}_{feature_str}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_progressive_checkpoint(
    checkpoint_path: Path,
    device: str = 'cuda'
) -> Tuple[Dict, ProgressiveTrainingSchedule]:
    """Load progressive training checkpoint."""
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct schedule
    schedule_info = checkpoint.get('progressive_schedule', {})
    schedule = ProgressiveTrainingSchedule(
        base_epochs=schedule_info.get('base_epochs', 10),
        feature_schedule=schedule_info.get('feature_schedule', {})
    )
    
    return checkpoint, schedule


if __name__ == "__main__":
    # Test progressive training schedule
    print("Testing ProgressiveTrainingSchedule...")
    
    schedule = ProgressiveTrainingSchedule(base_epochs=10)
    
    for epoch in [0, 5, 10, 15, 20, 25, 30, 35, 40]:
        active = schedule.get_active_features(epoch)
        print(f"Epoch {epoch}: {active}")
        
    # Test config generation
    print("\nTesting progressive config generation...")
    config_epoch_20 = schedule.get_config_for_epoch(20)
    print(f"Epoch 20 config:")
    print(f"  Multiscale: {config_epoch_20.multiscale.enabled}")
    print(f"  Distance loss: {config_epoch_20.distance_loss.enabled}")
    print(f"  Cascade: {config_epoch_20.cascade.enabled}")
    
    # Test warmup schedule
    print("\nTesting WarmupSchedule...")
    warmup = WarmupSchedule(warmup_epochs=5, warmup_factor=0.1)
    
    feature_start = 10
    for epoch in range(10, 16):
        scale = warmup.get_lr_scale(epoch, feature_start)
        print(f"Epoch {epoch}: LR scale = {scale:.2f}")