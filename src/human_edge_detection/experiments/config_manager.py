"""Configuration management for experiments."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
import copy


@dataclass
class MultiScaleConfig:
    """Configuration for multi-scale features."""
    enabled: bool = False
    target_layers: List[str] = field(default_factory=lambda: ['layer_3', 'layer_22', 'layer_34'])
    fusion_method: str = 'adaptive'  # 'adaptive', 'concat', 'sum'
    fusion_channels: int = 256


@dataclass
class DistanceLossConfig:
    """Configuration for distance-aware loss."""
    enabled: bool = False
    boundary_width: int = 5
    boundary_weight: float = 2.0
    instance_sep_weight: float = 3.0
    adaptive: bool = False
    adaptation_rate: float = 0.01


@dataclass
class CascadeConfig:
    """Configuration for cascade segmentation."""
    enabled: bool = False
    num_stages: int = 3
    stage_weights: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    share_features: bool = True


@dataclass
class RelationalConfig:
    """Configuration for relational reasoning."""
    enabled: bool = False
    num_heads: int = 8
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 100
    
    # Optimizer
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4
    
    # Scheduler
    scheduler: str = 'cosine'
    min_lr: float = 1e-6
    warmup_epochs: int = 5
    
    # Training options
    gradient_clip: float = 5.0
    mixed_precision: bool = True
    
    # Validation
    validate_every: int = 1
    save_every: int = 5
    early_stopping_patience: int = 10
    
    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0


@dataclass
class DataConfig:
    """Data configuration."""
    train_annotation: str = 'data/annotations/instances_train2017_person_only_no_crowd.json'
    val_annotation: str = 'data/annotations/instances_val2017_person_only_no_crowd.json'
    train_img_dir: str = 'data/images/train2017'
    val_img_dir: str = 'data/images/val2017'
    data_stats: str = 'data_analyze_full.json'
    
    # Data loading
    num_workers: int = 8
    pin_memory: bool = True
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_prob: float = 0.5


@dataclass
class ModelConfig:
    """Model configuration."""
    onnx_model: str = 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'
    num_classes: int = 3
    roi_size: int = 28
    mask_size: int = 56
    execution_provider: str = 'cuda'


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    name: str
    description: str = ""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    multiscale: MultiScaleConfig = field(default_factory=MultiScaleConfig)
    distance_loss: DistanceLossConfig = field(default_factory=DistanceLossConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    relational: RelationalConfig = field(default_factory=RelationalConfig)
    
    # Output settings
    output_dir: str = 'experiments'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        # Handle nested dataclasses
        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'data' in data and isinstance(data['data'], dict):
            data['data'] = DataConfig(**data['data'])
        if 'training' in data and isinstance(data['training'], dict):
            data['training'] = TrainingConfig(**data['training'])
        if 'multiscale' in data and isinstance(data['multiscale'], dict):
            data['multiscale'] = MultiScaleConfig(**data['multiscale'])
        if 'distance_loss' in data and isinstance(data['distance_loss'], dict):
            data['distance_loss'] = DistanceLossConfig(**data['distance_loss'])
        if 'cascade' in data and isinstance(data['cascade'], dict):
            data['cascade'] = CascadeConfig(**data['cascade'])
        if 'relational' in data and isinstance(data['relational'], dict):
            data['relational'] = RelationalConfig(**data['relational'])
            
        return cls(**data)
    
    def save(self, path: str):
        """Save configuration to file."""
        path = Path(path)
        data = self.to_dict()
        
        if path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load configuration from file."""
        path = Path(path)
        
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
            
        return cls.from_dict(data)


class ConfigManager:
    """Manager for experiment configurations."""
    
    # Predefined configurations
    CONFIGS = {
        'baseline': ExperimentConfig(
            name='baseline',
            description='Baseline single-scale model'
        ),
        
        'multiscale': ExperimentConfig(
            name='multiscale',
            description='Multi-scale features only',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            )
        ),
        
        'multiscale_distance': ExperimentConfig(
            name='multiscale_distance',
            description='Multi-scale + distance-aware loss',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=True,
                boundary_width=5,
                boundary_weight=2.0,
                instance_sep_weight=3.0
            )
        ),
        
        'multiscale_cascade': ExperimentConfig(
            name='multiscale_cascade',
            description='Multi-scale + cascade',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            cascade=CascadeConfig(
                enabled=True,
                num_stages=3,
                stage_weights=[0.3, 0.3, 0.4]
            )
        ),
        
        'full': ExperimentConfig(
            name='full',
            description='All features enabled',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=True,
                boundary_width=5,
                boundary_weight=2.0,
                instance_sep_weight=3.0,
                adaptive=True
            ),
            cascade=CascadeConfig(
                enabled=True,
                num_stages=3,
                stage_weights=[0.3, 0.3, 0.4]
            )
        ),
        
        'efficient': ExperimentConfig(
            name='efficient',
            description='Efficient configuration with fewer layers',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_5', 'layer_34'],  # Only 2 scales
                fusion_method='sum'  # Simpler fusion
            ),
            distance_loss=DistanceLossConfig(
                enabled=True,
                boundary_width=3,  # Smaller boundary
                boundary_weight=1.5,
                instance_sep_weight=2.0
            )
        )
    }
    
    @classmethod
    def get_config(cls, name: str) -> ExperimentConfig:
        """Get a predefined configuration."""
        if name not in cls.CONFIGS:
            raise ValueError(f"Unknown config: {name}. Available: {list(cls.CONFIGS.keys())}")
            
        # Return a deep copy to avoid mutations
        return copy.deepcopy(cls.CONFIGS[name])
    
    @classmethod
    def list_configs(cls) -> List[str]:
        """List available configurations."""
        return list(cls.CONFIGS.keys())
    
    @classmethod
    def create_custom_config(
        cls,
        base_config: str,
        modifications: Dict[str, Any]
    ) -> ExperimentConfig:
        """Create a custom config based on a predefined one."""
        config = cls.get_config(base_config)
        
        # Apply modifications
        for key, value in modifications.items():
            if '.' in key:
                # Nested attribute
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                setattr(config, key, value)
                
        return config


def create_experiment_dirs(config: ExperimentConfig) -> Dict[str, Path]:
    """Create experiment directories."""
    base_dir = Path(config.output_dir) / config.name
    
    dirs = {
        'base': base_dir,
        'checkpoints': base_dir / 'checkpoints',
        'logs': base_dir / 'logs',
        'configs': base_dir / 'configs',
        'visualizations': base_dir / 'visualizations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
        
    return dirs


if __name__ == "__main__":
    # Test configuration management
    print("Testing ConfigManager...")
    
    # List available configs
    print(f"Available configs: {ConfigManager.list_configs()}")
    
    # Get baseline config
    baseline = ConfigManager.get_config('baseline')
    print(f"\nBaseline config: {baseline.name}")
    print(f"  Multiscale enabled: {baseline.multiscale.enabled}")
    print(f"  Distance loss enabled: {baseline.distance_loss.enabled}")
    
    # Get full config
    full = ConfigManager.get_config('full')
    print(f"\nFull config: {full.name}")
    print(f"  Multiscale enabled: {full.multiscale.enabled}")
    print(f"  Distance loss enabled: {full.distance_loss.enabled}")
    print(f"  Cascade enabled: {full.cascade.enabled}")
    
    # Create custom config
    custom = ConfigManager.create_custom_config(
        'multiscale',
        {
            'name': 'custom_experiment',
            'training.batch_size': 32,
            'multiscale.fusion_method': 'concat'
        }
    )
    print(f"\nCustom config: {custom.name}")
    print(f"  Batch size: {custom.training.batch_size}")
    print(f"  Fusion method: {custom.multiscale.fusion_method}")
    
    # Save and load config
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        custom.save(f.name)
        loaded = ExperimentConfig.load(f.name)
        print(f"\nLoaded config: {loaded.name}")
        print(f"  Matches original: {loaded.name == custom.name}")