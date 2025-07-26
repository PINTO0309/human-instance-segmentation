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
class AuxiliaryTaskConfig:
    """Configuration for auxiliary foreground/background task."""
    enabled: bool = False
    weight: float = 0.3  # Weight for auxiliary loss
    mid_channels: int = 128  # Channels for auxiliary head
    pos_weight: Optional[float] = None  # Positive class weight for binary loss
    visualize: bool = True  # Whether to visualize auxiliary predictions


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    batch_size: int = 8
    learning_rate: float = 1e-3
    num_epochs: int = 100

    # Optimizer
    optimizer: str = 'adamw'
    weight_decay: float = 1e-4

    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'cosine_warm_restarts', or None
    min_lr: float = 1e-6
    warmup_epochs: int = 5

    # CosineAnnealingWarmRestarts specific parameters
    T_0: int = 10  # Number of iterations for the first restart
    T_mult: int = 2  # A factor increases T_i after a restart
    eta_min_restart: float = 1e-6  # Minimum learning rate for warm restarts

    # Training options
    gradient_clip: float = 5.0
    mixed_precision: bool = True

    # Validation
    validate_every: int = 1
    save_every: int = 10
    early_stopping_patience: int = 10

    # Loss weights
    ce_weight: float = 1.0
    dice_weight: float = 1.0

    # Focal loss settings
    use_focal: bool = False
    focal_gamma: float = 2.0

@dataclass
class DataConfig:
    """Data configuration."""
    train_annotation: str = 'data/annotations/instances_train2017_person_only_no_crowd_500.json'
    val_annotation: str = 'data/annotations/instances_val2017_person_only_no_crowd_100.json'
    train_img_dir: str = 'data/images/train2017'
    val_img_dir: str = 'data/images/val2017'
    data_stats: str = 'data_analyze_full.json'

    # Data loading
    num_workers: int = 8
    pin_memory: bool = True

    # ROI settings
    roi_padding: float = 0.0  # ROI padding ratio (0.0 = no padding, 0.1 = 10% padding)

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
    execution_provider: str = 'tensorrt'
    variable_roi_sizes: Optional[Dict[str, int]] = None  # For variable ROI experiments
    use_rgb_enhancement: bool = False  # Whether to use RGB enhancement
    rgb_enhanced_layers: List[str] = field(default_factory=lambda: ['layer_34'])  # Which layers to enhance with RGB
    use_hierarchical: bool = False  # Use hierarchical segmentation architecture
    use_hierarchical_unet: bool = False  # Use UNet-based hierarchical segmentation (V1)
    use_hierarchical_unet_v2: bool = False  # V2: Enhanced UNet for bg/fg only
    use_hierarchical_unet_v3: bool = False  # V3: Enhanced UNet for bg/fg, Shallow UNet for target/non-target
    use_hierarchical_unet_v4: bool = False  # V4: Enhanced UNet for both branches
    use_class_specific_decoder: bool = False  # Use class-specific decoders
    use_external_features: bool = False  # If True, expect pre-extracted features instead of images
    use_rgb_hierarchical: bool = False  # Use RGB-based hierarchical model without YOLOv9 features
    use_attention_module: bool = False  # Use attention modules in target/non-target branch


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
    auxiliary_task: AuxiliaryTaskConfig = field(default_factory=AuxiliaryTaskConfig)

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
        if 'auxiliary_task' in data and isinstance(data['auxiliary_task'], dict):
            data['auxiliary_task'] = AuxiliaryTaskConfig(**data['auxiliary_task'])

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
            description='Baseline single-scale model',
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'multiscale': ExperimentConfig(
            name='multiscale',
            description='Multi-scale features only',
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_hires': ExperimentConfig(
            name='variable_roi_hires',
            description='Variable ROI with high-res layer using larger ROI',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 28, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive1': ExperimentConfig(
            name='variable_roi_progressive1',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive2': ExperimentConfig(
            name='variable_roi_progressive2',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 80, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive3': ExperimentConfig(
            name='variable_roi_progressive3',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 80, 'layer_22': 40, 'layer_34': 40}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive4': ExperimentConfig(
            name='variable_roi_progressive4',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 160, 'layer_22': 80, 'layer_34': 80}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_full.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive5': ExperimentConfig(
            name='variable_roi_progressive5',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_all1.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive6': ExperimentConfig(
            name='variable_roi_progressive6',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_151.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive7': ExperimentConfig(
            name='variable_roi_progressive7',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_121.json",
                roi_padding=0.0  # No padding
            ),
        ),

        'variable_roi_progressive8': ExperimentConfig(
            name='variable_roi_progressive8',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_131.json",
                roi_padding=0.0  # No padding
            )
        ),

        'variable_roi_progressive9': ExperimentConfig(
            name='variable_roi_progressive9',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_051520.json",
                roi_padding=0.0  # No padding
            )
        ),

        'variable_roi_progressive_focal': ExperimentConfig(
            name='variable_roi_progressive_focal',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
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
            ),
            data=DataConfig(
                data_stats="data_analyze_061512.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                use_focal=True,
                focal_gamma=3.0
            )
        ),

        'variable_roi_progressive_focal2': ExperimentConfig(
            name='variable_roi_progressive_focal2',
            description='Progressive ROI sizes across scales',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28}
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=True,
                boundary_width=3,
                boundary_weight=1.5,
                instance_sep_weight=2.0
            ),
            data=DataConfig(
                data_stats="data_analyze_081012.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                use_focal=True,
                focal_gamma=2.0
            )
        ),

        'hierarchical_segmentation': ExperimentConfig(
            name='hierarchical_segmentation',
            description='Hierarchical segmentation to prevent mode collapse',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28},
                use_hierarchical=True
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=False  # Use hierarchical loss instead
            ),
            data=DataConfig(
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                use_focal=False,  # Hierarchical loss handles class balance
                learning_rate=5e-4,
                batch_size=2
            )
        ),

        'hierarchical_segmentation_unet': ExperimentConfig(
            name='hierarchical_segmentation_unet',
            description='Hierarchical segmentation with UNet-based foreground/background separation',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28},
                use_hierarchical_unet=True
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=False  # Use hierarchical loss instead
            ),
            data=DataConfig(
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=2
            )
        ),

        'hierarchical_segmentation_unet_v2': ExperimentConfig(
            name='hierarchical_segmentation_unet_v2',
            description='V2: Enhanced UNet for bg/fg only, standard CNN for target/non-target',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28},
                use_hierarchical_unet_v2=True
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=False  # Use hierarchical loss instead
            ),
            data=DataConfig(
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=2,
                gradient_clip=10.0  # Increased for deeper UNet
            )
        ),

        'hierarchical_segmentation_unet_v3': ExperimentConfig(
            name='hierarchical_segmentation_unet_v3',
            description='V3: Enhanced UNet for bg/fg, Shallow UNet for target/non-target',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28},
                use_hierarchical_unet_v3=True
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=False  # Use hierarchical loss instead
            ),
            data=DataConfig(
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=3e-4,  # Lower LR for dual UNet
                batch_size=2,
                gradient_clip=10.0
            )
        ),

        'hierarchical_segmentation_unet_v4': ExperimentConfig(
            name='hierarchical_segmentation_unet_v4',
            description='V4: Enhanced UNet for both branches with cross-attention',
            model=ModelConfig(
                variable_roi_sizes={'layer_3': 56, 'layer_22': 42, 'layer_34': 28},
                use_hierarchical_unet_v4=True
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            distance_loss=DistanceLossConfig(
                enabled=False  # Use hierarchical loss instead
            ),
            data=DataConfig(
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0  # No padding
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=2e-4,  # Even lower LR for complex architecture
                batch_size=2,
                gradient_clip=15.0,  # Higher clip for cross-attention
                mixed_precision=False  # Disable for stability with cross-attention
            )
        ),


        # External features versions - these accept pre-extracted YOLO features
        'hierarchical_unet_v1_external_adaptive': ExperimentConfig(
            name='hierarchical_unet_v1_external_adaptive',
            description='Hierarchical UNet V1 with external features and adaptive fusion',
            model=ModelConfig(
                use_hierarchical_unet=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=4,
                gradient_clip=10.0
            )
        ),
        'hierarchical_unet_v1_external_concat': ExperimentConfig(
            name='hierarchical_unet_v1_external_concat',
            description='Hierarchical UNet V1 with external features and concat fusion',
            model=ModelConfig(
                use_hierarchical_unet=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=4,
                gradient_clip=10.0
            )
        ),
        'hierarchical_unet_v1_external_sum': ExperimentConfig(
            name='hierarchical_unet_v1_external_sum',
            description='Hierarchical UNet V1 with external features and sum fusion',
            model=ModelConfig(
                use_hierarchical_unet=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='sum'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=4,
                gradient_clip=10.0
            )
        ),

        # Hierarchical UNet V2 - Low learning rate test
        'hierarchical_unet_v2_external_concat_low_lr': ExperimentConfig(
            name='hierarchical_unet_v2_external_concat_low_lr',
            description='Hierarchical UNet V2 with external features, concat fusion, and lower learning rate',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                learning_rate=5e-5,  # Reduced for stability
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,  # Reduced for stability
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,  # Increased 10x to reduce overfitting
                min_lr=1e-6,
                T_0=20,  # Increased for longer cycles
                early_stopping_patience=15  # Increased patience
            )
        ),

        # Improved version with stability fixes
        'hierarchical_unet_v2_external_concat_stabilized': ExperimentConfig(
            name='hierarchical_unet_v2_external_concat_stabilized',
            description='Hierarchical UNet V2 with stability improvements (dropout, EMA weights, better regularization)',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                learning_rate=5e-5,  # Moderate learning rate
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,  # Reduced for stability
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,  # Strong regularization
                min_lr=1e-6,
                T_0=20,  # Longer cosine cycles
                early_stopping_patience=15,
                mixed_precision=True
            )
        ),

        # Hierarchical UNet V2 with auxiliary fg/bg task
        'hierarchical_unet_v2_auxiliary': ExperimentConfig(
            name='hierarchical_unet_v2_auxiliary',
            description='Hierarchical UNet V2 with auxiliary foreground/background task',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=0,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6,
                mixed_precision=True
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                pos_weight=2.0,  # Foreground is less frequent
                visualize=True
            )
        ),

        'hierarchical_unet_v2_external_adaptive': ExperimentConfig(
            name='hierarchical_unet_v2_external_adaptive',
            description='Hierarchical UNet V2 with external features and adaptive fusion',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='adaptive'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=4,
                gradient_clip=10.0
            )
        ),
        'hierarchical_unet_v2_external_concat': ExperimentConfig(
            name='hierarchical_unet_v2_external_concat',
            description='Hierarchical UNet V2 with external features and concat fusion',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=1e-4,
                batch_size=4,
                gradient_clip=10.0,
                warmup_epochs=10  # Increased from default 5
            )
        ),
        'hierarchical_unet_v2_external_sum': ExperimentConfig(
            name='hierarchical_unet_v2_external_sum',
            description='Hierarchical UNet V2 with external features and sum fusion',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='sum'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-4,
                batch_size=4,
                gradient_clip=10.0
            )
        ),

        # Test configurations with fixed weights
        'hierarchical_unet_v2_fixed_weights': ExperimentConfig(
            name='hierarchical_unet_v2_fixed_weights',
            description='Hierarchical UNet V2 with fixed class weights (no dynamic balancing)',
            model=ModelConfig(
                use_hierarchical_unet_v2=True,
                use_external_features=True,
                roi_size=28,
                mask_size=56
            ),
            multiscale=MultiScaleConfig(
                enabled=True,
                target_layers=['layer_3', 'layer_22', 'layer_34'],
                fusion_method='concat'
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0
            ),
            training=TrainingConfig(
                num_epochs=100,
                learning_rate=5e-5,  # Even lower learning rate
                batch_size=4,
                gradient_clip=10.0,
                warmup_epochs=15  # Longer warmup
            )
        ),




        'rgb_hierarchical_unet_v2': ExperimentConfig(
            name='rgb_hierarchical_unet_v2',
            description='RGB-based Hierarchical UNet V2 without YOLOv9 features',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,  # We use RGB images directly
                roi_size=28,
                mask_size=56,
                onnx_model=None  # No YOLO model needed
            ),
            multiscale=MultiScaleConfig(
                enabled=False,  # Start with single-scale
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_100.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,  # Add some padding for context
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=50,
                batch_size=4,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),

        # RGB Hierarchical UNet V2 with Attention Modules
        'rgb_hierarchical_unet_v2_attention': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention',
            description='RGB-based Hierarchical UNet V2 with Attention Modules',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,  # Enable attention modules
                roi_size=28,
                mask_size=56,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        # ROI and Mask Size Optimization Experiments
        # ROIAlign: 112 series
        'rgb_hierarchical_unet_v2_attention_r112m224': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m224',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:224',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=224,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r112m192': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m192',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:192',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=192,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r112m160': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m160',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:160',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=160,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r112m112': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m112',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:112',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=112,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        # ROIAlign: 96 series
        'rgb_hierarchical_unet_v2_attention_r96m192': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m192',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:192',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=192,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r96m160': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m160',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:160',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=160,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r96m112': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m112',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:112',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=112,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        # ROIAlign: 80 series
        'rgb_hierarchical_unet_v2_attention_r80m160': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m160',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:160',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=160,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r80m112': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m112',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:112',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=112,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        # ROIAlign: 64 series
        'rgb_hierarchical_unet_v2_attention_r64m112': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m112',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:112',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=112,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
        
        'rgb_hierarchical_unet_v2_attention_r64m64': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:64',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
            ),
            multiscale=MultiScaleConfig(
                enabled=False,
                target_layers=None,
                fusion_method='concat'
            ),
            auxiliary_task=AuxiliaryTaskConfig(
                enabled=True,
                weight=0.3,
                mid_channels=128,
                visualize=True
            ),
            data=DataConfig(
                train_annotation="data/annotations/instances_train2017_person_only_no_crowd_500.json",
                val_annotation="data/annotations/instances_val2017_person_only_no_crowd_100.json",
                data_stats="data_analyze_no_separation.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=5.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.001,
                min_lr=1e-6
            )
        ),
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