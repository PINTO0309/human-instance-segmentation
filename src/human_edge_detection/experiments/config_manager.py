"""Configuration management for experiments."""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
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
    use_heavy_augmentation: bool = False  # If True, use heavy augmentation; if False, use light augmentation

    # Visualization
    use_roi_comparison: bool = False  # If True, show ROI Comparison row in visualization
    use_edge_visualize: bool = False  # If True, show edge detection results in visualization


@dataclass
class ModelConfig:
    """Model configuration."""
    onnx_model: str = 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx'
    num_classes: int = 3
    roi_size: Union[int, Tuple[int, int]] = 28  # Can be int for square or (height, width) for non-square
    mask_size: Union[int, Tuple[int, int]] = 56  # Can be int for square or (height, width) for non-square
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
    # Binary mask refinement modules
    use_boundary_refinement: bool = False  # Boundary Refinement Network (BRN)
    use_active_contour_loss: bool = False  # Active Contour Loss
    use_progressive_upsampling: bool = False  # Progressive Upsampling
    use_subpixel_conv: bool = False  # Sub-pixel Convolution
    use_contour_detection: bool = False  # Contour Detection Branch
    use_distance_transform: bool = False  # Distance Transform Prediction
    use_boundary_aware_loss: bool = False  # Boundary-aware Loss
    # Activation function configuration
    activation_function: str = 'relu'  # Options: 'relu', 'swish', 'gelu', 'silu'
    activation_beta: float = 1.0  # Beta parameter for Swish activation
    # Normalization configuration
    normalization_type: str = 'layernorm2d'  # Options: 'layernorm2d', 'batchnorm', 'instancenorm', 'groupnorm', 'mixed'
    normalization_groups: int = 8  # For GroupNorm
    normalization_mix_ratio: float = 0.5  # For MixedNorm (BatchNorm vs InstanceNorm ratio)
    # Pre-trained model configuration
    use_pretrained_unet: bool = False  # Use pre-trained UNet from people_segmentation
    pretrained_weights_path: str = ""  # Path to pre-trained weights
    freeze_pretrained_weights: bool = False  # Freeze pre-trained weights during training
    use_full_image_unet: bool = False  # Apply UNet to full image before ROI extraction


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

        'rgb_hierarchical_unet_v2_attention_r96m96': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m96',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:96',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=96,
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

        'rgb_hierarchical_unet_v2_attention_r80m96': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m96',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:96',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=96,
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

        'rgb_hierarchical_unet_v2_attention_r80m80': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m80',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:80',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=80,
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

        'rgb_hierarchical_unet_v2_attention_r64m96': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m96',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:96',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=96,
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

        'rgb_hierarchical_unet_v2_attention_r64m80': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m80',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:80',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=80,
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

        # RGB Hierarchical UNet V2 with Attention + Binary Mask Refinement
        'rgb_hierarchical_unet_v2_attention_r112m224_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m224_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:224 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=224,
                onnx_model=None,
                # Enable all refinement modules by default
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,  # Alternative to progressive upsampling
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",  # Use full stats for refined models
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r112m192_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m192_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:192 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=192,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r112m160_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m160_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:160 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=160,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r112m112_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r112m112_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:112, Mask:112 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=112,
                mask_size=112,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r96m192_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m192_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:192 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=192,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r96m160_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m160_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:160 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=160,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r96m112_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m112_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:112 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=112,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r96m96_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r96m96_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:96, Mask:96 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=96,
                mask_size=96,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r80m160_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m160_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:160 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=160,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r80m112_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m112_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:112 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=112,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r80m96_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m96_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:96 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=96,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r80m80_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r80m80_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:80, Mask:80 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=80,
                mask_size=80,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r64m112_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m112_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:112 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=112,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r64m96_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m96_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:96 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=96,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
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

        'rgb_hierarchical_unet_v2_attention_r64m80_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m80_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:80 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=80,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:64 with Binary Mask Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                use_boundary_refinement=True,
                use_active_contour_loss=True,
                use_progressive_upsampling=True,
                use_subpixel_conv=False,
                use_contour_detection=True,
                use_distance_transform=True,
                use_boundary_aware_loss=True,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=1e-5,  # Reduced from 5e-5
                warmup_epochs=5,    # Increased from 5
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,   # Reduced from 5.0
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001, # Reduced from 0.001
                min_lr=1e-7          # Reduced from 1e-6
            )
        ),

        ##################################################################################################
        # Stable refined configuration - start with minimal refinements
        ##################################################################################################
        'rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_boundaryrefinement': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_boundaryrefinement',
            description='RGB Hierarchical UNet V2 Attention - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Start with only boundary refinement
                use_boundary_refinement=True,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=True,
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            )
        ),
        ##################################################################################################
        # GroupNorm version of refined configuration
        ##################################################################################################
        'rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_groupnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_groupnorm',
            description='RGB Hierarchical UNet V2 Attention with GroupNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules (same as original)
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=True,
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # GroupNorm configuration
                normalization_type='groupnorm',
                normalization_groups=8,  # 8 groups is standard for GroupNorm
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_activecontourloss_distance_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules (same as original)
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=True,
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_distance_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_contour_distance_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=False,
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined_boundaryref_contour_distance_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_boundaryref_contour_distance_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules
                use_boundary_refinement=True,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=False,
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined_boundaryref_contour_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_boundaryref_contour_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules
                use_boundary_refinement=True,
                use_boundary_aware_loss=False,
                use_contour_detection=True,
                use_active_contour_loss=False,
                use_distance_transform=False,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        'rgb_hierarchical_unet_v2_attention_r64m64_refined_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64m64_refined_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64, Mask:64 with Stable Refinement',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=64,
                mask_size=64,
                onnx_model=None,
                # Refinement modules
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=False,
                use_active_contour_loss=False,
                use_distance_transform=False,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),

        # Non-square ROI configuration based on dataset statistics
        'rgb_hierarchical_unet_v2_attention_r64x48m64x48_refined_batchnorm': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_attention_r64x48m64x48_refined_batchnorm',
            description='RGB Hierarchical UNet V2 Attention with BatchNorm - ROI:64x48 (H:W), Mask:64x48 (H:W) - Optimized for dataset aspect ratio',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Refinement modules
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=False,
                use_active_contour_loss=False,
                use_distance_transform=False,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # BatchNorm configuration
                normalization_type='batchnorm',
                normalization_groups=8,  # Not used for BatchNorm but kept for compatibility
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),
        # Pre-trained people segmentation model variant
        'rgb_hierarchical_unet_v2_pretrained_peopleseg_r64x48m64x48': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_pretrained_peopleseg_r64x48m64x48',
            description='RGB Hierarchical UNet V2 with Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:64x48 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,  # Freeze the pre-trained UNet weights
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=False,
                use_active_contour_loss=False,
                use_distance_transform=False,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),
        # Pre-trained people segmentation model variant with frozen weights
        'rgb_hierarchical_unet_v2_pretrained_peopleseg_frozen_r64x48m64x48': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_pretrained_peopleseg_frozen_r64x48m64x48',
            description='RGB Hierarchical UNet V2 with Frozen Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:64x48 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,  # Freeze the pre-trained UNet weights
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False,
                use_boundary_aware_loss=False,
                use_contour_detection=False,
                use_active_contour_loss=False,
                use_distance_transform=False,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4
            ),
            training=TrainingConfig(
                learning_rate=5e-5,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.0001,
                min_lr=1e-7,
            ),
        ),


        ##############################################################################################
        # Full-image pre-trained people segmentation model variant
        ##############################################################################################
        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:64x48 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='relu',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
        ),
        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:128x96 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(128, 96),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='relu',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
        ),
        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:80x60 (H:W), Mask:160x120 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(80, 60),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(160, 120),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='relu',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
        ),



        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware_swish': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m64x48_disttrans_contdet_baware_swish',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:64x48 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='swish',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
        ),

        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r96x72m96x72_disttrans_contdet_baware_swish': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r96x72m96x72_disttrans_contdet_baware_swish',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:96x72 (H:W), Mask:96x72 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(96, 72),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(96, 72),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='swish',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
        ),

        'rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_swish': ExperimentConfig(
            name='rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_swish',
            description='RGB Hierarchical UNet V2 with Full-Image Pre-trained People Segmentation Model - ROI:64x48 (H:W), Mask:128x96 (H:W)',
            model=ModelConfig(
                use_rgb_hierarchical=True,
                use_external_features=False,
                use_attention_module=True,
                roi_size=(64, 48),  # (height, width) - matches dataset's natural aspect ratio
                mask_size=(128, 96),  # (height, width) - matches dataset's natural aspect ratio
                onnx_model=None,
                # Pre-trained model configuration
                use_pretrained_unet=True,
                pretrained_weights_path="ext_extractor/2020-09-23a.pth",
                freeze_pretrained_weights=True,
                use_full_image_unet=True,  # Apply UNet to full image first
                # Refinement modules (disabled for pre-trained model)
                use_boundary_refinement=False, # x
                use_boundary_aware_loss=True,
                use_contour_detection=True,
                use_active_contour_loss=False, # x
                use_distance_transform=True,
                use_progressive_upsampling=False,
                use_subpixel_conv=False,
                # Normalization type is ignored for pre-trained model
                normalization_type='batchnorm',
                normalization_groups=8,
                # Activation function configuration
                activation_function='swish',  # Options: 'relu', 'swish', 'gelu', 'silu'
                activation_beta=1.0,  # Beta parameter for Swish activation
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
                data_stats="data_analyze_full.json",
                roi_padding=0.0,
                num_workers=4,
                use_augmentation=True,
                use_heavy_augmentation=True,
                use_roi_comparison=False,
                use_edge_visualize=True,
            ),
            training=TrainingConfig(
                learning_rate=1e-4,
                warmup_epochs=5,
                scheduler='cosine',
                num_epochs=100,
                batch_size=2,
                gradient_clip=1.0,
                dice_weight=1.0,
                ce_weight=1.0,
                weight_decay=0.01,
                min_lr=1e-6,
            ),
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