"""UNet decoder with YOLO feature distillation for knowledge distillation training.

This module implements a UNet model that learns from both:
1. Teacher UNet's segmentation output
2. YOLOv9's intermediate feature representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
import segmentation_models_pytorch as smp
from ..feature_extractor import YOLOFeatureExtractor


class UNetWithYOLOFeatureDistillation(nn.Module):
    """UNet model with YOLO feature distillation capability.
    
    This model can learn from:
    - Teacher UNet's output (standard distillation)
    - YOLOv9's intermediate features (feature distillation)
    
    During inference, the projection layer is not used.
    """
    
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_weights: Optional[str] = "imagenet",
        freeze_encoder: bool = True,
        projection_hidden_dim: Optional[int] = 768  # Hidden dimension for projection
    ):
        """Initialize UNet with feature distillation.
        
        Args:
            encoder_name: Name of encoder architecture
            encoder_weights: Pretrained weights
            freeze_encoder: Whether to freeze encoder weights
            projection_hidden_dim: Hidden dimension for feature projection (None = direct projection)
        """
        super().__init__()
        
        # Create full UNet model
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.unet.encoder.parameters():
                param.requires_grad = False
        
        # Get encoder output channels at different levels
        # For EfficientNet-B0, the channels are typically:
        # [16, 24, 40, 112, 320] at different resolutions
        self.encoder_channels = self.unet.encoder.out_channels
        
        # The bottleneck (last encoder output) has the most channels
        # For B0: 320 channels at 80x80 resolution (after 8x downsampling)
        bottleneck_channels = self.encoder_channels[-1]  # 320 for B0
        
        # Feature projection layer (only used during training)
        # Projects from student bottleneck (320 for B0) to YOLO features (1024)
        if projection_hidden_dim is not None:
            self.feature_projector = nn.Sequential(
                nn.Conv2d(bottleneck_channels, projection_hidden_dim, kernel_size=1),
                nn.BatchNorm2d(projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(projection_hidden_dim, 1024, kernel_size=1)
            )
        else:
            # Direct projection
            self.feature_projector = nn.Conv2d(bottleneck_channels, 1024, kernel_size=1)
        
        # Initialize projection weights
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through UNet.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            return_features: If True, also return projected features for distillation
            
        Returns:
            If return_features=False: Segmentation output (B, 1, H, W)
            If return_features=True: (segmentation_output, projected_features)
        """
        # Encode features
        features = self.unet.encoder(x)
        
        # Get bottleneck features if needed for distillation
        if return_features:
            bottleneck = features[-1]  # Last encoder output (320x80x80 for B0)
            projected_features = self.feature_projector(bottleneck)  # Project to 1024x80x80
        
        # Decode to segmentation
        decoder_output = self.unet.decoder(*features)
        segmentation = self.unet.segmentation_head(decoder_output)
        
        if return_features:
            return segmentation, projected_features
        else:
            return segmentation
    
    def get_decoder_parameters(self):
        """Get decoder and projection parameters for optimization."""
        # Include both decoder and projection layer parameters
        params = list(self.unet.decoder.parameters()) + \
                 list(self.unet.segmentation_head.parameters()) + \
                 list(self.feature_projector.parameters())
        return params


class YOLOFeatureDistillationWrapper(nn.Module):
    """Wrapper for distillation training with YOLO features.
    
    This wrapper handles:
    1. Student model with feature projection
    2. Teacher UNet model
    3. YOLO feature extraction
    """
    
    def __init__(
        self,
        student_encoder: str = "timm-efficientnet-b0",
        teacher_checkpoint_path: str = "ext_extractor/2020-09-23a.pth",
        yolo_onnx_path: str = "ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx",
        yolo_target_layer: str = "segmentation_model_34_Concat_output_0",
        device: str = "cuda",
        freeze_teacher: bool = True,
        projection_hidden_dim: Optional[int] = 768
    ):
        """Initialize distillation wrapper with YOLO features.
        
        Args:
            student_encoder: Student model encoder name
            teacher_checkpoint_path: Path to teacher model checkpoint
            yolo_onnx_path: Path to YOLO ONNX model
            yolo_target_layer: YOLO layer to extract features from
            device: Device to use
            freeze_teacher: Whether to freeze teacher model
            projection_hidden_dim: Hidden dimension for projection
        """
        super().__init__()
        
        self.device = device
        
        # Initialize student model with projection capability
        self.student = UNetWithYOLOFeatureDistillation(
            encoder_name=student_encoder,
            encoder_weights="imagenet",
            freeze_encoder=True,  # Freeze student encoder too
            projection_hidden_dim=projection_hidden_dim
        ).to(device)
        
        # Load teacher model (same as train_distillation_staged.py)
        print(f"Loading teacher model from {teacher_checkpoint_path}")
        
        # Create teacher model with B3 encoder
        self.teacher = smp.Unet(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",  # Use ImageNet pretrained weights as base
            in_channels=3,
            classes=1,
            activation=None
        ).to(device)
        
        # Load teacher checkpoint
        checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.teacher.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.teacher.load_state_dict(checkpoint)
        
        # Freeze teacher
        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
        self.teacher.eval()
        
        # Initialize YOLO feature extractor
        print(f"Initializing YOLO feature extractor from {yolo_onnx_path}")
        self.yolo_extractor = YOLOFeatureExtractor(
            onnx_path=yolo_onnx_path,
            target_layer=yolo_target_layer,
            device="cuda" if device == "cuda" else "cpu"
        )
        
        # Cache for YOLO features (to avoid redundant extraction)
        self.yolo_features_cache = None
        self.cache_input_hash = None
    
    def extract_yolo_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract YOLO features with caching.
        
        Args:
            x: Input images (B, 3, H, W) - normalized with ImageNet stats
            
        Returns:
            YOLO features (B, 1024, 80, 80)
        """
        # Create a simple hash of the input to check if we need to recompute
        input_hash = (x.shape, x.device, x.data_ptr())
        
        if self.cache_input_hash == input_hash and self.yolo_features_cache is not None:
            return self.yolo_features_cache
        
        # Extract features using YOLO
        with torch.no_grad():
            yolo_features = self.yolo_extractor.extract_features(x)
        
        # Cache the features
        self.yolo_features_cache = yolo_features.detach()
        self.cache_input_hash = input_hash
        
        return self.yolo_features_cache
    
    def forward(self, x: torch.Tensor, extract_features: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass for distillation training.
        
        Args:
            x: Input images (B, 3, H, W)
            extract_features: Whether to extract features for distillation
            
        Returns:
            Dictionary containing:
                - student_output: Student segmentation output
                - teacher_output: Teacher segmentation output
                - student_features: Student projected features (if extract_features=True)
                - yolo_features: YOLO features (if extract_features=True)
        """
        # Teacher forward pass
        with torch.no_grad():
            teacher_output = self.teacher(x)
        
        # Student forward pass
        if extract_features:
            student_output, student_features = self.student(x, return_features=True)
            
            # Extract YOLO features
            yolo_features = self.extract_yolo_features(x)
            
            return {
                'student_output': student_output,
                'teacher_output': teacher_output,
                'student_features': student_features,
                'yolo_features': yolo_features
            }
        else:
            student_output = self.student(x, return_features=False)
            return {
                'student_output': student_output,
                'teacher_output': teacher_output
            }


class YOLODistillationLoss(nn.Module):
    """Combined loss for YOLO feature distillation.
    
    Combines:
    1. Output distillation loss (KL, MSE, BCE, Dice)
    2. Feature alignment loss (L2 or Cosine similarity)
    """
    
    def __init__(
        self,
        kl_weight: float = 1.0,
        mse_weight: float = 0.5,
        bce_weight: float = 0.5,
        dice_weight: float = 1.0,
        feature_weight: float = 1.0,
        feature_loss_type: str = "mse",  # "mse" or "cosine"
        temperature: float = 3.0
    ):
        """Initialize combined loss.
        
        Args:
            kl_weight: Weight for KL divergence loss
            mse_weight: Weight for MSE loss
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
            feature_weight: Weight for feature alignment loss
            feature_loss_type: Type of feature loss ("mse" or "cosine")
            temperature: Temperature for KL divergence
        """
        super().__init__()
        
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.feature_weight = feature_weight
        self.feature_loss_type = feature_loss_type
        self.temperature = temperature
        
        # Initialize loss functions
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
        """Compute Dice loss."""
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum(dim=(2, 3))
        union = pred_sigmoid.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - dice.mean()
    
    def feature_alignment_loss(self, student_features: torch.Tensor, yolo_features: torch.Tensor) -> torch.Tensor:
        """Compute feature alignment loss.
        
        Args:
            student_features: Student projected features (B, 1024, 80, 80)
            yolo_features: YOLO features (B, 1024, 80, 80)
            
        Returns:
            Feature alignment loss
        """
        if self.feature_loss_type == "mse":
            # Simple MSE loss
            return F.mse_loss(student_features, yolo_features)
        
        elif self.feature_loss_type == "cosine":
            # Cosine similarity loss
            # Flatten spatial dimensions
            student_flat = student_features.view(student_features.size(0), student_features.size(1), -1)  # B, C, H*W
            yolo_flat = yolo_features.view(yolo_features.size(0), yolo_features.size(1), -1)  # B, C, H*W
            
            # Normalize
            student_norm = F.normalize(student_flat, p=2, dim=1)
            yolo_norm = F.normalize(yolo_flat, p=2, dim=1)
            
            # Compute cosine similarity
            cosine_sim = (student_norm * yolo_norm).sum(dim=1).mean()  # Average over channels and spatial
            
            # Convert to loss (1 - similarity)
            return 1.0 - cosine_sim.mean()
        
        else:
            raise ValueError(f"Unknown feature loss type: {self.feature_loss_type}")
    
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        ground_truth: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.
        
        Args:
            outputs: Dictionary containing model outputs
            ground_truth: Ground truth masks (B, 1, H, W)
            
        Returns:
            Total loss and dictionary of individual losses
        """
        student_output = outputs['student_output']
        teacher_output = outputs['teacher_output']
        
        # Output distillation losses (same as before)
        # KL Divergence loss
        student_log_probs = F.log_softmax(student_output / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_output.detach() / self.temperature, dim=1)
        kl_loss = self.kl_loss(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # MSE loss
        mse_loss = self.mse_loss(student_output, teacher_output.detach())
        
        # BCE loss with ground truth
        bce_loss = self.bce_loss(student_output, ground_truth)
        
        # Dice loss with ground truth
        dice_loss = self.dice_loss(student_output, ground_truth)
        
        # Feature alignment loss (if features are provided)
        feature_loss = torch.tensor(0.0).to(student_output.device)
        if 'student_features' in outputs and 'yolo_features' in outputs:
            feature_loss = self.feature_alignment_loss(
                outputs['student_features'],
                outputs['yolo_features'].detach()  # Detach YOLO features
            )
        
        # Combine losses
        total_loss = (
            self.kl_weight * kl_loss +
            self.mse_weight * mse_loss +
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.feature_weight * feature_loss
        )
        
        # Return losses
        loss_dict = {
            'kl_loss': kl_loss.item(),
            'mse_loss': mse_loss.item(),
            'bce_loss': bce_loss.item(),
            'dice_loss': dice_loss.item(),
            'feature_loss': feature_loss.item(),
        }
        
        return total_loss, loss_dict


def create_yolo_distillation_model(
    student_encoder: str = "timm-efficientnet-b0",
    teacher_checkpoint: str = "ext_extractor/2020-09-23a.pth",
    yolo_onnx_path: str = "ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx",
    device: str = "cuda"
) -> Tuple[YOLOFeatureDistillationWrapper, YOLODistillationLoss]:
    """Create model and loss for YOLO feature distillation.
    
    Args:
        student_encoder: Student encoder architecture
        teacher_checkpoint: Path to teacher checkpoint
        yolo_onnx_path: Path to YOLO ONNX model
        device: Device to use
        
    Returns:
        Model wrapper and loss function
    """
    # Create model wrapper
    model = YOLOFeatureDistillationWrapper(
        student_encoder=student_encoder,
        teacher_checkpoint_path=teacher_checkpoint,
        yolo_onnx_path=yolo_onnx_path,
        device=device,
        freeze_teacher=True,
        projection_hidden_dim=768  # Use hidden layer for better projection
    )
    
    # Create loss function with balanced weights
    loss_fn = YOLODistillationLoss(
        kl_weight=1.0,
        mse_weight=0.5,
        bce_weight=0.5,
        dice_weight=1.0,
        feature_weight=0.5,  # Start with moderate feature weight
        feature_loss_type="mse",  # Use MSE for stability
        temperature=3.0
    )
    
    return model, loss_fn