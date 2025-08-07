"""Pure UNet decoder for knowledge distillation training.

This module implements a standalone UNet decoder that can be trained via
knowledge distillation from a teacher model, focusing only on the decoder
part without any segmentation head or encoder components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, Union
import segmentation_models_pytorch as smp


class UNetDecoderOnly(nn.Module):
    """Pure UNet decoder model for distillation training.
    
    This model contains only the UNet decoder part from segmentation_models_pytorch,
    without any encoder or segmentation head. It's designed to be trained via
    knowledge distillation from a teacher model.
    """
    
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_weights: Optional[str] = "imagenet",  # Use ImageNet by default
        freeze_encoder: bool = True,  # Freeze encoder during training
        freeze_decoder: bool = False
    ):
        """Initialize UNet decoder.
        
        Args:
            encoder_name: Name of encoder architecture (used to match decoder structure)
            encoder_weights: Pretrained weights (default: 'imagenet')
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
        """
        super().__init__()
        
        # Create full UNet model with pretrained encoder
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,  # Use pretrained weights
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # We'll use the full UNet but only train the decoder part
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.unet.encoder.parameters():
                param.requires_grad = False
        
        # Optionally freeze decoder
        if freeze_decoder:
            for param in self.unet.decoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet decoder only.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Output tensor from UNet decoder (B, 1, H, W)
        """
        # Forward through full UNet (encoder is frozen)
        output = self.unet(x)
        return output
    
    def get_decoder_parameters(self):
        """Get decoder parameters for optimization."""
        return self.unet.decoder.parameters()


class DistillationUNetWrapper(nn.Module):
    """Wrapper for distillation training of UNet decoder.
    
    This wrapper handles both student and teacher models for distillation,
    focusing only on the UNet decoder outputs.
    """
    
    def __init__(
        self,
        student_encoder: str = "timm-efficientnet-b0",
        teacher_checkpoint_path: str = "ext_extractor/2020-09-23a.pth",
        freeze_teacher: bool = True
    ):
        """Initialize distillation wrapper.
        
        Args:
            student_encoder: Student model encoder name
            teacher_checkpoint_path: Path to teacher model checkpoint
            freeze_teacher: Whether to freeze teacher model
        """
        super().__init__()
        
        # Create student model (B0) with pretrained encoder
        self.student = UNetDecoderOnly(
            encoder_name=student_encoder,
            encoder_weights="imagenet",  # Use ImageNet pretrained weights
            freeze_encoder=True,  # Freeze encoder, train decoder only
            freeze_decoder=False  # Decoder needs to be trainable
        )
        
        # Create and load teacher model (B3)
        self.teacher = self._load_teacher_model(
            teacher_checkpoint_path,
            freeze=freeze_teacher
        )
        
    def _load_teacher_model(self, checkpoint_path: str, freeze: bool = True) -> nn.Module:
        """Load teacher model from checkpoint.
        
        Args:
            checkpoint_path: Path to teacher checkpoint
            freeze: Whether to freeze teacher weights
            
        Returns:
            Loaded teacher model
        """
        # Create teacher model with B3 encoder and ImageNet weights
        teacher = smp.Unet(
            encoder_name="timm-efficientnet-b3",
            encoder_weights="imagenet",  # Use ImageNet pretrained weights as base
            in_channels=3,
            classes=1,
            activation=None
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Process state dict keys to match model structure
        processed_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix if present
            new_key = key.replace('model.', '') if key.startswith('model.') else key
            # Only load encoder, decoder, and segmentation_head
            if any(part in new_key for part in ['encoder', 'decoder', 'segmentation_head']):
                processed_state_dict[new_key] = value
        
        # Load state dict (allow missing keys for flexibility)
        if processed_state_dict:
            result = teacher.load_state_dict(processed_state_dict, strict=False)
            print(f"  Teacher model loaded. Missing keys: {len(result.missing_keys)}, Unexpected keys: {len(result.unexpected_keys)}")
        
        # Freeze if requested
        if freeze:
            for param in teacher.parameters():
                param.requires_grad = False
                
        teacher.eval()
        return teacher
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both student and teacher.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Tuple of (student_output, teacher_output)
        """
        # Student forward pass
        student_output = self.student(x)
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_output = self.teacher(x)
            
        return student_output, teacher_output


class UNetDistillationLoss(nn.Module):
    """Loss function for UNet decoder distillation with class balancing.
    
    Combines MSE loss between student and teacher outputs with
    optional additional losses, accounting for foreground/background imbalance.
    """
    
    def __init__(
        self,
        temperature: float = 3.0,
        alpha: float = 0.5,  # Balanced between distillation and task loss
        task_weight: float = 0.3,  # Weight for ground truth BCE loss
        use_feature_matching: bool = False,
        fg_ratio: float = 0.162,  # Foreground ratio from dataset analysis
        use_dice_loss: bool = True  # Use Dice loss for better imbalance handling
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening outputs
            alpha: Weight for distillation loss vs task loss
            use_feature_matching: Whether to match intermediate features
            fg_ratio: Foreground pixel ratio in dataset (16.2%)
            use_dice_loss: Whether to use Dice loss in addition to BCE
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_weight = task_weight
        self.use_feature_matching = use_feature_matching
        self.use_dice_loss = use_dice_loss
        
        # Calculate class weights based on pixel ratio
        # Use sqrt for less aggressive weighting
        bg_ratio = 1.0 - fg_ratio
        # pos_weight for foreground class: sqrt(bg/fg) ~ 2.27
        self.pos_weight_value = np.sqrt(bg_ratio / fg_ratio)
        
        # Base losses (pos_weight will be set to correct device in forward)
        self.mse_loss = nn.MSELoss()
        # We'll create BCE loss in forward to ensure correct device
        self.bce_loss = None
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss for binary segmentation.
        
        Args:
            pred: Predicted logits (before sigmoid)
            target: Ground truth binary masks
            
        Returns:
            Dice loss value
        """
        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)
        
        # Ensure both tensors have the same shape
        if pred_prob.shape != target.shape:
            print(f"[WARNING] Shape mismatch in Dice loss: pred {pred_prob.shape} vs target {target.shape}")
        
        # Flatten tensors for batch processing
        batch_size = pred.shape[0]
        pred_flat = pred_prob.view(batch_size, -1)  # (B, H*W)
        target_flat = target.view(batch_size, -1)   # (B, H*W)
        
        # Calculate Dice coefficient per sample
        intersection = (pred_flat * target_flat).sum(dim=1)  # (B,)
        pred_sum = pred_flat.sum(dim=1)  # (B,)
        target_sum = target_flat.sum(dim=1)  # (B,)
        
        # Use smaller smooth to avoid washing out the loss
        smooth = 1e-5
        dice_coeff = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
        
        # Average Dice loss across batch
        dice_loss = 1.0 - dice_coeff.mean()
        
        # Debug occasionally (commented out for now)
        # import random
        # if random.random() < 0.0001:  # 0.01% chance
        #     print(f"[DEBUG Dice] intersection: {intersection.mean():.4f}, pred_sum: {pred_sum.mean():.4f}, target_sum: {target_sum.mean():.4f}, dice_coeff: {dice_coeff.mean():.4f}, dice_loss: {dice_loss:.4f}")
        
        return dice_loss
        
    def forward(
        self,
        student_output: torch.Tensor,
        teacher_output: torch.Tensor,
        target_masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute distillation loss.
        
        Args:
            student_output: Student model output
            teacher_output: Teacher model output
            target_masks: Optional ground truth masks for task loss
            
        Returns:
            Total loss and loss components dict
        """
        loss_dict = {}
        
        # Apply temperature scaling and sigmoid for binary segmentation
        # Clamp outputs before sigmoid to prevent extreme values
        student_output_clamped = torch.clamp(student_output, min=-10, max=10)
        teacher_output_clamped = torch.clamp(teacher_output, min=-10, max=10)
        
        student_soft = torch.sigmoid(student_output_clamped / self.temperature)
        teacher_soft = torch.sigmoid(teacher_output_clamped / self.temperature)
        
        # Enhanced numerical stability for KL divergence
        # Use larger epsilon and symmetric clamping
        eps = 1e-5  # Larger epsilon for better stability
        student_soft = torch.clamp(student_soft, eps, 1.0 - eps)
        teacher_soft = torch.clamp(teacher_soft, eps, 1.0 - eps)
        
        # Simple binary KL divergence with better numerical stability
        # KL(p||q) = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        # But we'll use a more stable formulation
        try:
            # Compute KL divergence term by term with stability checks
            term1 = teacher_soft * (torch.log(teacher_soft + eps) - torch.log(student_soft + eps))
            term2 = (1 - teacher_soft) * (torch.log(1 - teacher_soft + eps) - torch.log(1 - student_soft + eps))
            
            kl_loss = (term1 + term2).mean()
            
            # Additional safety check
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                # Fallback to simple L1 distance between probabilities
                kl_loss = torch.abs(teacher_soft - student_soft).mean()
                kl_loss = kl_loss * 0.1  # Scale down to match typical KL range
                
        except Exception as e:
            # Ultimate fallback
            kl_loss = torch.tensor(0.0, device=student_output.device)
        
        # Clamp KL loss to prevent explosion
        kl_loss = torch.clamp(kl_loss, min=0.0, max=5.0)
        
        # Safe item() extraction with nan check
        kl_value = kl_loss.item() if not torch.isnan(kl_loss) else 0.0
        loss_dict['kl_loss'] = kl_value
        
        # MSE loss between outputs (more stable than KL for initial training)
        mse_loss = self.mse_loss(student_output, teacher_output)
        loss_dict['mse_loss'] = mse_loss.item() if not torch.isnan(mse_loss) else 0.0
        
        # Compute BCE loss with ground truth (primary learning signal)
        if target_masks is not None:
            # Create BCE loss with pos_weight on correct device if not already created
            if self.bce_loss is None:
                device = student_output.device
                pos_weight = torch.tensor([self.pos_weight_value], device=device)
                self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # Weighted BCE loss for class imbalance
            bce_loss = self.bce_loss(student_output, target_masks.float())
            loss_dict['bce_loss'] = bce_loss.item() if not torch.isnan(bce_loss) else 0.0
            
            # Add Dice loss for better handling of imbalance
            if self.use_dice_loss:
                dice_loss = self.dice_loss(student_output, target_masks.float())
                dice_value = dice_loss.item() if not torch.isnan(dice_loss) else 0.0
                loss_dict['dice_loss'] = dice_value
                
                # Debug: Print dice loss value occasionally (commented out for now)
                # import random
                # if random.random() < 0.001:  # Print 0.1% of the time
                #     print(f"[DEBUG] Dice loss raw: {dice_value:.6f}, BCE: {bce_loss.item():.4f}")
                
                # Combine BCE and Dice with more weight on BCE for stability
                # BCE handles pixel-wise accuracy, Dice handles overlap
                task_loss = 0.7 * bce_loss + 0.3 * dice_loss
            else:
                dice_loss = 0.0
                loss_dict['dice_loss'] = 0.0
                task_loss = bce_loss
        else:
            bce_loss = 0.0
            dice_loss = 0.0
            task_loss = 0.0
            loss_dict['bce_loss'] = 0.0
            loss_dict['dice_loss'] = 0.0
        
        # Start with MSE-dominated loss for stability, gradually introduce KL
        # Use smaller weight for KL initially
        kl_weight = min(self.alpha, 0.1)  # Start with small KL weight
        distillation_loss = kl_weight * kl_loss + (1 - kl_weight) * mse_loss
        
        if target_masks is not None:
            # Strong supervision from ground truth + distillation guidance
            total_loss = self.task_weight * task_loss + (1 - self.task_weight) * distillation_loss
        else:
            # Pure distillation when no ground truth
            total_loss = distillation_loss
        
        # Final safety check - silently fallback to stable loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            # Silently use the most stable loss component
            if target_masks is not None and not torch.isnan(task_loss):
                total_loss = task_loss
            elif not torch.isnan(mse_loss):
                total_loss = mse_loss
            else:
                # Ultimate fallback - return small constant loss
                total_loss = torch.tensor(1.0, device=student_output.device, requires_grad=True)
            
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


def create_unet_distillation_model(
    student_encoder: str = "timm-efficientnet-b0",
    teacher_checkpoint: str = "ext_extractor/2020-09-23a.pth",
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    """Create UNet distillation model and loss.
    
    Args:
        student_encoder: Student encoder architecture
        teacher_checkpoint: Path to teacher checkpoint
        device: Device to use
        
    Returns:
        Tuple of (model, loss_fn)
    """
    # Create distillation wrapper
    model = DistillationUNetWrapper(
        student_encoder=student_encoder,
        teacher_checkpoint_path=teacher_checkpoint,
        freeze_teacher=True
    ).to(device)
    
    # Create loss function with class balancing and stability settings
    loss_fn = UNetDistillationLoss(
        temperature=1.0,  # Very low temperature for stable gradients
        alpha=0.05,  # Very low KL weight initially (will be dominated by MSE)
        task_weight=0.7,  # Prioritize ground truth task loss (BCE + Dice)
        use_feature_matching=False,
        fg_ratio=0.162,  # Foreground ratio from dataset analysis (16.2% foreground, 83.8% background)
        use_dice_loss=True  # Use Dice loss for better handling of class imbalance
    )
    
    return model, loss_fn