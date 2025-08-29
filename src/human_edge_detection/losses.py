"""Loss functions for ROI-based instance segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class DiceLoss(nn.Module):
    """Dice loss for segmentation tasks."""
    
    def __init__(
        self,
        smooth: float = 1e-6,
        reduction: str = 'mean',
        apply_softmax: bool = True
    ):
        """Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
            apply_softmax: Whether to apply softmax to predictions
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.apply_softmax = apply_softmax
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        class_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """Compute Dice loss.
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Target labels (B, H, W) with values in [0, C-1]
            class_indices: Specific classes to compute Dice for (default: all except background)
            
        Returns:
            Dice loss value
        """
        num_classes = predictions.shape[1]
        
        # Apply softmax if needed
        if self.apply_softmax:
            predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()
        
        # Select classes to compute Dice for
        if class_indices is None:
            # Default: all classes except background (class 0)
            class_indices = list(range(1, num_classes))
        
        dice_losses = []
        
        for class_idx in class_indices:
            pred_class = predictions[:, class_idx, :, :]
            target_class = targets_one_hot[:, class_idx, :, :]
            
            # Compute intersection and union
            intersection = torch.sum(pred_class * target_class, dim=(1, 2))
            pred_sum = torch.sum(pred_class, dim=(1, 2))
            target_sum = torch.sum(target_class, dim=(1, 2))
            
            # Compute Dice coefficient
            dice_coeff = (2 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
            
            # Convert to loss
            dice_loss = 1 - dice_coeff
            
            if self.reduction == 'mean':
                dice_loss = dice_loss.mean()
            elif self.reduction == 'sum':
                dice_loss = dice_loss.sum()
            
            dice_losses.append(dice_loss)
        
        # Average over classes
        total_dice_loss = torch.stack(dice_losses).mean()
        
        return total_dice_loss


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """Initialize Focal loss.
        
        Args:
            alpha: Class weights (C,)
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal loss."""
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class SegmentationLoss(nn.Module):
    """Combined loss for instance segmentation.
    
    Combines weighted cross-entropy and Dice loss.
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_classes: Optional[List[int]] = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0
    ):
        """Initialize segmentation loss.
        
        Args:
            class_weights: Weights for each class (C,)
            ce_weight: Weight for cross-entropy loss
            dice_weight: Weight for Dice loss
            dice_classes: Classes to compute Dice for (default: [1] for target class only)
            use_focal: Whether to use focal loss instead of cross-entropy
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()
        
        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_classes = dice_classes if dice_classes is not None else [1]  # Default: target class only
        self.use_focal = use_focal
        
        # Initialize loss components
        if use_focal:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        self.dice_loss = DiceLoss()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute combined loss.
        
        Args:
            predictions: Predicted logits (B, C, H, W)
            targets: Target labels (B, H, W)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with individual loss components
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        # Dice loss
        dice_loss = self.dice_loss(predictions, targets, self.dice_classes)
        
        # Combined loss
        total_loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'dice_loss': dice_loss
        }
        
        return total_loss, loss_dict


def create_loss_function(
    pixel_ratios: dict,
    use_log_weights: bool = True,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    dice_classes: Optional[List[int]] = None,
    device: str = 'cpu',
    separation_aware_weights: Optional[dict] = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0
) -> SegmentationLoss:
    """Create loss function with appropriate class weights.
    
    Args:
        pixel_ratios: Dictionary with pixel ratios for each class
        use_log_weights: Whether to use log-scaled weights (more moderate)
        ce_weight: Weight for cross-entropy component
        dice_weight: Weight for Dice component
        dice_classes: Classes to compute Dice for
        device: Device to put weights on
        separation_aware_weights: Optional pre-computed separation-aware weights
        use_focal: Whether to use focal loss instead of cross-entropy
        focal_gamma: Gamma parameter for focal loss
        
    Returns:
        SegmentationLoss instance
    """
    # Use pre-computed separation-aware weights if provided
    if separation_aware_weights is not None:
        weights = separation_aware_weights
        print("Using pre-computed separation-aware weights")
    else:
        # Calculate class weights based on pixel ratios
        eps = 1e-3
        
        if use_log_weights:
            # Log-scaled weights (more moderate)
            weights = {
                'background': torch.log(torch.tensor(1.0 / (pixel_ratios['background'] + eps))),
                'target': torch.log(torch.tensor(1.0 / (pixel_ratios['target'] + eps))),
                'non_target': torch.log(torch.tensor(1.0 / (pixel_ratios['non_target'] + eps)))
            }
        else:
            # Inverse frequency weights
            weights = {
                'background': 1.0 / (pixel_ratios['background'] + eps),
                'target': 1.0 / (pixel_ratios['target'] + eps),
                'non_target': 1.0 / (pixel_ratios['non_target'] + eps)
            }
        
        # Normalize weights
        weight_sum = sum(weights.values())
        weights = {k: v / weight_sum * 3.0 for k, v in weights.items()}
    
    # Convert to tensor
    class_weights = torch.tensor([
        weights['background'],
        weights['target'],
        weights['non_target']
    ], dtype=torch.float32).to(device)
    
    print(f"Using class weights: {class_weights.tolist()}")
    
    # Create loss function
    loss_fn = SegmentationLoss(
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        dice_classes=dice_classes,
        use_focal=use_focal,
        focal_gamma=focal_gamma
    )
    
    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 2
    num_classes = 3
    height, width = 56, 56
    
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test individual losses
    print("\nTesting Dice Loss:")
    dice_loss = DiceLoss()
    dice_value = dice_loss(predictions, targets, class_indices=[1, 2])
    print(f"Dice loss: {dice_value.item():.4f}")
    
    print("\nTesting Cross-Entropy Loss:")
    ce_loss = nn.CrossEntropyLoss()
    ce_value = ce_loss(predictions, targets)
    print(f"CE loss: {ce_value.item():.4f}")
    
    # Test combined loss with class weights from data analysis
    print("\nTesting Combined Loss with class weights:")
    pixel_ratios = {
        'background': 0.485,
        'target': 0.393,
        'non_target': 0.122
    }
    
    loss_fn = create_loss_function(pixel_ratios, use_log_weights=True)
    total_loss, loss_dict = loss_fn(predictions, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"CE loss component: {loss_dict['ce_loss'].item():.4f}")
    print(f"Dice loss component: {loss_dict['dice_loss'].item():.4f}")