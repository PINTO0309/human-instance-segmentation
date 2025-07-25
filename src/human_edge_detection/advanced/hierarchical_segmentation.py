"""Hierarchical segmentation architecture for stable multi-class learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..model import LayerNorm2d, ResidualBlock


class HierarchicalSegmentationHead(nn.Module):
    """Hierarchical segmentation head that separates background vs foreground from target vs non-target."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head.

        Args:
            in_channels: Input feature channels
            mid_channels: Intermediate channel count
            num_classes: Total number of classes (should be 3)
            mask_size: Output mask size
        """
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"

        self.num_classes = num_classes
        self.mask_size = mask_size

        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
        )

        # Branch 1: Background vs Foreground (binary)
        self.bg_vs_fg_branch = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, 2, 1)  # 2 classes: bg, fg
        )

        # Branch 2: Target vs Non-target (within foreground)
        self.target_vs_nontarget_branch = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, 2, 1)  # 2 classes: target, non-target
        )

        # Gating mechanism to focus target/non-target branch on foreground regions
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical prediction.

        Args:
            features: Input features (N, C, H, W)

        Returns:
            final_logits: Combined 3-class logits (N, 3, H, W)
            aux_outputs: Dictionary with intermediate predictions
        """
        # Shared feature extraction
        shared = self.shared_features(features)

        # Branch 1: Background vs Foreground
        bg_fg_logits = self.bg_vs_fg_branch(shared)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)

        # Create foreground attention gate
        fg_attention = self.fg_gate(bg_fg_logits)

        # Downsample attention to match shared features size
        # bg_fg_logits is 56x56, but shared is 28x28
        # Use fixed downsampling to avoid dynamic condition in ONNX export
        fg_attention = F.interpolate(fg_attention, size=shared.shape[-2:], mode='bilinear', align_corners=False)

        # Branch 2: Target vs Non-target (modulated by foreground attention)
        gated_features = shared * fg_attention
        target_nontarget_logits = self.target_vs_nontarget_branch(gated_features)

        # Combine predictions hierarchically
        # Method: Use hierarchical probability decomposition
        # P(background) = P(bg|bg_vs_fg)
        # P(target) = P(fg|bg_vs_fg) * P(target|fg)
        # P(non-target) = P(fg|bg_vs_fg) * P(non-target|fg)

        # Get probabilities from both branches
        bg_prob = bg_fg_probs[:, 0:1]  # P(background)
        fg_prob = bg_fg_probs[:, 1:2]  # P(foreground)
        target_nontarget_probs = F.softmax(target_nontarget_logits, dim=1)

        # For numerical stability, work in log space instead of probability space
        # This avoids log(0) issues and provides more stable gradients
        
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_size, self.mask_size,
                                  device=features.device)
        
        # Get log probabilities
        log_bg_fg_probs = F.log_softmax(bg_fg_logits, dim=1)
        log_target_nontarget_probs = F.log_softmax(target_nontarget_logits, dim=1)
        
        # Method 1: Direct hierarchical decomposition
        # This maintains the hierarchical structure while being numerically stable
        
        # Background logit remains from the bg/fg branch
        final_logits[:, 0] = bg_fg_logits[:, 0]
        
        # For target and non-target, we need to ensure they compete only in foreground regions
        # We'll use the foreground logit as a base and add target/non-target distinctions
        fg_logit = bg_fg_logits[:, 1]
        
        # Scale target/non-target logits by a factor to control their influence
        # This prevents the model from ignoring the target/non-target branch
        scale_factor = 0.5
        final_logits[:, 1] = fg_logit + scale_factor * target_nontarget_logits[:, 0]
        final_logits[:, 2] = fg_logit + scale_factor * target_nontarget_logits[:, 1]

        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention
        }

        return final_logits, aux_outputs


class HierarchicalLoss(nn.Module):
    """Loss function for hierarchical segmentation."""

    def __init__(
        self,
        bg_weight: float = 1.0,
        fg_weight: float = 1.0,
        target_weight: float = 1.0,
        consistency_weight: float = 0.1,
        use_dynamic_weights: bool = True,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0
    ):
        """Initialize hierarchical loss.

        Args:
            bg_weight: Weight for background vs foreground loss
            fg_weight: Weight for target vs non-target loss
            target_weight: Additional weight for target class
            consistency_weight: Weight for consistency between branches
            use_dynamic_weights: Whether to use dynamic class balancing
            dice_weight: Weight for dice loss component
            ce_weight: Weight for cross-entropy component
        """
        super().__init__()
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight
        self.target_weight = target_weight
        self.consistency_weight = consistency_weight
        self.use_dynamic_weights = use_dynamic_weights
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Initialize Dice loss for target class
        from ..losses import DiceLoss
        self.dice_loss = DiceLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        aux_outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute hierarchical loss.

        Args:
            predictions: Final 3-class predictions (N, 3, H, W)
            targets: Ground truth labels (N, H, W)
            aux_outputs: Auxiliary outputs from hierarchical head

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses
        """
        # Create binary masks for hierarchical supervision
        bg_mask = (targets == 0).long()
        fg_mask = (targets > 0).long()
        target_mask = (targets == 1).long()
        nontarget_mask = (targets == 2).long()

        # Loss 1: Background vs Foreground
        bg_fg_targets = fg_mask  # 0 for background, 1 for foreground

        if self.use_dynamic_weights:
            # Calculate balanced weights based on class frequency
            bg_count = bg_mask.float().sum()
            fg_count = fg_mask.float().sum()
            total_count = bg_count + fg_count

            # Avoid division by zero and create balanced weights
            bg_weight = total_count / (2 * bg_count.clamp(min=1))
            fg_weight = total_count / (2 * fg_count.clamp(min=1))

            # Apply additional scaling for foreground importance
            fg_weight = fg_weight * self.target_weight

            # Clamp weights to reasonable ranges to prevent instability
            bg_weight = bg_weight.clamp(min=0.5, max=3.0)
            fg_weight = fg_weight.clamp(min=0.5, max=3.0)
            
            # Apply exponential moving average for stability
            if hasattr(self, '_ema_bg_weight'):
                alpha = 0.9  # EMA coefficient
                self._ema_bg_weight = alpha * self._ema_bg_weight + (1 - alpha) * bg_weight.item()
                self._ema_fg_weight = alpha * self._ema_fg_weight + (1 - alpha) * fg_weight.item()
            else:
                self._ema_bg_weight = bg_weight.item()
                self._ema_fg_weight = fg_weight.item()

            # Log weight changes for debugging (will be in loss_dict)
            self._last_bg_weight = self._ema_bg_weight
            self._last_fg_weight = self._ema_fg_weight

            bg_fg_loss = F.cross_entropy(
                aux_outputs['bg_fg_logits'],
                bg_fg_targets,
                weight=torch.tensor([self._ema_bg_weight, self._ema_fg_weight]).to(predictions.device)
            )
        else:
            # Use fixed weights
            self._last_bg_weight = 1.0
            self._last_fg_weight = self.target_weight

            bg_fg_loss = F.cross_entropy(
                aux_outputs['bg_fg_logits'],
                bg_fg_targets,
                weight=torch.tensor([1.0, self.target_weight]).to(predictions.device)
            )

        # Loss 2: Target vs Non-target (only on foreground pixels)
        if fg_mask.any():
            # Create targets for foreground pixels: 0 for target, 1 for non-target
            target_nontarget_targets = torch.zeros_like(targets)
            target_nontarget_targets[target_mask.bool()] = 0
            target_nontarget_targets[nontarget_mask.bool()] = 1

            # Calculate dynamic weights for target vs non-target
            target_count = (target_mask * fg_mask).float().sum()
            nontarget_count = (nontarget_mask * fg_mask).float().sum()
            fg_total = target_count + nontarget_count

            if fg_total > 0:
                if self.use_dynamic_weights:
                    # Dynamic balancing weights
                    target_weight_dynamic = fg_total / (2 * target_count.clamp(min=1))
                    nontarget_weight_dynamic = fg_total / (2 * nontarget_count.clamp(min=1))
                    
                    # Clamp weights to reasonable ranges
                    target_weight_dynamic = target_weight_dynamic.clamp(min=0.5, max=3.0)
                    nontarget_weight_dynamic = nontarget_weight_dynamic.clamp(min=0.5, max=3.0)
                    
                    # Apply exponential moving average for stability
                    if hasattr(self, '_ema_target_weight_tn'):
                        alpha = 0.9  # EMA coefficient
                        self._ema_target_weight_tn = alpha * self._ema_target_weight_tn + (1 - alpha) * target_weight_dynamic.item()
                        self._ema_nontarget_weight_tn = alpha * self._ema_nontarget_weight_tn + (1 - alpha) * nontarget_weight_dynamic.item()
                    else:
                        self._ema_target_weight_tn = target_weight_dynamic.item()
                        self._ema_nontarget_weight_tn = nontarget_weight_dynamic.item()

                    # Apply class weights
                    class_weights = torch.tensor([self._ema_target_weight_tn, self._ema_nontarget_weight_tn]).to(predictions.device)

                    # Log for debugging
                    self._last_target_weight = self._ema_target_weight_tn
                    self._last_nontarget_weight = self._ema_nontarget_weight_tn
                else:
                    # Use fixed weights
                    class_weights = torch.tensor([1.0, 1.0]).to(predictions.device)
                    self._last_target_weight = 1.0
                    self._last_nontarget_weight = 1.0

                # Compute weighted loss with foreground mask
                target_nontarget_loss = F.cross_entropy(
                    aux_outputs['target_nontarget_logits'],
                    target_nontarget_targets.long(),
                    weight=class_weights,
                    reduction='none'
                )
                # Apply foreground mask and average
                target_nontarget_loss = (target_nontarget_loss * fg_mask.float()).sum() / fg_mask.float().sum().clamp(min=1)
            else:
                target_nontarget_loss = torch.tensor(0.0, device=predictions.device)
        else:
            target_nontarget_loss = torch.tensor(0.0, device=predictions.device)

        # Loss 3: Final 3-class cross entropy
        final_loss = F.cross_entropy(predictions, targets)

        # Loss 4: Consistency regularization
        # The hierarchical decomposition should be consistent
        # Check that P(bg) + P(target) + P(non-target) = 1 (implicitly satisfied)
        # Instead, ensure foreground predictions sum correctly
        bg_fg_probs = F.softmax(aux_outputs['bg_fg_logits'], dim=1)
        final_probs = F.softmax(predictions, dim=1)

        # Foreground probability should match sum of target and non-target
        fg_from_final = final_probs[:, 1] + final_probs[:, 2]
        fg_from_branch = bg_fg_probs[:, 1]

        consistency_loss = F.mse_loss(fg_from_branch, fg_from_final)

        # Compute Dice loss for target class
        dice_loss_value = self.dice_loss(predictions, targets, class_indices=[1])
        
        # Combine losses
        total_loss = (
            self.bg_weight * bg_fg_loss +
            self.fg_weight * target_nontarget_loss +
            self.ce_weight * final_loss +
            self.dice_weight * dice_loss_value +
            self.consistency_weight * consistency_loss
        )

        loss_dict = {
            'bg_fg_loss': bg_fg_loss.item(),
            'target_nontarget_loss': target_nontarget_loss.item(),
            'final_loss': final_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'total_loss': total_loss.item(),
            # For compatibility with train_advanced.py
            'ce_loss': final_loss.item(),  # Cross-entropy is the main classification loss
            'dice_loss': dice_loss_value.item(),  # Now includes actual Dice loss
            # Debug weights
            'bg_weight': getattr(self, '_last_bg_weight', 1.0),
            'fg_weight': getattr(self, '_last_fg_weight', 1.0),
            'target_weight': getattr(self, '_last_target_weight', 1.0),
            'nontarget_weight': getattr(self, '_last_nontarget_weight', 1.0)
        }

        return total_loss, loss_dict


def create_hierarchical_model(base_model: nn.Module) -> nn.Module:
    """Wrap a base model with hierarchical segmentation head.

    Args:
        base_model: Base segmentation model

    Returns:
        Model with hierarchical segmentation
    """
    class HierarchicalSegmentationModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

            # Replace the segmentation head
            if hasattr(base_model, 'segmentation_head'):
                # Get the input channels from the original head
                old_head = base_model.segmentation_head
                if hasattr(old_head, 'decoder'):
                    # Find the first conv layer to get input channels
                    first_layer = old_head.decoder[0]
                    if isinstance(first_layer, ResidualBlock):
                        in_channels = first_layer.conv1.in_channels
                    else:
                        in_channels = 256  # Default
                else:
                    in_channels = 256

                # Create hierarchical head
                self.hierarchical_head = HierarchicalSegmentationHead(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )

                # Keep original components for feature extraction
                if hasattr(old_head, 'var_roi_align'):
                    self.roi_align = old_head.var_roi_align
                if hasattr(old_head, 'feature_fusion'):
                    self.feature_fusion = old_head.feature_fusion

        def forward(self, features, rois, rgb_features=None):
            # Handle both tensor (from base model) and dict (from multiscale) inputs
            if isinstance(features, torch.Tensor):
                # Images passed in - need to extract features first
                # Call base model's feature extraction
                if hasattr(self.base_model, 'extractor'):
                    # Variable ROI model has integrated extractor
                    features = self.base_model.extractor.extract_features(features)
                else:
                    # For other models, features should already be extracted
                    raise RuntimeError("Hierarchical model needs feature extractor")

            # Now features is a dict
            # Extract ROI features using original logic
            if hasattr(self, 'roi_align'):
                roi_features = self.roi_align(features, rois)

                # Handle RGB features if present
                if rgb_features is not None and hasattr(self.base_model, 'segmentation_head'):
                    old_head = self.base_model.segmentation_head
                    if hasattr(old_head, 'rgb_roi_aligns'):
                        # Process RGB features for enhanced layers
                        for layer_name in old_head.rgb_enhanced_layers:
                            if layer_name in roi_features and layer_name in old_head.rgb_roi_aligns:
                                rgb_roi_features = old_head.rgb_roi_aligns[layer_name](
                                    rgb_features, rois,
                                    old_head.rgb_roi_sizes[layer_name],
                                    old_head.rgb_roi_sizes[layer_name]
                                )
                                rgb_roi_features = old_head.rgb_projections[layer_name](rgb_roi_features)
                                roi_features[layer_name] = torch.cat([
                                    roi_features[layer_name],
                                    rgb_roi_features
                                ], dim=1)

                # Fuse features
                if hasattr(self, 'feature_fusion'):
                    fused_features = self.feature_fusion(roi_features)
                else:
                    fused_features = list(roi_features.values())[0]
            else:
                # Fallback for simpler models
                output = self.base_model(features=features, rois=rois)
                fused_features = output['features'] if isinstance(output, dict) else features

            # Apply hierarchical head
            logits, aux_outputs = self.hierarchical_head(fused_features)

            # Always return tuple for consistency
            return logits, aux_outputs

    return HierarchicalSegmentationModel(base_model)