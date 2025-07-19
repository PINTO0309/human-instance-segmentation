"""Distance-aware loss functions for improved instance separation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class DistanceTransform:
    """Compute distance transforms for masks."""

    @staticmethod
    def compute_boundary_distance(
        mask: torch.Tensor,
        max_distance: float = 10.0
    ) -> torch.Tensor:
        """Compute distance to nearest boundary.

        Args:
            mask: Binary mask or class mask (H, W)
            max_distance: Maximum distance to compute

        Returns:
            Distance map (H, W)
        """
        # Convert to numpy for distance transform
        mask_np = mask.cpu().numpy()

        # Import scipy here to avoid dependency issues
        from scipy.ndimage import distance_transform_edt

        # For multi-class mask, compute edges
        if len(np.unique(mask_np)) > 2:
            # Compute gradients to find boundaries
            dy = np.abs(np.diff(mask_np, axis=0, prepend=mask_np[0:1]))
            dx = np.abs(np.diff(mask_np, axis=1, prepend=mask_np[:, 0:1]))
            edges = ((dy > 0) | (dx > 0)).astype(np.float32)
        else:
            # Binary mask - find edges
            edges = mask_np.astype(np.float32)

        # Compute distance transform
        dist = distance_transform_edt(1 - edges)

        # Clip to max distance
        dist = np.minimum(dist, max_distance)

        # Convert back to tensor
        return torch.from_numpy(dist).to(mask.device)

    @staticmethod
    def compute_instance_distances(
        instance_masks: List[torch.Tensor],
        mask_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Compute pairwise distances between instances.

        Args:
            instance_masks: List of binary masks for each instance
            mask_shape: Shape of output mask

        Returns:
            Distance map showing nearest instance distance
        """
        if len(instance_masks) < 2:
            # No other instances - return zeros
            return torch.zeros(mask_shape)

        # Stack all instance masks
        masks_stack = torch.stack(instance_masks, dim=0)  # (N, H, W)

        # Compute centroids
        centroids = []
        for mask in instance_masks:
            if mask.sum() > 0:
                y_coords, x_coords = torch.where(mask > 0)
                centroid_y = y_coords.float().mean()
                centroid_x = x_coords.float().mean()
                centroids.append(torch.tensor([centroid_y, centroid_x]))
            else:
                centroids.append(torch.tensor([mask_shape[0]/2, mask_shape[1]/2]))

        centroids = torch.stack(centroids)  # (N, 2)

        # Compute pairwise distances between centroids
        dist_matrix = torch.cdist(centroids.unsqueeze(0), centroids.unsqueeze(0))[0]

        # For each pixel, find distance to nearest other instance
        H, W = mask_shape
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )

        pixel_coords = torch.stack([y_grid, x_grid], dim=-1).to(masks_stack.device)  # (H, W, 2)

        # Initialize distance map
        distance_map = torch.full((H, W), float('inf'), device=masks_stack.device)

        # For each instance
        for i, mask in enumerate(instance_masks):
            if mask.sum() == 0:
                continue

            # Find nearest other instance
            other_dists = dist_matrix[i].clone()
            other_dists[i] = float('inf')  # Exclude self

            if torch.isfinite(other_dists).any():
                nearest_idx = other_dists.argmin()
                nearest_distance = other_dists[nearest_idx]

                # Update distance map for pixels in this instance
                distance_map[mask > 0] = torch.minimum(
                    distance_map[mask > 0],
                    nearest_distance
                )

        # Replace inf with max distance
        distance_map[torch.isinf(distance_map)] = 100.0

        return distance_map


class BoundaryWeightGenerator:
    """Generate weights for boundary regions."""

    def __init__(
        self,
        boundary_width: int = 5,
        boundary_weight: float = 2.0,
        instance_sep_weight: float = 3.0
    ):
        """Initialize boundary weight generator.

        Args:
            boundary_width: Width of boundary region in pixels
            boundary_weight: Weight for general boundaries
            instance_sep_weight: Weight for instance separation boundaries
        """
        self.boundary_width = boundary_width
        self.boundary_weight = boundary_weight
        self.instance_sep_weight = instance_sep_weight

    def generate_weights(
        self,
        target_mask: torch.Tensor,
        instance_info: Optional[Dict] = None
    ) -> torch.Tensor:
        """Generate weight map for loss computation.

        Args:
            target_mask: Target segmentation mask (H, W)
            instance_info: Optional instance information

        Returns:
            Weight map (H, W)
        """
        H, W = target_mask.shape
        weights = torch.ones_like(target_mask, dtype=torch.float32)

        # Compute boundary distances
        boundary_dist = DistanceTransform.compute_boundary_distance(
            target_mask,
            max_distance=self.boundary_width
        )

        # Apply boundary weights
        boundary_mask = boundary_dist < self.boundary_width
        weight_scale = 1.0 + (self.boundary_weight - 1.0) * (
            1.0 - boundary_dist / self.boundary_width
        )
        weights[boundary_mask] *= weight_scale[boundary_mask]

        # Apply instance separation weights if available
        if instance_info and 'instance_distances' in instance_info:
            instance_dist = instance_info['instance_distances']

            # Boost weights for pixels near other instances
            close_instances = instance_dist < 50.0  # Within 50 pixels
            weights[close_instances] *= self.instance_sep_weight

        return weights


class DistanceAwareSegmentationLoss(nn.Module):
    """Distance-aware loss for instance segmentation."""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        boundary_width: int = 5,
        boundary_weight: float = 2.0,
        instance_sep_weight: float = 3.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0
    ):
        """Initialize distance-aware loss.

        Args:
            class_weights: Weights for each class
            ce_weight: Weight for cross-entropy component
            dice_weight: Weight for Dice component
            boundary_width: Width of boundary region
            boundary_weight: Weight multiplier for boundaries
            instance_sep_weight: Weight multiplier for instance separation
            use_focal: Whether to use focal loss
            focal_gamma: Gamma parameter for focal loss
        """
        super().__init__()

        self.class_weights = class_weights
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        # Boundary weight generator
        self.weight_generator = BoundaryWeightGenerator(
            boundary_width=boundary_width,
            boundary_weight=boundary_weight,
            instance_sep_weight=instance_sep_weight
        )

        # Base losses
        from ..losses import DiceLoss, FocalLoss

        if use_focal:
            self.ce_loss = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            # We'll apply weights manually for distance awareness
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')

        self.dice_loss = DiceLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        instance_info: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute distance-aware loss.

        Args:
            predictions: Predicted logits (N, C, H, W)
            targets: Target masks (N, H, W)
            instance_info: Optional list of instance information per sample

        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary with loss components
        """
        # Handle both batch format (B, C, H, W) and ROI format (N_rois, C, H, W)
        if len(predictions.shape) == 4:
            N, C, H, W = predictions.shape
        else:
            raise ValueError(f"Expected 4D predictions, got shape {predictions.shape}")

        # Compute base CE loss (per-pixel)
        ce_loss_map = self.ce_loss(predictions, targets)

        # Generate weight maps
        total_weights = torch.ones_like(targets, dtype=torch.float32)

        for i in range(N):
            # Generate weights for this sample
            sample_info = instance_info[i] if instance_info else None
            weights = self.weight_generator.generate_weights(
                targets[i],
                sample_info
            )
            total_weights[i] = weights

        # Apply weights to CE loss
        weighted_ce_loss = (ce_loss_map * total_weights).mean()

        # Apply class weights if provided
        if self.class_weights is not None:
            class_weight_map = self.class_weights[targets]
            weighted_ce_loss = (weighted_ce_loss * class_weight_map).mean()
        else:
            weighted_ce_loss = weighted_ce_loss.mean()

        # Dice loss (already handles reduction)
        dice_loss = self.dice_loss(predictions, targets, class_indices=[1])

        # Combine losses
        total_loss = self.ce_weight * weighted_ce_loss + self.dice_weight * dice_loss

        # Additional distance-aware metrics
        with torch.no_grad():
            # Compute boundary accuracy
            boundary_masks = total_weights > 1.5  # Boundary regions
            if boundary_masks.any():
                pred_classes = predictions.argmax(dim=1)
                boundary_acc = (pred_classes[boundary_masks] == targets[boundary_masks]).float().mean()
            else:
                boundary_acc = torch.tensor(0.0)

        loss_dict = {
            'total_loss': total_loss,
            'ce_loss': weighted_ce_loss,
            'dice_loss': dice_loss,
            'boundary_accuracy': boundary_acc,
            'avg_boundary_weight': total_weights.mean()
        }

        return total_loss, loss_dict


class AdaptiveBoundaryLoss(nn.Module):
    """Adaptive boundary loss that adjusts weights during training."""

    def __init__(
        self,
        base_loss: DistanceAwareSegmentationLoss,
        adaptation_rate: float = 0.01,
        min_weight: float = 1.0,
        max_weight: float = 5.0
    ):
        """Initialize adaptive boundary loss.

        Args:
            base_loss: Base distance-aware loss
            adaptation_rate: Rate of weight adaptation
            min_weight: Minimum boundary weight
            max_weight: Maximum boundary weight
        """
        super().__init__()

        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # Adaptive parameters
        self.boundary_weight = nn.Parameter(
            torch.tensor(base_loss.weight_generator.boundary_weight)
        )
        self.instance_weight = nn.Parameter(
            torch.tensor(base_loss.weight_generator.instance_sep_weight)
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        instance_info: Optional[List[Dict]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with adaptive weights."""
        # Update base loss weights
        self.base_loss.weight_generator.boundary_weight = torch.clamp(
            self.boundary_weight,
            self.min_weight,
            self.max_weight
        ).item()

        self.base_loss.weight_generator.instance_sep_weight = torch.clamp(
            self.instance_weight,
            self.min_weight,
            self.max_weight
        ).item()

        # Compute loss
        total_loss, loss_dict = self.base_loss(predictions, targets, instance_info)

        # Add weight info to loss dict
        loss_dict['boundary_weight'] = self.boundary_weight.item()
        loss_dict['instance_weight'] = self.instance_weight.item()

        return total_loss, loss_dict


def create_distance_aware_loss(
    pixel_ratios: dict,
    boundary_width: int = 5,
    boundary_weight: float = 2.0,
    instance_sep_weight: float = 3.0,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    adaptive: bool = False,
    device: str = 'cpu',
    separation_aware_weights: Optional[dict] = None
) -> nn.Module:
    """Create distance-aware loss function.

    Args:
        pixel_ratios: Class pixel ratios
        boundary_width: Width of boundary region
        boundary_weight: Weight for boundaries
        instance_sep_weight: Weight for instance separation
        ce_weight: Cross-entropy weight
        dice_weight: Dice loss weight
        adaptive: Whether to use adaptive weights
        device: Device for tensors
        separation_aware_weights: Pre-computed class weights

    Returns:
        Loss function module
    """
    # Get class weights
    from ..losses import create_loss_function

    # Create base loss with class weights
    base_loss_fn = create_loss_function(
        pixel_ratios,
        ce_weight=1.0,  # We'll apply CE weight in our loss
        dice_weight=1.0,
        device=device,
        separation_aware_weights=separation_aware_weights
    )

    # Extract class weights
    class_weights = base_loss_fn.ce_loss.weight

    # Create distance-aware loss
    distance_loss = DistanceAwareSegmentationLoss(
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        boundary_width=boundary_width,
        boundary_weight=boundary_weight,
        instance_sep_weight=instance_sep_weight
    )

    if adaptive:
        return AdaptiveBoundaryLoss(distance_loss)
    else:
        return distance_loss


if __name__ == "__main__":
    # Test distance-aware loss
    print("Testing DistanceAwareSegmentationLoss...")

    # Create dummy data
    batch_size = 2
    num_classes = 3
    H, W = 56, 56

    predictions = torch.randn(batch_size, num_classes, H, W)
    targets = torch.randint(0, num_classes, (batch_size, H, W))

    # Create dummy instance info
    instance_info = []
    for i in range(batch_size):
        # Create fake instance masks
        instance_masks = []
        for j in range(3):  # 3 instances
            mask = torch.zeros(H, W)
            # Random rectangular region
            x1, y1 = torch.randint(0, W//2, (2,))
            x2, y2 = x1 + torch.randint(10, 30, (1,)), y1 + torch.randint(10, 30, (1,))
            mask[y1:y2, x1:x2] = 1
            instance_masks.append(mask)

        # Compute instance distances
        instance_dist = DistanceTransform.compute_instance_distances(
            instance_masks,
            (H, W)
        )

        instance_info.append({
            'instance_masks': instance_masks,
            'instance_distances': instance_dist
        })

    # Test loss computation
    loss_fn = create_distance_aware_loss(
        pixel_ratios={'background': 0.5, 'target': 0.3, 'non_target': 0.2},
        boundary_width=5,
        boundary_weight=2.0,
        instance_sep_weight=3.0
    )

    total_loss, loss_dict = loss_fn(predictions, targets, instance_info)

    print(f"Total loss: {total_loss.item():.4f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.item():.4f}")
        else:
            print(f"  {k}: {v}")

    # Test adaptive loss
    print("\nTesting AdaptiveBoundaryLoss...")
    adaptive_loss = create_distance_aware_loss(
        pixel_ratios={'background': 0.5, 'target': 0.3, 'non_target': 0.2},
        adaptive=True
    )

    total_loss, loss_dict = adaptive_loss(predictions, targets, instance_info)
    print(f"Adaptive loss: {total_loss.item():.4f}")
    print(f"Adaptive weights: boundary={loss_dict['boundary_weight']:.2f}, "
          f"instance={loss_dict['instance_weight']:.2f}")