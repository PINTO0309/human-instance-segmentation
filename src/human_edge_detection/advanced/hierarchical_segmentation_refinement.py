"""Binary Mask Refinement Modules for Hierarchical Segmentation.

This module implements various refinement techniques to improve binary mask quality,
particularly focusing on boundary precision and smoothness.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
from .activation_utils import ActivationConfig, get_activation


class LayerNorm2d(nn.Module):
    """LayerNorm for 2D inputs (B, C, H, W)."""
    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
        
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ResidualBlock(nn.Module):
    """Residual block for easier gradient flow."""
    def __init__(self, channels: int, normalization_type: str = 'layernorm2d', normalization_groups: int = 8):
        super().__init__()
        from .normalization_comparison import get_normalization_layer
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = get_normalization_layer(normalization_type, channels, num_groups=normalization_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = get_normalization_layer(normalization_type, channels, num_groups=normalization_groups)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.relu(x)
        return x


class BoundaryRefinementModule(nn.Module):
    """Refines mask boundaries using specialized edge processing."""
    
    def __init__(self, in_channels: int = 3, edge_channels: int = 32,
                 normalization_type: str = 'layernorm2d', normalization_groups: int = 8):
        """Initialize boundary refinement module.
        
        Args:
            in_channels: Number of input channels (3 for 3-class mask)
            edge_channels: Number of channels for edge processing
            normalization_type: Type of normalization to use
            normalization_groups: Number of groups for GroupNorm
        """
        super().__init__()
        from .normalization_comparison import get_normalization_layer
        
        # Edge detection and refinement network
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels, 3, padding=1),
            get_normalization_layer(normalization_type, edge_channels, num_groups=min(normalization_groups, edge_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, edge_channels, 3, padding=1),
            get_normalization_layer(normalization_type, edge_channels, num_groups=min(normalization_groups, edge_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, in_channels, 1)
        )
        
        # Learnable blending weight - initialize smaller for stability
        self.blend_weight = nn.Parameter(torch.tensor(0.01))
        
        # Initialize edge conv weights smaller
        for m in self.edge_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def detect_edges(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """Detect edges in the mask using Sobel-like filters.
        
        Args:
            mask_logits: Input mask logits of shape (B, C, H, W)
            
        Returns:
            Edge map of shape (B, 1, H, W)
        """
        # Convert logits to probabilities
        probs = torch.softmax(mask_logits, dim=1)
        
        # Compute gradients
        dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        
        # Pad to maintain size
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        
        # Combine gradients
        edges = torch.sqrt(dy**2 + dx**2).mean(dim=1, keepdim=True)
        
        # Normalize and threshold with numerical stability
        edge_min = edges.min()
        edge_max = edges.max()
        if edge_max - edge_min < 1e-6:
            # Avoid division by near-zero
            edges = torch.zeros_like(edges)
        else:
            edges = (edges - edge_min) / (edge_max - edge_min + 1e-6)
        return edges
        
    def forward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """Apply boundary refinement to mask logits.
        
        Args:
            mask_logits: Input mask logits of shape (B, C, H, W)
            
        Returns:
            Refined mask logits of shape (B, C, H, W)
        """
        # Detect edges
        edges = self.detect_edges(mask_logits)
        
        # Apply edge-aware refinement
        refined_edges = self.edge_conv(mask_logits)
        
        # Blend with original based on edge strength
        refined = mask_logits + self.blend_weight * refined_edges * edges
        
        return refined


class ProgressiveUpsamplingDecoder(nn.Module):
    """Progressive upsampling for smoother boundaries."""
    
    def __init__(self, in_channels: int, num_classes: int = 3,
                 normalization_type: str = 'layernorm2d',
                 normalization_groups: int = 8):
        """Initialize progressive upsampling decoder.
        
        Args:
            in_channels: Number of input feature channels
            num_classes: Number of output classes
            normalization_type: Type of normalization to use
            normalization_groups: Number of groups for group normalization
        """
        super().__init__()
        from .normalization_comparison import get_normalization_layer
        
        # Progressive upsampling stages
        self.stages = nn.ModuleList([
            # Stage 1: 2x upsampling
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, 4, stride=2, padding=1),
                get_normalization_layer(normalization_type, in_channels//2, num_groups=min(normalization_groups, in_channels//2)),
                nn.ReLU(inplace=True),
                ResidualBlock(in_channels//2, normalization_type, normalization_groups),
            ),
            # Stage 2: 2x upsampling
            nn.Sequential(
                nn.ConvTranspose2d(in_channels//2, in_channels//4, 4, stride=2, padding=1),
                get_normalization_layer(normalization_type, in_channels//4, num_groups=min(normalization_groups, in_channels//4)),
                nn.ReLU(inplace=True),
                ResidualBlock(in_channels//4, normalization_type, normalization_groups),
            ),
            # Final projection
            nn.Conv2d(in_channels//4, num_classes, 1)
        ])
        
    def forward(self, features: torch.Tensor, target_size: int) -> torch.Tensor:
        """Apply progressive upsampling.
        
        Args:
            features: Input features
            target_size: Target output size
            
        Returns:
            Upsampled output
        """
        x = features
        
        # Apply progressive upsampling
        for i, stage in enumerate(self.stages[:-1]):
            x = stage(x)
        
        # Final projection
        x = self.stages[-1](x)
        
        # Ensure output is target size
        if x.shape[-1] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
        return x


class SubPixelDecoder(nn.Module):
    """Sub-pixel convolution for high-quality upsampling."""
    
    def __init__(self, in_channels: int, num_classes: int = 3, upscale_factor: int = 2):
        """Initialize sub-pixel decoder.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
            upscale_factor: Upsampling factor
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # Sub-pixel convolution
        self.conv = nn.Conv2d(
            in_channels, 
            num_classes * upscale_factor**2, 
            3, 
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply sub-pixel upsampling.
        
        Args:
            features: Input features
            
        Returns:
            Upsampled output
        """
        x = self.conv(features)
        x = self.pixel_shuffle(x)
        return x


class ContourDetectionBranch(nn.Module):
    """Explicit contour detection branch for boundary refinement."""
    
    def __init__(self, in_channels: int, contour_channels: int = 64,
                 normalization_type: str = 'layernorm2d', normalization_groups: int = 8):
        """Initialize contour detection branch.
        
        Args:
            in_channels: Number of input channels
            contour_channels: Number of channels for contour processing
            normalization_type: Type of normalization to use
            normalization_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        from .normalization_comparison import get_normalization_layer
        
        self.contour_branch = nn.Sequential(
            nn.Conv2d(in_channels, contour_channels, 3, padding=1),
            get_normalization_layer(normalization_type, contour_channels, num_groups=normalization_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(contour_channels, contour_channels, 3, padding=1),
            get_normalization_layer(normalization_type, contour_channels, num_groups=normalization_groups),
            nn.ReLU(inplace=True),
            nn.Conv2d(contour_channels, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Detect contours from features.
        
        Args:
            features: Input features
            
        Returns:
            Contour map of shape (B, 1, H, W)
        """
        return self.contour_branch(features)


class DistanceTransformDecoder(nn.Module):
    """Distance transform prediction for mask refinement."""
    
    def __init__(self, in_channels: int, distance_channels: int = 128,
                 normalization_type: str = 'layernorm2d', normalization_groups: int = 8):
        """Initialize distance transform decoder.
        
        Args:
            in_channels: Number of input channels
            distance_channels: Number of channels for distance processing
            normalization_type: Type of normalization to use
            normalization_groups: Number of groups for GroupNorm
        """
        super().__init__()
        
        from .normalization_comparison import get_normalization_layer
        
        self.distance_head = nn.Sequential(
            nn.Conv2d(in_channels, distance_channels, 3, padding=1),
            get_normalization_layer(normalization_type, distance_channels, num_groups=normalization_groups),
            nn.ReLU(inplace=True),
            ResidualBlock(distance_channels, normalization_type, normalization_groups),
            nn.Conv2d(distance_channels, 1, 1)
        )
        
        # Learnable threshold - initialize smaller
        self.threshold = nn.Parameter(torch.tensor(0.3))
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict distance transform and convert to mask.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (mask, distance_map)
        """
        # Predict distance from boundary
        distance_map = self.distance_head(features)
        
        # Convert to mask using learnable threshold
        mask = torch.sigmoid((distance_map - self.threshold) * 10)  # Sharp transition
        
        return mask, distance_map


def active_contour_loss(pred_mask: torch.Tensor, smoothness_weight: float = 0.01) -> torch.Tensor:
    """Active contour loss for smooth boundaries.
    
    Args:
        pred_mask: Predicted mask probabilities (after softmax)
        smoothness_weight: Weight for curvature term
        
    Returns:
        Scalar loss value
    """
    # Get target class probability (class 1)
    if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
        # Multi-class case - use target class
        pred_mask = pred_mask[:, 1:2, :, :]  # Target class
    
    # Compute gradients
    dy = pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]
    dx = pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]
    
    # Boundary length (L1 norm of gradients) with clamping for stability
    dy_clamped = torch.clamp(torch.abs(dy), max=10.0)
    dx_clamped = torch.clamp(torch.abs(dx), max=10.0)
    boundary_length = torch.mean(dy_clamped) + torch.mean(dx_clamped)
    
    # Curvature (second derivatives)
    if dy.shape[2] > 1:
        ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        curvature_y = torch.mean(torch.abs(ddy))
    else:
        curvature_y = 0
        
    if dx.shape[3] > 1:
        ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        curvature_x = torch.mean(torch.abs(ddx))
    else:
        curvature_x = 0
    
    curvature = curvature_y + curvature_x
    
    return boundary_length + smoothness_weight * curvature


def boundary_aware_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    boundary_width: int = 3,
    boundary_weight: float = 5.0
) -> torch.Tensor:
    """Boundary-aware weighted cross-entropy loss.
    
    Args:
        pred: Predicted logits of shape (B, C, H, W)
        target: Target labels of shape (B, H, W)
        boundary_width: Width of boundary region
        boundary_weight: Weight for boundary pixels
        
    Returns:
        Weighted loss tensor
    """
    # Get one-hot encoding of target
    B, C, H, W = pred.shape
    target_onehot = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()
    
    # Detect boundaries using morphological operations
    kernel_size = boundary_width
    pool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
    
    # Dilate target masks
    dilated = pool(target_onehot)
    
    # Erode target masks (using -MaxPool on inverted mask)
    eroded = 1 - pool(1 - target_onehot)
    
    # Boundary is dilated - eroded
    boundary = (dilated - eroded).sum(dim=1, keepdim=True) > 0
    
    # Create weight map
    weights = torch.ones_like(target, dtype=torch.float32)
    weights[boundary.squeeze(1)] = boundary_weight
    
    # Compute weighted cross-entropy
    loss = F.cross_entropy(pred, target, reduction='none')
    weighted_loss = loss * weights
    
    return weighted_loss.mean()


class ExtendedHierarchicalSegmentationHeadUNetV2(nn.Module):
    """Extended version of HierarchicalSegmentationHeadUNetV2 that exposes shared features."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: Union[int, Tuple[int, int]] = 56,
        dropout_rate: float = 0.1,
        use_attention_module: bool = False,
        normalization_type: str = 'layernorm2d',
        normalization_groups: int = 8
    ):
        """Initialize extended hierarchical segmentation head V2."""
        super().__init__()
        
        # Import necessary components
        from .hierarchical_segmentation_unet import (
            SpatialAttentionModule, ChannelAttentionModule
        )
        
        # Import EnhancedUNet which now supports normalization
        from .hierarchical_segmentation_unet import EnhancedUNet
        base_channels = 64  # Keep consistent base channels
        
        assert num_classes == 3, "Hierarchical model designed for 3 classes"
        self.num_classes = num_classes
        # Support non-square mask sizes
        if isinstance(mask_size, (tuple, list)):
            self.mask_height = int(mask_size[0])
            self.mask_width = int(mask_size[1])
            self.mask_size = mask_size  # Keep original for compatibility
        else:
            self.mask_height = self.mask_width = int(mask_size)
            self.mask_size = mask_size
        self.use_attention_module = use_attention_module
        
        # Import normalization utility
        from .normalization_comparison import get_normalization_layer
        
        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            get_normalization_layer(normalization_type, mid_channels, num_groups=normalization_groups),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            ResidualBlock(mid_channels, normalization_type, normalization_groups),
            nn.Dropout2d(dropout_rate),
            ResidualBlock(mid_channels, normalization_type, normalization_groups),
        )
        
        # Branch 1: Background vs Foreground using Enhanced UNet
        self.bg_vs_fg_unet = EnhancedUNet(
            mid_channels, 
            base_channels=base_channels, 
            depth=3,
            normalization_type=normalization_type,
            normalization_groups=normalization_groups
        )
        
        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            get_normalization_layer(normalization_type, 32, num_groups=min(normalization_groups, 32)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        # Branch 2: Target vs Non-target
        if use_attention_module:
            # Build modules list manually to handle normalization
            modules = [
                ResidualBlock(mid_channels, normalization_type, normalization_groups),
                SpatialAttentionModule(kernel_size=7),
                nn.Dropout2d(dropout_rate),
                nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
                get_normalization_layer(normalization_type, mid_channels // 2, num_groups=min(normalization_groups, mid_channels // 2)),
                nn.ReLU(inplace=True),
                ChannelAttentionModule(mid_channels // 2, reduction_ratio=8),
                nn.Dropout2d(dropout_rate),
                ResidualBlock(mid_channels // 2, normalization_type, min(normalization_groups, mid_channels // 2)),
                nn.Conv2d(mid_channels // 2, 2, 1)
            ]
            self.target_vs_nontarget_branch = nn.ModuleList(modules)
        else:
            self.target_vs_nontarget_branch = nn.Sequential(
                ResidualBlock(mid_channels, normalization_type, normalization_groups),
                nn.Dropout2d(dropout_rate),
                nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
                get_normalization_layer(normalization_type, mid_channels // 2, num_groups=min(normalization_groups, mid_channels // 2)),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                ResidualBlock(mid_channels // 2, normalization_type, min(normalization_groups, mid_channels // 2)),
                nn.Conv2d(mid_channels // 2, 2, 1)
            )
        
        # Enhanced gating
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(mid_channels // 4, mid_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels, 1),
            nn.Sigmoid()
        )
        
        # Store shared features for refinement modules
        self._shared_features_cache = None
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with feature caching."""
        # Shared feature extraction
        shared = self.shared_features(features)
        self._shared_features_cache = shared  # Cache for refinement modules
        
        # Rest is same as original
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        if bg_fg_logits.shape[2] != self.mask_height or bg_fg_logits.shape[3] != self.mask_width:
            bg_fg_logits = F.interpolate(
                bg_fg_logits,
                size=(self.mask_height, self.mask_width),
                mode='bilinear',
                align_corners=False
            )
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        fg_attention = self.fg_gate(bg_fg_logits_low)
        gated_features = shared * fg_attention
        
        if self.use_attention_module:
            x = gated_features
            for module in self.target_vs_nontarget_branch:
                x = module(x)
            target_nontarget_logits = x
        else:
            target_nontarget_logits = self.target_vs_nontarget_branch(gated_features)
        
        if target_nontarget_logits.shape[2] != self.mask_height or target_nontarget_logits.shape[3] != self.mask_width:
            target_nontarget_logits = F.interpolate(
                target_nontarget_logits,
                size=(self.mask_height, self.mask_width),
                mode='bilinear',
                align_corners=False
            )
        
        # Combine predictions
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_height, self.mask_width,
                                  device=features.device)
        
        final_logits[:, 0] = bg_fg_logits[:, 0]
        fg_mask = bg_fg_probs[:, 1:2]
        final_logits[:, 1] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 0] * fg_mask.squeeze(1)
        final_logits[:, 2] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 1] * fg_mask.squeeze(1)
        
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention,
            'shared_features': self._shared_features_cache  # Include shared features
        }
        
        return final_logits, aux_outputs


class RefinedHierarchicalSegmentationHead(nn.Module):
    """Enhanced hierarchical segmentation head with refinement modules."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: Union[int, Tuple[int, int]] = 56,
        use_attention_module: bool = False,
        # Refinement module flags
        use_boundary_refinement: bool = False,
        use_progressive_upsampling: bool = False,
        use_subpixel_conv: bool = False,
        use_contour_detection: bool = False,
        use_distance_transform: bool = False,
        # Normalization configuration
        normalization_type: str = 'layernorm2d',
        normalization_groups: int = 8,
    ):
        """Initialize refined hierarchical segmentation head.
        
        Args:
            in_channels: Number of input channels
            mid_channels: Number of channels in intermediate layers
            num_classes: Number of output classes
            mask_size: Size of output mask
            use_attention_module: Whether to use attention modules
            use_boundary_refinement: Enable boundary refinement
            use_progressive_upsampling: Enable progressive upsampling
            use_subpixel_conv: Enable sub-pixel convolution
            use_contour_detection: Enable contour detection branch
            use_distance_transform: Enable distance transform prediction
        """
        super().__init__()
        
        # Use extended version that exposes shared features
        self.base_head = ExtendedHierarchicalSegmentationHeadUNetV2(
            in_channels=in_channels,
            mid_channels=mid_channels,
            num_classes=num_classes,
            mask_size=mask_size,
            use_attention_module=use_attention_module,
            normalization_type=normalization_type,
            normalization_groups=normalization_groups
        )
        
        # Support non-square mask sizes
        if isinstance(mask_size, (tuple, list)):
            self.mask_height = int(mask_size[0])
            self.mask_width = int(mask_size[1])
            self.mask_size = mask_size  # Keep original for compatibility
        else:
            self.mask_height = self.mask_width = int(mask_size)
            self.mask_size = mask_size
        self.num_classes = num_classes
        
        # Refinement modules
        self.use_boundary_refinement = use_boundary_refinement
        self.use_progressive_upsampling = use_progressive_upsampling
        self.use_subpixel_conv = use_subpixel_conv
        self.use_contour_detection = use_contour_detection
        self.use_distance_transform = use_distance_transform
        
        if use_boundary_refinement:
            self.boundary_refiner = BoundaryRefinementModule(
                in_channels=num_classes,
                edge_channels=32,
                normalization_type=normalization_type,
                normalization_groups=normalization_groups
            )
            
        if use_progressive_upsampling:
            # Replace final decoder with progressive upsampling
            self.progressive_decoder = ProgressiveUpsamplingDecoder(
                in_channels=mid_channels,
                num_classes=num_classes
            )
            
        if use_subpixel_conv:
            # Alternative to progressive upsampling
            self.subpixel_decoder = SubPixelDecoder(
                in_channels=mid_channels,
                num_classes=num_classes,
                upscale_factor=2
            )
            
        if use_contour_detection:
            self.contour_branch = ContourDetectionBranch(
                in_channels=mid_channels,
                contour_channels=64,
                normalization_type=normalization_type,
                normalization_groups=normalization_groups
            )
            
        if use_distance_transform:
            self.distance_decoder = DistanceTransformDecoder(
                in_channels=mid_channels,
                distance_channels=128,
                normalization_type=normalization_type,
                normalization_groups=normalization_groups
            )
            
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with refinement.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (refined masks, auxiliary outputs)
        """
        # Get base predictions
        base_masks, aux_outputs = self.base_head(features)
        
        # Extract shared features from aux_outputs
        bg_fg_features = aux_outputs.get('shared_features', None)
            
        # Apply refinements
        refined_masks = base_masks
        
        # Replace decoder if using progressive upsampling or subpixel conv
        if self.use_progressive_upsampling and bg_fg_features is not None:
            # Re-decode with progressive upsampling
            refined_masks = self.progressive_decoder(bg_fg_features, self.mask_size)
        elif self.use_subpixel_conv and bg_fg_features is not None:
            # Re-decode with sub-pixel convolution
            refined_masks = self.subpixel_decoder(bg_fg_features)
            if refined_masks.shape[2] != self.mask_height or refined_masks.shape[3] != self.mask_width:
                refined_masks = F.interpolate(
                    refined_masks, 
                    size=(self.mask_height, self.mask_width), 
                    mode='bilinear', 
                    align_corners=False
                )
                
        # Apply boundary refinement
        if self.use_boundary_refinement:
            refined_masks = self.boundary_refiner(refined_masks)
            
        # Add auxiliary outputs
        if self.use_contour_detection and bg_fg_features is not None:
            contours = self.contour_branch(bg_fg_features)
            # Ensure contours match mask size
            if contours.shape[2] != self.mask_height or contours.shape[3] != self.mask_width:
                contours = F.interpolate(
                    contours, 
                    size=(self.mask_height, self.mask_width), 
                    mode='bilinear', 
                    align_corners=False
                )
            aux_outputs['contours'] = contours
            
        if self.use_distance_transform and bg_fg_features is not None:
            dist_mask, dist_map = self.distance_decoder(bg_fg_features)
            # Ensure distance outputs match mask size
            if dist_mask.shape[2] != self.mask_height or dist_mask.shape[3] != self.mask_width:
                dist_mask = F.interpolate(
                    dist_mask,
                    size=(self.mask_height, self.mask_width),
                    mode='bilinear',
                    align_corners=False
                )
            if dist_map.shape[2] != self.mask_height or dist_map.shape[3] != self.mask_width:
                dist_map = F.interpolate(
                    dist_map,
                    size=(self.mask_height, self.mask_width),
                    mode='bilinear',
                    align_corners=False
                )
            aux_outputs['distance_mask'] = dist_mask
            aux_outputs['distance_map'] = dist_map
            
        return refined_masks, aux_outputs


class RefinedHierarchicalLoss(nn.Module):
    """Hierarchical loss with refinement components."""
    
    def __init__(
        self,
        bg_weight: float = 1.5,
        fg_weight: float = 1.5,
        target_weight: float = 1.2,
        consistency_weight: float = 0.3,
        use_dynamic_weights: bool = True,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        # Refinement loss weights - start small to avoid instability
        active_contour_weight: float = 0.01,
        boundary_aware_weight: float = 0.01,
        contour_loss_weight: float = 0.01,
        distance_loss_weight: float = 0.01,
        # Refinement flags
        use_active_contour_loss: bool = False,
        use_boundary_aware_loss: bool = False,
        use_contour_detection: bool = False,
        use_distance_transform: bool = False,
    ):
        """Initialize refined hierarchical loss.
        
        Args:
            bg_weight: Weight for background/foreground loss
            fg_weight: Weight for foreground loss
            target_weight: Weight for target/non-target loss
            consistency_weight: Weight for consistency loss
            use_dynamic_weights: Whether to use dynamic weight adjustment
            dice_weight: Weight for Dice loss
            ce_weight: Weight for CrossEntropy loss
            active_contour_weight: Weight for active contour loss
            boundary_aware_weight: Weight for boundary-aware loss
            contour_loss_weight: Weight for contour detection loss
            distance_loss_weight: Weight for distance transform loss
            use_active_contour_loss: Enable active contour loss
            use_boundary_aware_loss: Enable boundary-aware loss
            use_contour_detection: Enable contour detection loss
            use_distance_transform: Enable distance transform loss
        """
        super().__init__()
        
        # Import base hierarchical loss
        from .hierarchical_segmentation import HierarchicalLoss
        
        # Base hierarchical loss
        self.base_loss = HierarchicalLoss(
            bg_weight=bg_weight,
            fg_weight=fg_weight,
            target_weight=target_weight,
            consistency_weight=consistency_weight,
            use_dynamic_weights=use_dynamic_weights,
            dice_weight=dice_weight,
            ce_weight=ce_weight
        )
        
        # Refinement loss weights
        self.active_contour_weight = active_contour_weight
        self.boundary_aware_weight = boundary_aware_weight
        self.contour_loss_weight = contour_loss_weight
        self.distance_loss_weight = distance_loss_weight
        
        # Refinement flags
        self.use_active_contour_loss = use_active_contour_loss
        self.use_boundary_aware_loss = use_boundary_aware_loss
        self.use_contour_detection = use_contour_detection
        self.use_distance_transform = use_distance_transform
        
        # Contour loss (BCE)
        if use_contour_detection:
            self.contour_criterion = nn.BCEWithLogitsLoss()
            
        # Distance transform loss (L1)
        if use_distance_transform:
            self.distance_criterion = nn.L1Loss()
            
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        aux_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute refined hierarchical loss.
        
        Args:
            pred: Predicted logits
            target: Target masks
            aux_outputs: Auxiliary outputs from model
            
        Returns:
            Tuple of (total loss, loss components dict)
        """
        # Get base hierarchical loss
        base_loss, base_components = self.base_loss(pred, target, aux_outputs)
        
        # Initialize total loss and components
        total_loss = base_loss
        loss_components = base_components.copy()
        
        # Add refinement losses with clamping for stability
        if self.use_active_contour_loss:
            # Convert logits to probabilities
            probs = torch.softmax(pred, dim=1)
            ac_loss = active_contour_loss(probs, smoothness_weight=0.01)
            # Clamp to prevent explosion
            ac_loss = torch.clamp(ac_loss, max=10.0)
            total_loss = total_loss + self.active_contour_weight * ac_loss
            loss_components['active_contour'] = ac_loss.item()
            
        if self.use_boundary_aware_loss:
            ba_loss = boundary_aware_loss(
                pred, target, 
                boundary_width=3,
                boundary_weight=2.0
            )
            # Clamp to prevent explosion
            ba_loss = torch.clamp(ba_loss, max=10.0)
            total_loss = total_loss + self.boundary_aware_weight * ba_loss
            loss_components['boundary_aware'] = ba_loss.item()
            
        if self.use_contour_detection and aux_outputs and 'contours' in aux_outputs:
            # Generate contour targets from masks
            contour_targets = self._generate_contour_targets(target)
            contour_loss = self.contour_criterion(
                aux_outputs['contours'], 
                contour_targets
            )
            # Clamp to prevent explosion
            contour_loss = torch.clamp(contour_loss, max=10.0)
            total_loss = total_loss + self.contour_loss_weight * contour_loss
            loss_components['contour'] = contour_loss.item()
            
        if self.use_distance_transform and aux_outputs and 'distance_map' in aux_outputs:
            # Generate distance targets from masks
            distance_targets = self._generate_distance_targets(target)
            distance_loss = self.distance_criterion(
                aux_outputs['distance_map'],
                distance_targets
            )
            # Clamp to prevent explosion
            distance_loss = torch.clamp(distance_loss, max=10.0)
            total_loss = total_loss + self.distance_loss_weight * distance_loss
            loss_components['distance_transform'] = distance_loss.item()
            
        return total_loss, loss_components
        
    def _generate_contour_targets(self, masks: torch.Tensor) -> torch.Tensor:
        """Generate contour targets from segmentation masks.
        
        Args:
            masks: Target masks of shape (B, H, W)
            
        Returns:
            Binary contour targets of shape (B, 1, H, W)
        """
        B, H, W = masks.shape
        
        # Convert to one-hot for edge detection
        masks_onehot = F.one_hot(masks, num_classes=3).permute(0, 3, 1, 2).float()
        
        # Target class (class 1) edges
        target_mask = masks_onehot[:, 1:2, :, :]
        
        # Compute gradients
        dy = torch.abs(target_mask[:, :, 1:, :] - target_mask[:, :, :-1, :])
        dx = torch.abs(target_mask[:, :, :, 1:] - target_mask[:, :, :, :-1])
        
        # Pad to maintain size
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        
        # Combine gradients
        contours = torch.max(dy, dx)
        
        return contours
        
    def _generate_distance_targets(self, masks: torch.Tensor) -> torch.Tensor:
        """Generate distance transform targets from segmentation masks.
        
        Note: This is a simplified version. In practice, you might want to
        use scipy.ndimage.distance_transform_edt on CPU for accurate results.
        
        Args:
            masks: Target masks of shape (B, H, W)
            
        Returns:
            Distance maps of shape (B, 1, H, W)
        """
        B, H, W = masks.shape
        
        # For now, return a simple approximation
        # Real implementation would compute true distance transform
        target_mask = (masks == 1).float().unsqueeze(1)
        
        # Simple distance approximation using max pooling
        distances = target_mask.clone()
        
        # Iterative dilation to approximate distance
        for i in range(5):
            dilated = F.max_pool2d(distances, 3, stride=1, padding=1)
            distances = distances + (1 - distances) * dilated * 0.5
            
        return distances