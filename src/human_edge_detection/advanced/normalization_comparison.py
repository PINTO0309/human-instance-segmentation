"""Normalization layer comparison for foreground/background segmentation.

This module provides different normalization options optimized for 
foreground/background separation tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AdaptiveInstanceNorm2d(nn.Module):
    """Adaptive Instance Normalization for better fg/bg separation.
    
    This normalization computes statistics per-instance and per-channel,
    which helps preserve spatial structure important for segmentation.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for inference stability
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Compute instance statistics
        x_reshaped = x.view(B, C, -1)
        mean = x_reshaped.mean(dim=2, keepdim=True)
        var = x_reshaped.var(dim=2, keepdim=True, unbiased=False)
        
        # Update running statistics during training
        if self.training:
            # Compute batch statistics
            batch_mean = mean.mean(dim=0).squeeze()
            batch_var = var.mean(dim=0).squeeze()
            
            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        # Normalize
        x_norm = (x_reshaped - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(B, C, H, W)
        
        # Apply affine transformation
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        return x_norm * weight + bias


class SpatialGroupNorm(nn.Module):
    """Spatial Group Normalization optimized for segmentation.
    
    Groups channels and normalizes them separately, which can help
    different channel groups specialize for fg/bg features.
    """
    
    def __init__(self, num_channels: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # GroupNorm layer
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class ForegroundAwareNorm(nn.Module):
    """Foreground-aware normalization that adapts based on fg/bg ratio.
    
    This experimental normalization adjusts its behavior based on the
    estimated foreground content in the feature maps.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Base normalization
        self.norm = nn.InstanceNorm2d(num_features, eps=eps, affine=False)
        
        # Adaptive scaling based on content
        self.fg_scale = nn.Parameter(torch.ones(num_features))
        self.fg_bias = nn.Parameter(torch.zeros(num_features))
        self.bg_scale = nn.Parameter(torch.ones(num_features))
        self.bg_bias = nn.Parameter(torch.zeros(num_features))
        
        # Attention mechanism to detect foreground
        self.fg_detector = nn.Sequential(
            nn.Conv2d(num_features, num_features // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize
        x_norm = self.norm(x)
        
        # Detect foreground probability
        fg_prob = self.fg_detector(x.detach())  # Detach to prevent gradient flow
        bg_prob = 1 - fg_prob
        
        # Apply adaptive affine transformation
        B, C, H, W = x.shape
        fg_scale = self.fg_scale.view(1, C, 1, 1)
        fg_bias = self.fg_bias.view(1, C, 1, 1)
        bg_scale = self.bg_scale.view(1, C, 1, 1)
        bg_bias = self.bg_bias.view(1, C, 1, 1)
        
        # Weighted combination based on fg/bg probability
        scale = fg_prob * fg_scale + bg_prob * bg_scale
        bias = fg_prob * fg_bias + bg_prob * bg_bias
        
        return x_norm * scale + bias


class MixedNormalization(nn.Module):
    """Mixed normalization that combines BatchNorm and InstanceNorm.
    
    Uses BatchNorm for early layers (global patterns) and InstanceNorm
    for later layers (local details).
    """
    
    def __init__(self, num_features: int, mix_ratio: float = 0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=True)
        self.mix_ratio = mix_ratio
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # During training, use mixed normalization
            bn_out = self.batch_norm(x)
            in_out = self.instance_norm(x)
            return self.mix_ratio * bn_out + (1 - self.mix_ratio) * in_out
        else:
            # During inference, use BatchNorm for stability
            return self.batch_norm(x)


def get_normalization_layer(
    norm_type: str,
    num_features: int,
    **kwargs
) -> nn.Module:
    """Get normalization layer by type.
    
    Args:
        norm_type: Type of normalization ('layer', 'batch', 'instance', 'group', 
                   'adaptive_instance', 'spatial_group', 'foreground_aware', 'mixed')
        num_features: Number of features/channels
        **kwargs: Additional arguments for specific normalizations
        
    Returns:
        Normalization module
    """
    from ..model import LayerNorm2d
    
    norm_type = norm_type.lower()
    
    if norm_type == 'layer' or norm_type == 'layernorm' or norm_type == 'layernorm2d':
        return LayerNorm2d(num_features)
    elif norm_type == 'batch' or norm_type == 'batchnorm' or norm_type == 'batchnorm2d':
        return nn.BatchNorm2d(num_features)
    elif norm_type == 'instance' or norm_type == 'instancenorm' or norm_type == 'instancenorm2d':
        return nn.InstanceNorm2d(num_features, affine=True)
    elif norm_type == 'group' or norm_type == 'groupnorm':
        num_groups = kwargs.get('num_groups', 8)
        # Ensure num_features is divisible by num_groups
        if num_features % num_groups != 0:
            # Find the closest valid num_groups
            for g in [8, 4, 2, 1]:
                if num_features % g == 0:
                    num_groups = g
                    break
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == 'adaptive_instance':
        return AdaptiveInstanceNorm2d(num_features)
    elif norm_type == 'spatial_group':
        num_groups = kwargs.get('num_groups', 8)
        return SpatialGroupNorm(num_features, num_groups)
    elif norm_type == 'foreground_aware':
        return ForegroundAwareNorm(num_features)
    elif norm_type == 'mixed':
        mix_ratio = kwargs.get('mix_ratio', 0.5)
        return MixedNormalization(num_features, mix_ratio)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


class NormComparisonUNet(nn.Module):
    """U-Net variant for comparing different normalization strategies.
    
    This model allows easy switching between normalization types to
    evaluate their effectiveness for fg/bg segmentation.
    """
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        norm_type: str = 'batch',
        **norm_kwargs
    ):
        super().__init__()
        
        # Encoder with selected normalization
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            get_normalization_layer(norm_type, base_channels, **norm_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            get_normalization_layer(norm_type, base_channels, **norm_kwargs),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 2, **norm_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 2, **norm_kwargs),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 4, **norm_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 4, **norm_kwargs),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 2, **norm_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            get_normalization_layer(norm_type, base_channels * 2, **norm_kwargs),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            get_normalization_layer(norm_type, base_channels, **norm_kwargs),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            get_normalization_layer(norm_type, base_channels, **norm_kwargs),
            nn.ReLU(inplace=True)
        )
        
        # Output
        self.output = nn.Conv2d(base_channels, 2, 1)  # Binary fg/bg
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.bottleneck(enc2)
        
        # Decoder with skip connections
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return self.output(x)