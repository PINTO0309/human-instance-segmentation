"""Attention modules for enhanced feature extraction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .activation_utils import get_activation


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module (SE-Block style).
    
    Adaptively recalibrates channel-wise feature responses by explicitly
    modeling interdependencies between channels.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        reduction_ratio: int = 8,
        min_channels: int = 8,
        activation_function: str = 'relu',
        activation_beta: float = 1.0
    ):
        """Initialize Channel Attention Module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the bottleneck
            min_channels: Minimum number of channels in bottleneck
            activation_function: Activation function to use
            activation_beta: Beta parameter for Swish
        """
        super().__init__()
        
        # Ensure bottleneck has at least min_channels
        bottleneck_channels = max(in_channels // reduction_ratio, min_channels)
        
        # Global average pooling (implicit)
        self.fc1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.activation = get_activation(activation_function, inplace=True, beta=activation_beta)
        self.fc2 = nn.Conv2d(bottleneck_channels, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Channel-attention weighted features
        """
        # Global average pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
        
        # Channel attention
        channel_att = self.fc1(avg_pool)
        channel_att = self.activation(channel_att)
        channel_att = self.fc2(channel_att)
        channel_att = self.sigmoid(channel_att)
        
        # Apply attention
        return x * channel_att


class SpatialAttentionModule(nn.Module):
    """Spatial Attention Module.
    
    Generates a spatial attention map by utilizing the inter-spatial
    relationship of features.
    """
    
    def __init__(self, kernel_size: int = 7):
        """Initialize Spatial Attention Module.
        
        Args:
            kernel_size: Size of the convolution kernel
        """
        super().__init__()
        
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            2, 1, kernel_size, 
            padding=padding, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Spatially-attention weighted features
        """
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate along channel dimension
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        
        # Spatial attention map
        spatial_att = self.conv(concat)
        spatial_att = self.sigmoid(spatial_att)
        
        # Apply attention
        return x * spatial_att


class CBAMModule(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    Combines both channel and spatial attention mechanisms for
    improved feature refinement.
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 8,
        kernel_size: int = 7,
        activation_function: str = 'relu',
        activation_beta: float = 1.0
    ):
        """Initialize CBAM.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Channel reduction ratio
            kernel_size: Spatial attention kernel size
            activation_function: Activation function to use
            activation_beta: Beta parameter for Swish
        """
        super().__init__()
        
        self.channel_attention = ChannelAttentionModule(
            in_channels, reduction_ratio,
            activation_function=activation_function,
            activation_beta=activation_beta
        )
        self.spatial_attention = SpatialAttentionModule(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM attention.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention-refined features
        """
        # Apply channel attention first
        x = self.channel_attention(x)
        # Then spatial attention
        x = self.spatial_attention(x)
        return x


class AttentionGate(nn.Module):
    """Attention Gate for feature gating.
    
    Used to focus on relevant regions based on gating signal.
    """
    
    def __init__(
        self,
        in_channels: int,
        gating_channels: int,
        inter_channels: Optional[int] = None,
        activation_function: str = 'relu',
        activation_beta: float = 1.0
    ):
        """Initialize Attention Gate.
        
        Args:
            in_channels: Number of input feature channels
            gating_channels: Number of gating signal channels
            inter_channels: Number of intermediate channels
            activation_function: Activation function to use
            activation_beta: Beta parameter for Swish
        """
        super().__init__()
        
        if inter_channels is None:
            inter_channels = in_channels // 2
            
        self.W_g = nn.Conv2d(
            gating_channels, inter_channels, 
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.W_x = nn.Conv2d(
            in_channels, inter_channels,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.psi = nn.Conv2d(
            inter_channels, 1,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.activation = get_activation(activation_function, inplace=True, beta=activation_beta)
        self.sigmoid = nn.Sigmoid()
        
    def forward(
        self, 
        x: torch.Tensor, 
        g: torch.Tensor
    ) -> torch.Tensor:
        """Apply attention gating.
        
        Args:
            x: Input features (B, C, H, W)
            g: Gating signal (B, C_g, H_g, W_g)
            
        Returns:
            Gated features
        """
        # Ensure spatial dimensions match
        if x.shape[2:] != g.shape[2:]:
            g = F.interpolate(
                g, size=x.shape[2:], 
                mode='bilinear', align_corners=False
            )
            
        # Compute attention
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        attention = self.sigmoid(psi)
        
        # Apply attention
        return x * attention