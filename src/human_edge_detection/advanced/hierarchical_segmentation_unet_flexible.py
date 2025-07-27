"""Flexible hierarchical segmentation UNet with configurable activation functions.

This module extends the hierarchical segmentation UNet models to support
different activation functions (ReLU, Swish, etc.) through configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from ..model import LayerNorm2d, ResidualBlock
from .attention_modules import ChannelAttentionModule, SpatialAttentionModule
from .activation_utils import ActivationConfig, get_activation


class FlexibleResidualBlock(nn.Module):
    """Residual block with configurable activation function."""
    
    def __init__(self, channels: int, activation_config: ActivationConfig,
                 normalization_type: str = 'layernorm2d',
                 normalization_groups: int = 8):
        """Initialize flexible residual block.
        
        Args:
            channels: Number of channels
            activation_config: Configuration for activation functions
            normalization_type: Type of normalization to use
            normalization_groups: Number of groups for group normalization
        """
        super().__init__()
        from .normalization_comparison import get_normalization_layer
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = get_normalization_layer(normalization_type, channels, num_groups=min(normalization_groups, channels))
        self.activation1 = activation_config.get_activation()
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = get_normalization_layer(normalization_type, channels, num_groups=min(normalization_groups, channels))
        self.activation2 = activation_config.get_activation()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = x + residual
        x = self.activation2(x)
        return x


class FlexibleEnhancedUNet(nn.Module):
    """Enhanced UNet with configurable activation functions."""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 3,
        activation_config: Optional[ActivationConfig] = None
    ):
        """Initialize flexible enhanced UNet.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base number of channels
            depth: Depth of the UNet (number of downsampling layers)
            activation_config: Configuration for activation functions. 
                              If None, uses ReLU by default.
        """
        super().__init__()
        
        # Default to ReLU if no config provided
        if activation_config is None:
            activation_config = ActivationConfig(activation="relu")
        
        self.activation_config = activation_config
        self.depth = depth
        
        # Build encoder path
        self.encoders = nn.ModuleList()
        channels = base_channels
        
        for i in range(depth):
            in_ch = in_channels if i == 0 else channels
            out_ch = channels * (2 ** i)
            
            encoder = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                activation_config.get_norm_activation(out_ch),
                FlexibleResidualBlock(out_ch, activation_config)
            )
            self.encoders.append(encoder)
        
        # Bottleneck
        bottleneck_channels = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * (2 ** (depth - 1)), bottleneck_channels, 3, padding=1),
            activation_config.get_norm_activation(bottleneck_channels),
            FlexibleResidualBlock(bottleneck_channels, activation_config),
            FlexibleResidualBlock(bottleneck_channels, activation_config)
        )
        
        # Build decoder path
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            in_ch = bottleneck_channels if i == depth - 1 else base_channels * (2 ** (i + 1))
            out_ch = base_channels * (2 ** i)
            
            upsampler = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            decoder = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),  # *2 for skip connection
                activation_config.get_norm_activation(out_ch),
                FlexibleResidualBlock(out_ch, activation_config)
            )
            
            self.upsamplers.append(upsampler)
            self.decoders.append(decoder)
        
        # Output layer
        self.output = nn.Conv2d(base_channels, 2, 1)  # 2 classes for bg/fg
        
        # Pooling for encoder
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path with skip connections
        skip_connections = []
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skip_connections.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upsampler, decoder) in enumerate(zip(self.upsamplers, self.decoders)):
            x = upsampler(x)
            
            # Get corresponding skip connection
            skip = skip_connections[self.depth - 1 - i]
            
            # Ensure spatial dimensions match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Output
        out = self.output(x)
        return out


class FlexibleHierarchicalSegmentationHeadUNetV2(nn.Module):
    """Flexible version of HierarchicalSegmentationHeadUNetV2 with configurable activations."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56,
        dropout_rate: float = 0.1,
        use_attention_module: bool = False,
        activation: str = "relu",
        activation_beta: float = 1.0,
        norm_type: str = "layernorm2d"
    ):
        """Initialize flexible hierarchical segmentation head V2.
        
        Args:
            in_channels: Input feature channels
            mid_channels: Intermediate channel count
            num_classes: Total number of classes (should be 3)
            mask_size: Output mask size
            dropout_rate: Dropout rate for regularization
            use_attention_module: Whether to use attention modules
            activation: Name of activation function ("relu", "swish", "gelu", "silu")
            activation_beta: Beta parameter for Swish activation
            norm_type: Type of normalization to use
        """
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"
        
        self.num_classes = num_classes
        self.mask_size = mask_size
        self.use_attention_module = use_attention_module
        
        # Create activation configuration
        self.activation_config = ActivationConfig(
            activation=activation,
            norm_type=norm_type,
            inplace=True,
            beta=activation_beta
        )
        
        # Shared feature processing with dropout
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            self.activation_config.get_norm_activation(mid_channels),
            nn.Dropout2d(dropout_rate),
            FlexibleResidualBlock(mid_channels, self.activation_config),
            nn.Dropout2d(dropout_rate),
            FlexibleResidualBlock(mid_channels, self.activation_config),
        )
        
        # Branch 1: Background vs Foreground using Enhanced UNet
        self.bg_vs_fg_unet = FlexibleEnhancedUNet(
            mid_channels, 
            base_channels=96, 
            depth=3,
            activation_config=self.activation_config
        )
        
        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            self.activation_config.get_norm_activation(32),
            nn.Conv2d(32, 2, 1)
        )
        
        # Branch 2: Target vs Non-target
        if use_attention_module:
            # Build attention-based branch
            self.target_vs_nontarget_branch = nn.ModuleList([
                FlexibleResidualBlock(mid_channels, self.activation_config),
                SpatialAttentionModule(kernel_size=7),
                nn.Dropout2d(dropout_rate),
                nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
                self.activation_config.get_norm_activation(mid_channels // 2),
                ChannelAttentionModule(mid_channels // 2, reduction_ratio=8),
                nn.Dropout2d(dropout_rate),
                FlexibleResidualBlock(mid_channels // 2, self.activation_config),
                nn.Conv2d(mid_channels // 2, 2, 1)
            ])
        else:
            self.target_vs_nontarget_branch = nn.Sequential(
                FlexibleResidualBlock(mid_channels, self.activation_config),
                nn.Dropout2d(dropout_rate),
                nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
                self.activation_config.get_norm_activation(mid_channels // 2),
                nn.Dropout2d(dropout_rate),
                FlexibleResidualBlock(mid_channels // 2, self.activation_config),
                nn.Conv2d(mid_channels // 2, 2, 1)
            )
        
        # Enhanced gating
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            self.activation_config.get_activation(),
            nn.Dropout2d(dropout_rate * 0.5),
            nn.Conv2d(mid_channels // 4, mid_channels // 2, 1),
            self.activation_config.get_activation(),
            nn.Conv2d(mid_channels // 2, mid_channels, 1),
            nn.Sigmoid()
        )
        
        # Store shared features for refinement modules
        self._shared_features_cache = None
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with feature caching.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (logits, auxiliary outputs)
        """
        # Shared feature extraction
        shared = self.shared_features(features)
        self._shared_features_cache = shared  # Cache for refinement modules
        
        # Branch 1: Background vs Foreground
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        
        # Upsample to mask size
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        if bg_fg_logits.shape[2] != self.mask_size or bg_fg_logits.shape[3] != self.mask_size:
            bg_fg_logits = F.interpolate(
                bg_fg_logits,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Get foreground probability for gating
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        # Generate attention/gating from bg/fg prediction
        fg_attention = self.fg_gate(bg_fg_logits_low)
        
        # Apply gating to shared features
        gated_features = shared * fg_attention
        
        # Branch 2: Target vs Non-target classification
        if self.use_attention_module:
            # Process through attention modules
            x = gated_features
            for module in self.target_vs_nontarget_branch:
                if isinstance(module, (SpatialAttentionModule, ChannelAttentionModule)):
                    x = module(x)
                elif isinstance(module, nn.ModuleList):
                    # Skip ModuleList processing
                    continue
                else:
                    x = module(x)
            target_nontarget_logits = x
        else:
            target_nontarget_logits = self.target_vs_nontarget_branch(gated_features)
        
        # Ensure target/non-target logits match mask size
        if target_nontarget_logits.shape[2] != self.mask_size or target_nontarget_logits.shape[3] != self.mask_size:
            target_nontarget_logits = F.interpolate(
                target_nontarget_logits,
                size=(self.mask_size, self.mask_size),
                mode='bilinear',
                align_corners=False
            )
        
        # Combine predictions hierarchically
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_size, self.mask_size,
                                  device=features.device)
        
        # Background class (from bg/fg branch)
        final_logits[:, 0] = bg_fg_logits[:, 0]
        
        # Foreground classes weighted by foreground probability
        fg_mask = bg_fg_probs[:, 1:2]  # Keep dimensions for broadcasting
        final_logits[:, 1] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 0] * fg_mask.squeeze(1)
        final_logits[:, 2] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 1] * fg_mask.squeeze(1)
        
        # Auxiliary outputs for loss computation
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention,
            'shared_features': self._shared_features_cache
        }
        
        return final_logits, aux_outputs


# Export convenience function
def create_flexible_hierarchical_head(
    in_channels: int,
    mid_channels: int = 256,
    num_classes: int = 3,
    mask_size: int = 56,
    use_attention_module: bool = False,
    activation: str = "relu",
    **kwargs
) -> FlexibleHierarchicalSegmentationHeadUNetV2:
    """Create a flexible hierarchical segmentation head with configurable activation.
    
    Args:
        in_channels: Number of input channels
        mid_channels: Number of intermediate channels
        num_classes: Number of output classes
        mask_size: Size of output masks
        use_attention_module: Whether to use attention modules
        activation: Name of activation function
        **kwargs: Additional arguments passed to the model
        
    Returns:
        Flexible hierarchical segmentation head
    """
    return FlexibleHierarchicalSegmentationHeadUNetV2(
        in_channels=in_channels,
        mid_channels=mid_channels,
        num_classes=num_classes,
        mask_size=mask_size,
        use_attention_module=use_attention_module,
        activation=activation,
        **kwargs
    )