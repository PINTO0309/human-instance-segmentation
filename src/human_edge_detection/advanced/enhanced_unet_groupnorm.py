"""Enhanced UNet with GroupNorm for foreground/background segmentation.

This module provides GroupNorm-based versions of the Enhanced UNet architecture,
which are more suitable for foreground/background separation tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ResidualBlockGroupNorm(nn.Module):
    """Residual block with GroupNorm instead of LayerNorm."""
    
    def __init__(self, channels: int, groups: int = 8):
        """Initialize residual block with GroupNorm.
        
        Args:
            channels: Number of channels
            groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        
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


class EnhancedUNetGroupNorm(nn.Module):
    """Enhanced U-Net with GroupNorm for better fg/bg separation."""
    
    def __init__(
        self, 
        in_channels: int, 
        base_channels: int = 64, 
        depth: int = 4,
        groups: int = 8
    ):
        """Initialize enhanced U-Net with GroupNorm.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base channel count for the network
            depth: Depth of the U-Net (number of downsampling operations)
            groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.depth = depth
        self.groups = groups
        
        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        channels = [in_channels] + [base_channels * (2**i) for i in range(depth)]
        
        for i in range(depth):
            # Ensure number of channels is divisible by groups
            out_channels = channels[i+1]
            if out_channels % groups != 0:
                # Adjust channels to be divisible by groups
                out_channels = ((out_channels + groups - 1) // groups) * groups
                channels[i+1] = out_channels
            
            encoder = nn.Sequential(
                nn.Conv2d(channels[i], out_channels, 3, padding=1),
                nn.GroupNorm(min(groups, out_channels), out_channels),
                nn.ReLU(inplace=True),
                ResidualBlockGroupNorm(out_channels, min(groups, out_channels))
            )
            self.encoders.append(encoder)
            
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))
        
        # Bottleneck
        bottleneck_channels = channels[-1] * 2
        if bottleneck_channels % groups != 0:
            bottleneck_channels = ((bottleneck_channels + groups - 1) // groups) * groups
            
        self.bottleneck = nn.Sequential(
            nn.Conv2d(channels[-1], bottleneck_channels, 3, padding=1),
            nn.GroupNorm(min(groups, bottleneck_channels), bottleneck_channels),
            nn.ReLU(inplace=True),
            ResidualBlockGroupNorm(bottleneck_channels, min(groups, bottleneck_channels)),
            ResidualBlockGroupNorm(bottleneck_channels, min(groups, bottleneck_channels))
        )
        
        # Build decoder
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        # Update channels list to reflect any adjustments
        decoder_channels = [bottleneck_channels] + channels[1:][::-1]
        
        for i in range(depth):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1] if i < depth - 1 else base_channels
            
            # Ensure output channels divisible by groups
            if out_ch % groups != 0 and out_ch > groups:
                out_ch = ((out_ch + groups - 1) // groups) * groups
            
            upsampler = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
            
            # Decoder expects concatenated features (skip connection)
            decoder = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1),
                nn.GroupNorm(min(groups, out_ch), out_ch),
                nn.ReLU(inplace=True),
                ResidualBlockGroupNorm(out_ch, min(groups, out_ch))
            )
            
            self.upsamplers.append(upsampler)
            self.decoders.append(decoder)
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, 2, 1)  # 2 classes for bg/fg
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        encoder_features = []
        
        for i, (encoder, pool) in enumerate(zip(self.encoders, self.pools + [None])):
            x = encoder(x)
            encoder_features.append(x)
            if pool is not None:
                x = pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upsampler, decoder, skip) in enumerate(
            zip(self.upsamplers, self.decoders, encoder_features[::-1])
        ):
            x = upsampler(x)
            
            # Handle size mismatches
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        
        # Final output
        out = self.final(x)
        return out


class EnhancedUNetGroupNormV2(nn.Module):
    """Enhanced U-Net V2 with GroupNorm - optimized for depth=3."""
    
    def __init__(self, in_channels: int, base_channels: int = 96, groups: int = 8):
        """Initialize enhanced U-Net V2 with GroupNorm.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base channel count (default 96 to be divisible by 8)
            groups: Number of groups for GroupNorm
        """
        super().__init__()
        self.groups = groups
        
        # Ensure base_channels is divisible by groups
        if base_channels % groups != 0:
            base_channels = ((base_channels + groups - 1) // groups) * groups
        
        # Encoder path - depth 3
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 8),
            nn.ReLU(inplace=True),
            ResidualBlockGroupNorm(base_channels * 8, groups),
            ResidualBlockGroupNorm(base_channels * 8, groups)
        )
        
        # Decoder path
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(groups, base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(groups, base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(base_channels, 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)
        
        # Bottleneck
        x = self.bottleneck(enc3)
        
        # Decoder with skip connections
        x = self.up3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        # Output
        return self.final(x)