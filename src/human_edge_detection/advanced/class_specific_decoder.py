"""Class-specific decoder architecture for preventing mode collapse."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from ..model import LayerNorm2d, ResidualBlock


class ClassSpecificDecoder(nn.Module):
    """Decoder with separate pathways for each class."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        mid_channels: int = 128,
        mask_size: int = 56,
        share_features: bool = True
    ):
        """Initialize class-specific decoder.
        
        Args:
            in_channels: Input feature channels
            num_classes: Number of output classes
            mid_channels: Channels for each class-specific decoder
            mask_size: Output mask size
            share_features: Whether to share initial feature processing
        """
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        self.share_features = share_features
        
        # Shared initial processing (optional)
        if share_features:
            self.shared_conv = nn.Sequential(
                ResidualBlock(in_channels),
                nn.Conv2d(in_channels, in_channels, 1)
            )
        
        # Class-specific decoders
        self.class_decoders = nn.ModuleList()
        
        for class_idx in range(num_classes):
            # Each class gets its own decoder pathway
            if class_idx == 0:  # Background - simpler decoder
                decoder = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                    LayerNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
                    LayerNorm2d(mid_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels // 2, 1, 1)
                )
            else:  # Target and non-target - more complex decoders
                decoder = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                    LayerNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    ResidualBlock(mid_channels),
                    ResidualBlock(mid_channels),
                    nn.ConvTranspose2d(mid_channels, mid_channels, 2, stride=2),
                    LayerNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    ResidualBlock(mid_channels),
                    nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1),
                    LayerNorm2d(mid_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels // 2, 1, 1)
                )
            
            self.class_decoders.append(decoder)
        
        # Cross-class interaction module (prevents complete isolation)
        self.interaction = nn.Sequential(
            nn.Conv2d(num_classes, num_classes * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes * 4, num_classes, 1)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass with class-specific decoding.
        
        Args:
            features: Input features (N, C, H, W)
            
        Returns:
            Class logits (N, num_classes, H, W)
        """
        # Shared feature processing
        if self.share_features:
            features = self.shared_conv(features)
        
        # Process each class independently
        class_outputs = []
        for class_idx, decoder in enumerate(self.class_decoders):
            class_output = decoder(features)
            class_outputs.append(class_output)
        
        # Stack class outputs
        stacked_outputs = torch.cat(class_outputs, dim=1)
        
        # Apply cross-class interaction
        final_outputs = stacked_outputs + self.interaction(stacked_outputs)
        
        return final_outputs


class ClassBalancedSegmentationHead(nn.Module):
    """Segmentation head with class-specific processing and balancing."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        mask_size: int = 56,
        use_class_specific: bool = True
    ):
        """Initialize class-balanced segmentation head.
        
        Args:
            in_channels: Input channels
            num_classes: Number of classes
            mask_size: Output size
            use_class_specific: Whether to use class-specific decoders
        """
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            ResidualBlock(in_channels),
            nn.Conv2d(in_channels, in_channels, 1)
        )
        
        # Main decoder
        if use_class_specific:
            self.decoder = ClassSpecificDecoder(
                in_channels=in_channels,
                num_classes=num_classes,
                mid_channels=128,
                mask_size=mask_size
            )
        else:
            # Standard decoder as fallback
            self.decoder = nn.Sequential(
                ResidualBlock(in_channels),
                nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
                LayerNorm2d(in_channels // 2),
                nn.ReLU(inplace=True),
                ResidualBlock(in_channels // 2),
                nn.Conv2d(in_channels // 2, num_classes, 1)
            )
        
        # Class-wise attention to prevent background dominance
        self.class_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_classes)
        ])
        
    def forward(self, features: torch.Tensor, rois: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with class balancing.
        
        Args:
            features: Input features
            rois: ROI boxes (optional, for compatibility with variable ROI models)
            
        Returns:
            Segmentation logits
        """
        # Note: This implementation doesn't use rois directly as it operates on 
        # already extracted ROI features. The rois parameter is for interface compatibility.
        
        # Refine features
        refined = self.feature_refine(features)
        
        # Apply class-wise attention
        attended_features = []
        for i, attention in enumerate(self.class_attention):
            att_map = attention(refined)
            if i == 0:  # Reduce background attention
                att_map = att_map * 0.5  # Dampening factor for background
            attended = refined * att_map
            attended_features.append(attended)
        
        # Use mean of attended features
        attended_mean = torch.stack(attended_features).mean(dim=0)
        
        # Decode
        logits = self.decoder(attended_mean)
        
        return logits


class InstanceAwareDecoder(nn.Module):
    """Decoder that maintains instance awareness to prevent class confusion."""
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        mask_size: int = 56,
        instance_dims: int = 64
    ):
        """Initialize instance-aware decoder.
        
        Args:
            in_channels: Input channels
            num_classes: Number of classes
            mask_size: Output size
            instance_dims: Dimensions for instance embedding
        """
        super().__init__()
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Instance embedding branch
        self.instance_branch = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            LayerNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels // 2),
            nn.Conv2d(in_channels // 2, instance_dims, 1),
            nn.ReLU(inplace=True)
        )
        
        # Class prediction branch (conditioned on instance features)
        self.class_branch = nn.Sequential(
            ResidualBlock(in_channels + instance_dims, in_channels),
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2),
            LayerNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            ResidualBlock(in_channels // 2)
        )
        
        # Separate heads for each class type
        self.bg_head = nn.Conv2d(in_channels // 2, 1, 1)
        self.instance_head = nn.Conv2d(in_channels // 2, 2, 1)  # Target & non-target
        
        # Instance discrimination loss preparation
        self.instance_proj = nn.Conv2d(instance_dims, 32, 1)
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with instance awareness.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with logits and instance embeddings
        """
        # Extract instance features
        instance_features = self.instance_branch(features)
        
        # Concatenate with original features
        combined = torch.cat([features, instance_features], dim=1)
        
        # Process through class branch
        class_features = self.class_branch(combined)
        
        # Separate predictions
        bg_logits = self.bg_head(class_features)
        instance_logits = self.instance_head(class_features)
        
        # Combine logits
        final_logits = torch.cat([bg_logits, instance_logits], dim=1)
        
        # Project instance features for discrimination loss
        instance_proj = self.instance_proj(instance_features)
        
        return {
            'logits': final_logits,
            'instance_embeddings': instance_proj,
            'bg_logits': bg_logits,
            'instance_logits': instance_logits
        }