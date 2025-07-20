"""Hierarchical segmentation with UNet-based foreground/background separation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..model import LayerNorm2d, ResidualBlock


class ShallowUNet(nn.Module):
    """Shallow U-Net for foreground/background separation."""
    
    def __init__(self, in_channels: int, base_channels: int = 64):
        """Initialize shallow U-Net.
        
        Args:
            in_channels: Number of input channels
            base_channels: Base channel count for the network
        """
        super().__init__()
        
        # Encoder path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        # Bottom
        self.bottom = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            LayerNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            LayerNorm2d(base_channels * 4),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Conv2d(base_channels, 2, 1)  # 2 classes: bg, fg
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Args:
            x: Input tensor (N, C, H, W)
            
        Returns:
            Logits for background/foreground (N, 2, H, W)
        """
        # Encoder
        enc1 = self.enc1(x)
        x1 = self.pool1(enc1)
        
        enc2 = self.enc2(x1)
        x2 = self.pool2(enc2)
        
        # Bottom
        bottom = self.bottom(x2)
        
        # Decoder
        up2 = self.up2(bottom)
        # Handle size mismatch
        if up2.shape[2:] != enc2.shape[2:]:
            up2 = F.interpolate(up2, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
        
        up1 = self.up1(dec2)
        # Handle size mismatch
        if up1.shape[2:] != enc1.shape[2:]:
            up1 = F.interpolate(up1, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))
        
        # Final output
        out = self.final(dec1)
        
        return out


class HierarchicalSegmentationHeadUNet(nn.Module):
    """Hierarchical segmentation head with UNet-based foreground/background separation."""
    
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head with UNet.
        
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
        
        # Branch 1: Background vs Foreground using UNet
        self.bg_vs_fg_unet = ShallowUNet(mid_channels, base_channels=128)
        
        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
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
        
        # Enhanced gating mechanism using UNet features
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical prediction using UNet.
        
        Args:
            features: Input features (N, C, H, W)
            
        Returns:
            final_logits: Combined 3-class logits (N, 3, H, W)
            aux_outputs: Dictionary with intermediate predictions
        """
        # Shared feature extraction
        shared = self.shared_features(features)
        
        # Branch 1: Background vs Foreground with UNet
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        
        # Upsample to mask size
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        # Create foreground attention gate from UNet features
        fg_attention = self.fg_gate(bg_fg_logits_low)
        
        # Branch 2: Target vs Non-target (modulated by foreground attention)
        gated_features = shared * fg_attention
        target_nontarget_logits = self.target_vs_nontarget_branch(gated_features)
        
        # Combine predictions hierarchically
        # Final logits: [background, target, non-target]
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_size, self.mask_size, 
                                  device=features.device)
        
        # Background probability from UNet branch
        final_logits[:, 0] = bg_fg_logits[:, 0]  # Background logit
        
        # Target and non-target from second branch, scaled by foreground probability
        # This ensures target/non-target predictions are only strong in foreground regions
        fg_mask = bg_fg_probs[:, 1:2]  # Foreground probability
        final_logits[:, 1] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 0] * fg_mask.squeeze(1)
        final_logits[:, 2] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 1] * fg_mask.squeeze(1)
        
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention
        }
        
        return final_logits, aux_outputs


def create_hierarchical_model_unet(base_model: nn.Module) -> nn.Module:
    """Wrap a base model with UNet-based hierarchical segmentation head.
    
    Args:
        base_model: Base segmentation model
        
    Returns:
        Model with UNet-based hierarchical segmentation
    """
    class HierarchicalSegmentationModelUNet(nn.Module):
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
                
                # Create UNet-based hierarchical head
                self.hierarchical_head = HierarchicalSegmentationHeadUNet(
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
    
    return HierarchicalSegmentationModelUNet(base_model)