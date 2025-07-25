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
        # Use fixed sizes to avoid dynamic shape operations in ONNX
        # For input size 28x28, enc2 will be 14x14
        up2 = F.interpolate(up2, size=(14, 14), mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc2], dim=1))

        up1 = self.up1(dec2)
        # For input size 28x28, enc1 will be 28x28
        up1 = F.interpolate(up1, size=(28, 28), mode='bilinear', align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

        # Final output
        out = self.final(dec1)

        return out


class EnhancedUNetONNX(nn.Module):
    """ONNX-compatible Enhanced UNet with fixed sizes for depth=3."""
    def __init__(self, in_channels: int, base_channels: int = 64):
        """Initialize enhanced UNet for fixed depth=3."""
        super().__init__()
        self.depth = 3  # Fixed depth
        
        # Encoder path - depth 3
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
            LayerNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 8, 3, padding=1),
            LayerNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 8),
            ResidualBlock(base_channels * 8)
        )
        
        # Decoder path
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            LayerNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 4)
        )
        
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            LayerNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels * 2)
        )
        
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),
            LayerNorm2d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(base_channels)
        )
        
        # Final output
        self.final = nn.Conv2d(base_channels, 2, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with fixed sizes for 28x28 input."""
        # Encoder
        enc1 = self.enc1(x)  # 28x28
        x = self.pool1(enc1)
        
        enc2 = self.enc2(x)  # 14x14
        x = self.pool2(enc2)
        
        enc3 = self.enc3(x)  # 7x7
        x = self.pool3(enc3)
        
        # Bottleneck at 3x3
        x = self.bottleneck(x)
        
        # Decoder with fixed interpolation sizes
        x = self.up3(x)
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, enc3], dim=1))
        
        x = self.up2(x)
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, enc2], dim=1))
        
        x = self.up1(x)
        x = F.interpolate(x, size=(28, 28), mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, enc1], dim=1))
        
        # Final output
        return self.final(x)

class EnhancedUNet(nn.Module):
    """Enhanced U-Net with deeper architecture and residual connections."""

    def __init__(self, in_channels: int, base_channels: int = 64, depth: int = 4):
        """Initialize enhanced U-Net.

        Args:
            in_channels: Number of input channels
            base_channels: Base channel count for the network
            depth: Depth of the U-Net (number of downsampling operations)
        """
        super().__init__()
        self.depth = depth

        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        channels = [in_channels] + [base_channels * (2**i) for i in range(depth)]

        for i in range(depth):
            if i == 0:
                # First encoder with input processing
                encoder = nn.Sequential(
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                    LayerNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True),
                    ResidualBlock(channels[i+1]),
                    ResidualBlock(channels[i+1])
                )
            else:
                # Deeper encoders with residual blocks
                encoder = nn.Sequential(
                    ResidualBlock(channels[i]),
                    ResidualBlock(channels[i]),
                    nn.Conv2d(channels[i], channels[i+1], 3, padding=1),
                    LayerNorm2d(channels[i+1]),
                    nn.ReLU(inplace=True)
                )

            self.encoders.append(encoder)
            if i < depth - 1:
                self.pools.append(nn.MaxPool2d(2))

        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(channels[-1]),
            ResidualBlock(channels[-1]),
            nn.Conv2d(channels[-1], channels[-1], 3, padding=1),
            LayerNorm2d(channels[-1]),
            nn.ReLU(inplace=True),
            # Spatial attention
            nn.Conv2d(channels[-1], channels[-1], 1),
            nn.Sigmoid(),
        )
        self.bottleneck_conv = nn.Conv2d(channels[-1], channels[-1], 3, padding=1)

        # Build decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for i in range(depth - 1, 0, -1):
            # Upsampling convolution
            self.upconvs.append(
                nn.ConvTranspose2d(channels[i+1], channels[i], 2, stride=2)
            )

            # Decoder block with residual connections
            decoder = nn.Sequential(
                nn.Conv2d(channels[i] * 2, channels[i], 3, padding=1),
                LayerNorm2d(channels[i]),
                nn.ReLU(inplace=True),
                ResidualBlock(channels[i]),
                ResidualBlock(channels[i])
            )
            self.decoders.append(decoder)

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(channels[1], channels[1] // 2, 3, padding=1),
            LayerNorm2d(channels[1] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[1] // 2, 2, 1)  # 2 classes: bg, fg
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced U-Net.

        Args:
            x: Input tensor (N, C, H, W)

        Returns:
            Logits for background/foreground (N, 2, H, W)
        """
        # Encoder path
        encoder_features = []

        for i in range(self.depth):
            x = self.encoders[i](x)
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pools[i](x)

        # Bottleneck with attention
        attention = self.bottleneck(x)
        x = self.bottleneck_conv(x) * attention

        # Decoder path
        for i in range(self.depth - 1):
            # Upsample
            x = self.upconvs[i](x)

            # Skip connection
            skip = encoder_features[self.depth - 2 - i]

            # Ensure sizes match - ConvTranspose2d might not produce exact size
            # Always interpolate to avoid TracerWarning during ONNX export
            # This ensures consistent behavior whether sizes match or not
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate and decode
            x = torch.cat([x, skip], dim=1)
            x = self.decoders[i](x)

        # Final output
        out = self.final(x)

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
                # Check if this is a model with integrated feature extractor
                if hasattr(self.base_model, 'feature_extractor'):
                    # This is actually images, not features
                    images = features  # Rename for clarity
                    # Extract features using the model's feature extractor
                    features = self.base_model.feature_extractor.extract_features(images)
                    # Now continue with the regular flow using extracted features
                    
                # Images passed in - need to extract features first
                elif hasattr(self.base_model, 'extractor'):
                    # Variable ROI model has integrated extractor
                    features = self.base_model.extractor.extract_features(features)
                elif not isinstance(features, dict):
                    # For other models, features should already be extracted
                    raise RuntimeError("Hierarchical model needs feature extractor or pre-extracted features")

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
                # Store fused features for auxiliary task
                self.last_roi_features = fused_features
            elif hasattr(self.base_model, 'segmentation_head') and hasattr(self.base_model.segmentation_head, 'ms_roi_align'):
                # For MultiScaleSegmentationModel, use its segmentation head's ROI align and fusion
                seg_head = self.base_model.segmentation_head
                roi_features = seg_head.ms_roi_align(features, rois)
                fused_features = seg_head.feature_fusion(roi_features)
                # Store fused features for auxiliary task
                self.last_roi_features = fused_features
            else:
                # Fallback for simpler models
                # Check if this is a head-only model which expects dict features
                if hasattr(self.base_model, 'segmentation_head') and (
                    self.base_model.__class__.__name__ == 'MultiScaleSegmentationHeadOnly' or
                    self.base_model.__class__.__name__ == 'VariableROISegmentationHeadOnly'
                ):
                    # Direct call with features dict
                    output = self.base_model(features, rois)
                    # For head-only models, we need to extract intermediate features
                    # This is a simplified approach - in production you'd extract from decoder
                    fused_features = output  # Use output as features (not ideal but works for ONNX export)
                elif hasattr(self.base_model, '__class__') and (
                    self.base_model.__class__.__name__ == 'MultiScaleHeadWithFeatures' or
                    self.base_model.__class__.__name__ == 'VariableROIHeadWithFeatures'
                ):
                    # These wrappers return features dict
                    output = self.base_model(features, rois)
                    fused_features = output['features']
                else:
                    output = self.base_model(features=features, rois=rois)
                    fused_features = output['features'] if isinstance(output, dict) else features

            # Store fused features for auxiliary task
            self.last_roi_features = fused_features
            # Apply hierarchical head
            logits, aux_outputs = self.hierarchical_head(fused_features)

            # Always return tuple for consistency
            return logits, aux_outputs

    return HierarchicalSegmentationModelUNet(base_model)


class HierarchicalSegmentationHeadUNetV2(nn.Module):
    """V2: Enhanced UNet for bg/fg, standard CNN for target/non-target."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56,
        dropout_rate: float = 0.1
    ):
        """Initialize hierarchical segmentation head V2.

        Args:
            in_channels: Input feature channels
            mid_channels: Intermediate channel count
            num_classes: Total number of classes (should be 3)
            mask_size: Output mask size
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"

        self.num_classes = num_classes
        self.mask_size = mask_size

        # Shared feature processing with dropout
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            ResidualBlock(mid_channels),
            nn.Dropout2d(dropout_rate),
            ResidualBlock(mid_channels),
        )

        # Branch 1: Background vs Foreground using Enhanced UNet
        self.bg_vs_fg_unet = EnhancedUNet(mid_channels, base_channels=96, depth=3)

        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

        # Branch 2: Target vs Non-target (standard CNN) with dropout
        self.target_vs_nontarget_branch = nn.Sequential(
            ResidualBlock(mid_channels),
            nn.Dropout2d(dropout_rate),
            nn.ConvTranspose2d(mid_channels, mid_channels // 2, 2, stride=2),
            LayerNorm2d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            ResidualBlock(mid_channels // 2),
            nn.Conv2d(mid_channels // 2, 2, 1)
        )

        # Enhanced gating with multi-scale attention and dropout
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5),  # Lower dropout for gate
            nn.Conv2d(mid_channels // 4, mid_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, mid_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with hierarchical prediction using enhanced UNet."""
        # Shared feature extraction
        shared = self.shared_features(features)

        # Branch 1: Background vs Foreground with Enhanced UNet
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
        batch_size = features.shape[0]
        final_logits = torch.zeros(batch_size, 3, self.mask_size, self.mask_size,
                                  device=features.device)

        final_logits[:, 0] = bg_fg_logits[:, 0]
        fg_mask = bg_fg_probs[:, 1:2]
        final_logits[:, 1] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 0] * fg_mask.squeeze(1)
        final_logits[:, 2] = bg_fg_logits[:, 1] + target_nontarget_logits[:, 1] * fg_mask.squeeze(1)

        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'fg_attention': fg_attention
        }

        return final_logits, aux_outputs


class HierarchicalSegmentationHeadUNetV3Static(nn.Module):
    """V3 fully static: No loops, no conditionals, for clean ONNX export."""
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize fully static hierarchical segmentation head V3."""
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"
        self.num_classes = num_classes
        self.mask_size = mask_size
        
        # Shared feature processing
        self.shared_conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.shared_norm = LayerNorm2d(mid_channels)
        self.shared_relu = nn.ReLU(inplace=True)
        self.shared_res1 = ResidualBlock(mid_channels)
        self.shared_res2 = ResidualBlock(mid_channels)
        
        # Branch 1: Background vs Foreground - Static UNet
        # Encoder
        self.bg_enc1 = nn.Sequential(
            nn.Conv2d(mid_channels, 96, 3, padding=1),
            LayerNorm2d(96),
            nn.ReLU(inplace=True),
            ResidualBlock(96)
        )
        self.bg_pool1 = nn.MaxPool2d(2)
        
        self.bg_enc2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            LayerNorm2d(192),
            nn.ReLU(inplace=True),
            ResidualBlock(192)
        )
        self.bg_pool2 = nn.MaxPool2d(2)
        
        self.bg_enc3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            LayerNorm2d(384),
            nn.ReLU(inplace=True),
            ResidualBlock(384)
        )
        self.bg_pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bg_bottleneck = nn.Sequential(
            nn.Conv2d(384, 768, 3, padding=1),
            LayerNorm2d(768),
            nn.ReLU(inplace=True),
            ResidualBlock(768),
            ResidualBlock(768)
        )
        
        # Decoder
        self.bg_up3 = nn.ConvTranspose2d(768, 384, 2, stride=2)
        self.bg_dec3 = nn.Sequential(
            nn.Conv2d(768, 384, 3, padding=1),
            LayerNorm2d(384),
            nn.ReLU(inplace=True),
            ResidualBlock(384)
        )
        
        self.bg_up2 = nn.ConvTranspose2d(384, 192, 2, stride=2)
        self.bg_dec2 = nn.Sequential(
            nn.Conv2d(384, 192, 3, padding=1),
            LayerNorm2d(192),
            nn.ReLU(inplace=True),
            ResidualBlock(192)
        )
        
        self.bg_up1 = nn.ConvTranspose2d(192, 96, 2, stride=2)
        self.bg_dec1 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),
            LayerNorm2d(96),
            nn.ReLU(inplace=True),
            ResidualBlock(96)
        )
        
        self.bg_final = nn.Conv2d(96, 2, 1)
        
        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        # Branch 2: Target vs Non-target - Shallow static UNet
        self.target_enc1 = nn.Sequential(
            nn.Conv2d(mid_channels, 64, 3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.target_pool1 = nn.MaxPool2d(2)
        
        self.target_bottom = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            LayerNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.target_up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.target_dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.target_final = nn.Conv2d(64, 2, 1)
        
        # Upsampling for target/non-target
        self.upsample_target = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        # Attention gates
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, mid_channels, 1),
            nn.Sigmoid()
        )
        
        self.target_gate = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with fully static operations."""
        # Shared feature extraction
        shared = self.shared_conv(features)
        shared = self.shared_norm(shared)
        shared = self.shared_relu(shared)
        shared = self.shared_res1(shared)
        shared = self.shared_res2(shared)
        
        # Branch 1: Background vs Foreground - Static UNet
        # Encoder
        bg_enc1 = self.bg_enc1(shared)  # 28x28
        bg_x = self.bg_pool1(bg_enc1)
        
        bg_enc2 = self.bg_enc2(bg_x)  # 14x14
        bg_x = self.bg_pool2(bg_enc2)
        
        bg_enc3 = self.bg_enc3(bg_x)  # 7x7
        bg_x = self.bg_pool3(bg_enc3)
        
        # Bottleneck at 3x3
        bg_x = self.bg_bottleneck(bg_x)
        
        # Decoder with fixed sizes
        bg_x = self.bg_up3(bg_x)
        bg_x = F.interpolate(bg_x, size=(7, 7), mode='bilinear', align_corners=False)
        bg_x = self.bg_dec3(torch.cat([bg_x, bg_enc3], dim=1))
        
        bg_x = self.bg_up2(bg_x)
        bg_x = F.interpolate(bg_x, size=(14, 14), mode='bilinear', align_corners=False)
        bg_x = self.bg_dec2(torch.cat([bg_x, bg_enc2], dim=1))
        
        bg_x = self.bg_up1(bg_x)
        bg_x = F.interpolate(bg_x, size=(28, 28), mode='bilinear', align_corners=False)
        bg_x = self.bg_dec1(torch.cat([bg_x, bg_enc1], dim=1))
        
        bg_fg_logits_low = self.bg_final(bg_x)
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        # Create foreground attention gate
        fg_attention = self.fg_gate(bg_fg_logits_low)
        
        # Branch 2: Target vs Non-target - Static shallow UNet
        gated_features = shared * fg_attention
        
        # Encoder
        target_enc1 = self.target_enc1(gated_features)  # 28x28
        target_x = self.target_pool1(target_enc1)  # 14x14
        
        # Bottom
        target_x = self.target_bottom(target_x)
        
        # Decoder
        target_x = self.target_up1(target_x)
        target_x = F.interpolate(target_x, size=(28, 28), mode='bilinear', align_corners=False)
        target_x = self.target_dec1(torch.cat([target_x, target_enc1], dim=1))
        
        target_logits_low = self.target_final(target_x)
        target_nontarget_logits = self.upsample_target(target_logits_low)
        
        # Additional gating from target branch
        target_attention = self.target_gate(target_logits_low)
        
        # Upsample target attention to 56x56
        target_attention_upsampled = F.interpolate(
            target_attention, 
            size=(56, 56),
            mode='bilinear', 
            align_corners=False
        )
        
        # Combine predictions - avoid torch.zeros for cleaner ONNX
        # Background channel
        bg_channel = bg_fg_logits[:, 0:1]  # Keep dims
        
        # Foreground channels with gating
        fg_mask = bg_fg_probs[:, 1:2]
        fg_base = bg_fg_logits[:, 1:2]
        
        # Target channel
        target_channel = fg_base + target_nontarget_logits[:, 0:1] * fg_mask * target_attention_upsampled
        
        # Non-target channel
        nontarget_channel = fg_base + target_nontarget_logits[:, 1:2] * fg_mask
        
        # Stack channels
        final_logits = torch.cat([bg_channel, target_channel, nontarget_channel], dim=1)
        
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'target_logits_low': target_logits_low,
            'fg_attention': fg_attention,
            'target_attention': target_attention
        }
        
        return final_logits, aux_outputs

class HierarchicalSegmentationHeadUNetV3ONNX(nn.Module):
    """V3 ONNX-compatible: Enhanced UNet for bg/fg, Shallow UNet for target/non-target."""
    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head V3 for ONNX."""
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
        
        # Branch 1: Background vs Foreground using ONNX-compatible Enhanced UNet
        self.bg_vs_fg_unet = EnhancedUNetONNX(mid_channels, base_channels=96)
        
        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        # Branch 2: Target vs Non-target using Shallow UNet
        self.target_nontarget_unet = ShallowUNet(mid_channels, base_channels=64)
        
        # Upsampling for target/non-target
        self.upsample_target = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )
        
        # Dual attention gates
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, 1),
            nn.Sigmoid()
        )
        
        self.target_gate = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with dual UNet branches."""
        # Shared feature extraction
        shared = self.shared_features(features)
        
        # Branch 1: Background vs Foreground with Enhanced UNet
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)
        
        # Create foreground attention gate
        fg_attention = self.fg_gate(bg_fg_logits_low)
        
        # Branch 2: Target vs Non-target with Shallow UNet
        gated_features = shared * fg_attention
        target_logits_low = self.target_nontarget_unet(gated_features)
        target_nontarget_logits = self.upsample_target(target_logits_low)
        
        # Additional gating from target branch
        target_attention = self.target_gate(target_logits_low)
        
        # Upsample target attention to match final resolution (always 56x56)
        target_attention_upsampled = F.interpolate(
            target_attention, 
            size=(56, 56),
            mode='bilinear', 
            align_corners=False
        )
        
        # Combine predictions hierarchically with dual gating
        batch_size = features.shape[0]
        
        # Initialize output tensor with fixed size
        final_logits = torch.zeros(batch_size, 3, 56, 56, dtype=features.dtype, device=features.device)
        
        # Direct assignment without indexing
        final_logits[:, 0] = bg_fg_logits[:, 0]
        
        # Compute foreground components
        fg_mask = bg_fg_probs[:, 1:2]
        fg_base = bg_fg_logits[:, 1:2]
        
        # Compute target and non-target with consistent dimensions
        target_component = target_nontarget_logits[:, 0:1] * fg_mask * target_attention_upsampled
        nontarget_component = target_nontarget_logits[:, 1:2] * fg_mask
        
        final_logits[:, 1] = (fg_base + target_component).squeeze(1)
        final_logits[:, 2] = (fg_base + nontarget_component).squeeze(1)
        
        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'target_logits_low': target_logits_low,
            'fg_attention': fg_attention,
            'target_attention': target_attention
        }
        
        return final_logits, aux_outputs

class HierarchicalSegmentationHeadUNetV3(nn.Module):
    """V3: Enhanced UNet for bg/fg, Shallow UNet for target/non-target."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head V3."""
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

        # Branch 1: Background vs Foreground using Enhanced UNet
        self.bg_vs_fg_unet = EnhancedUNet(mid_channels, base_channels=96, depth=3)

        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

        # Branch 2: Target vs Non-target using Shallow UNet
        self.target_nontarget_unet = ShallowUNet(mid_channels, base_channels=64)

        # Upsampling for target/non-target
        self.upsample_target = nn.Sequential(
            nn.ConvTranspose2d(2, 32, 2, stride=2),
            LayerNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)
        )

        # Dual attention gates
        self.fg_gate = nn.Sequential(
            nn.Conv2d(2, mid_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, 1),
            nn.Sigmoid()
        )

        self.target_gate = nn.Sequential(
            nn.Conv2d(2, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with dual UNet branches."""
        # Shared feature extraction
        shared = self.shared_features(features)

        # Branch 1: Background vs Foreground with Enhanced UNet
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)
        bg_fg_probs = F.softmax(bg_fg_logits, dim=1)

        # Create foreground attention gate
        fg_attention = self.fg_gate(bg_fg_logits_low)

        # Branch 2: Target vs Non-target with Shallow UNet
        gated_features = shared * fg_attention
        target_logits_low = self.target_nontarget_unet(gated_features)
        target_nontarget_logits = self.upsample_target(target_logits_low)

        # Additional gating from target branch
        target_attention = self.target_gate(target_logits_low)
        
        # Upsample target attention to match final resolution
        # Use fixed size to avoid dynamic shape operations
        target_attention_upsampled = F.interpolate(
            target_attention, 
            size=(56, 56),  # Use fixed mask_size instead of self.mask_size
            mode='bilinear', 
            align_corners=False
        )

        # Combine predictions hierarchically with dual gating
        # Use shape[0] only once and reuse
        batch_size = features.shape[0]
        
        # Initialize output tensor
        final_logits = torch.zeros(batch_size, 3, 56, 56, dtype=features.dtype, device=features.device)

        # Direct assignment without indexing
        final_logits[:, 0] = bg_fg_logits[:, 0]
        
        # Compute foreground components
        fg_mask = bg_fg_probs[:, 1:2]
        fg_base = bg_fg_logits[:, 1:2]  # Keep dimensions
        
        # Compute target and non-target with consistent dimensions
        target_component = target_nontarget_logits[:, 0:1] * fg_mask * target_attention_upsampled
        nontarget_component = target_nontarget_logits[:, 1:2] * fg_mask
        
        final_logits[:, 1] = (fg_base + target_component).squeeze(1)
        final_logits[:, 2] = (fg_base + nontarget_component).squeeze(1)

        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'target_logits_low': target_logits_low,
            'fg_attention': fg_attention,
            'target_attention': target_attention
        }

        return final_logits, aux_outputs


class HierarchicalSegmentationHeadUNetV4(nn.Module):
    """V4: Enhanced UNet for both bg/fg and target/non-target branches."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 256,
        num_classes: int = 3,
        mask_size: int = 56
    ):
        """Initialize hierarchical segmentation head V4."""
        super().__init__()
        assert num_classes == 3, "Hierarchical model designed for 3 classes"

        self.num_classes = num_classes
        self.mask_size = mask_size

        # Shared feature processing with stronger backbone
        self.shared_features = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            LayerNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),
            ResidualBlock(mid_channels),  # Extra depth
        )

        # Branch 1: Background vs Foreground using Enhanced UNet
        self.bg_vs_fg_unet = EnhancedUNet(mid_channels, base_channels=128, depth=4)

        # Upsampling to match mask_size
        self.upsample_bg_fg = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 2, stride=2),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 2, 1)
        )

        # Branch 2: Target vs Non-target using Enhanced UNet
        self.target_nontarget_unet = EnhancedUNet(mid_channels, base_channels=96, depth=3)

        # Upsampling for target/non-target with residual
        self.upsample_target = nn.Sequential(
            nn.ConvTranspose2d(2, 64, 2, stride=2),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 2, 1)
        )

        # Cross-attention between branches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=4,  # 2 classes per branch
            num_heads=1,
            batch_first=True
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            LayerNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with dual enhanced UNet branches and cross-attention."""
        # Shared feature extraction
        shared = self.shared_features(features)

        # Branch 1: Background vs Foreground with Enhanced UNet
        bg_fg_logits_low = self.bg_vs_fg_unet(shared)
        bg_fg_logits = self.upsample_bg_fg(bg_fg_logits_low)

        # Branch 2: Target vs Non-target with Enhanced UNet
        target_logits_low = self.target_nontarget_unet(shared)
        target_nontarget_logits = self.upsample_target(target_logits_low)

        # Cross-attention between branches
        batch_size = features.shape[0]
        h, w = bg_fg_logits.shape[2:]

        # Reshape for attention (B, HW, C)
        bg_fg_flat = bg_fg_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)
        target_flat = target_nontarget_logits.permute(0, 2, 3, 1).reshape(batch_size, -1, 2)

        # Concatenate for cross-attention
        combined = torch.cat([bg_fg_flat, target_flat], dim=-1)  # (B, HW, 4)
        attended, _ = self.cross_attention(combined, combined, combined)

        # Reshape back
        attended = attended.reshape(batch_size, h, w, 4).permute(0, 3, 1, 2)

        # Final fusion
        final_logits = self.fusion(attended)

        aux_outputs = {
            'bg_fg_logits': bg_fg_logits,
            'bg_fg_logits_low': bg_fg_logits_low,
            'target_nontarget_logits': target_nontarget_logits,
            'target_logits_low': target_logits_low,
            'attended_features': attended
        }

        return final_logits, aux_outputs


def create_hierarchical_model_unet_v2(base_model: nn.Module) -> nn.Module:
    """Create V2: Enhanced UNet for bg/fg only."""
    class HierarchicalSegmentationModelUNetV2(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

            # Replace the segmentation head
            if hasattr(base_model, 'segmentation_head'):
                old_head = base_model.segmentation_head
                if hasattr(old_head, 'decoder'):
                    first_layer = old_head.decoder[0]
                    if isinstance(first_layer, ResidualBlock):
                        in_channels = first_layer.conv1.in_channels
                    else:
                        in_channels = 256
                else:
                    in_channels = 256

                self.hierarchical_head = HierarchicalSegmentationHeadUNetV2(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )

                if hasattr(old_head, 'var_roi_align'):
                    self.roi_align = old_head.var_roi_align
                if hasattr(old_head, 'feature_fusion'):
                    self.feature_fusion = old_head.feature_fusion

        def forward(self, features, rois, rgb_features=None):
            # Same forward logic as V1
            if isinstance(features, torch.Tensor):
                # Check if this is a model with integrated feature extractor
                if hasattr(self.base_model, 'feature_extractor'):
                    # This is actually images, not features
                    images = features  # Rename for clarity
                    # Extract features using the model's feature extractor
                    features = self.base_model.feature_extractor.extract_features(images)
                    # Now continue with the regular flow using extracted features
                    
                # Images passed in - need to extract features first
                elif hasattr(self.base_model, 'extractor'):
                    # Variable ROI model has integrated extractor
                    features = self.base_model.extractor.extract_features(features)
                elif not isinstance(features, dict):
                    # For other models, features should already be extracted
                    raise RuntimeError("Hierarchical model needs feature extractor or pre-extracted features")

            if hasattr(self, 'roi_align'):
                roi_features = self.roi_align(features, rois)

                if rgb_features is not None and hasattr(self.base_model, 'segmentation_head'):
                    old_head = self.base_model.segmentation_head
                    if hasattr(old_head, 'rgb_roi_aligns'):
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

                if hasattr(self, 'feature_fusion'):
                    fused_features = self.feature_fusion(roi_features)
                else:
                    fused_features = list(roi_features.values())[0]
                # Store fused features for auxiliary task
                self.last_roi_features = fused_features
            elif hasattr(self.base_model, 'segmentation_head') and hasattr(self.base_model.segmentation_head, 'ms_roi_align'):
                # For MultiScaleSegmentationModel, use its segmentation head's ROI align and fusion
                seg_head = self.base_model.segmentation_head
                roi_features = seg_head.ms_roi_align(features, rois)
                fused_features = seg_head.feature_fusion(roi_features)
                # Store fused features for auxiliary task
                self.last_roi_features = fused_features
            else:
                # For other models, call with appropriate arguments
                # Check if this is a head-only model which expects dict features
                if hasattr(self.base_model, 'segmentation_head') and (
                    self.base_model.__class__.__name__ == 'MultiScaleSegmentationHeadOnly' or
                    self.base_model.__class__.__name__ == 'VariableROISegmentationHeadOnly'
                ):
                    # Direct call with features dict
                    output = self.base_model(features, rois)
                    # For head-only models, we need to extract intermediate features
                    # This is a simplified approach - in production you'd extract from decoder
                    fused_features = output  # Use output as features (not ideal but works for ONNX export)
                elif hasattr(self.base_model, '__class__') and (
                    self.base_model.__class__.__name__ == 'MultiScaleHeadWithFeatures' or
                    self.base_model.__class__.__name__ == 'VariableROIHeadWithFeatures'
                ):
                    # These wrappers return features dict
                    output = self.base_model(features, rois)
                    fused_features = output['features']
                else:
                    output = self.base_model(features=features, rois=rois)
                    fused_features = output['features'] if isinstance(output, dict) else features

            # Store fused features for auxiliary task  
            self.last_roi_features = fused_features
            logits, aux_outputs = self.hierarchical_head(fused_features)
            return logits, aux_outputs

    return HierarchicalSegmentationModelUNetV2(base_model)


def create_hierarchical_model_unet_v3(base_model: nn.Module) -> nn.Module:
    """Create V3: Enhanced UNet for bg/fg, Shallow UNet for target/non-target."""
    class HierarchicalSegmentationModelUNetV3(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

            if hasattr(base_model, 'segmentation_head'):
                old_head = base_model.segmentation_head
                if hasattr(old_head, 'decoder'):
                    first_layer = old_head.decoder[0]
                    if isinstance(first_layer, ResidualBlock):
                        in_channels = first_layer.conv1.in_channels
                    else:
                        in_channels = 256
                else:
                    in_channels = 256

                self.hierarchical_head = HierarchicalSegmentationHeadUNetV3(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )

                if hasattr(old_head, 'var_roi_align'):
                    self.roi_align = old_head.var_roi_align
                if hasattr(old_head, 'feature_fusion'):
                    self.feature_fusion = old_head.feature_fusion

        def forward(self, features, rois, rgb_features=None):
            # Same forward logic
            if isinstance(features, torch.Tensor):
                if hasattr(self.base_model, 'extractor'):
                    features = self.base_model.extractor.extract_features(features)
                else:
                    raise RuntimeError("Hierarchical model needs feature extractor")

            if hasattr(self, 'roi_align'):
                roi_features = self.roi_align(features, rois)

                if rgb_features is not None and hasattr(self.base_model, 'segmentation_head'):
                    old_head = self.base_model.segmentation_head
                    if hasattr(old_head, 'rgb_roi_aligns'):
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

                if hasattr(self, 'feature_fusion'):
                    fused_features = self.feature_fusion(roi_features)
                else:
                    fused_features = list(roi_features.values())[0]
            else:
                # For MultiScaleSegmentationModel, pass images and rois as positional args
                output = self.base_model(features, rois)
                fused_features = output['features'] if isinstance(output, dict) else output

            # Store fused features for auxiliary task  
            self.last_roi_features = fused_features
            logits, aux_outputs = self.hierarchical_head(fused_features)
            return logits, aux_outputs

    return HierarchicalSegmentationModelUNetV3(base_model)


def create_hierarchical_model_unet_v4(base_model: nn.Module) -> nn.Module:
    """Create V4: Enhanced UNet for both branches."""
    class HierarchicalSegmentationModelUNetV4(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model

            if hasattr(base_model, 'segmentation_head'):
                old_head = base_model.segmentation_head
                if hasattr(old_head, 'decoder'):
                    first_layer = old_head.decoder[0]
                    if isinstance(first_layer, ResidualBlock):
                        in_channels = first_layer.conv1.in_channels
                    else:
                        in_channels = 256
                else:
                    in_channels = 256

                self.hierarchical_head = HierarchicalSegmentationHeadUNetV4(
                    in_channels=in_channels,
                    mid_channels=256,
                    num_classes=3,
                    mask_size=56
                )

                if hasattr(old_head, 'var_roi_align'):
                    self.roi_align = old_head.var_roi_align
                if hasattr(old_head, 'feature_fusion'):
                    self.feature_fusion = old_head.feature_fusion

        def forward(self, features, rois, rgb_features=None):
            # Same forward logic
            if isinstance(features, torch.Tensor):
                if hasattr(self.base_model, 'extractor'):
                    features = self.base_model.extractor.extract_features(features)
                else:
                    raise RuntimeError("Hierarchical model needs feature extractor")

            if hasattr(self, 'roi_align'):
                roi_features = self.roi_align(features, rois)

                if rgb_features is not None and hasattr(self.base_model, 'segmentation_head'):
                    old_head = self.base_model.segmentation_head
                    if hasattr(old_head, 'rgb_roi_aligns'):
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

                if hasattr(self, 'feature_fusion'):
                    fused_features = self.feature_fusion(roi_features)
                else:
                    fused_features = list(roi_features.values())[0]
            else:
                # For MultiScaleSegmentationModel, pass images and rois as positional args
                output = self.base_model(features, rois)
                fused_features = output['features'] if isinstance(output, dict) else output

            # Store fused features for auxiliary task  
            self.last_roi_features = fused_features
            logits, aux_outputs = self.hierarchical_head(fused_features)
            return logits, aux_outputs

    return HierarchicalSegmentationModelUNetV4(base_model)