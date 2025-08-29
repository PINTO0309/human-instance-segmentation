"""Bilateral filter implementation for ONNX export."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BilateralFilter(nn.Module):
    """Bilateral filter implementation compatible with ONNX export.

    This implementation uses Gaussian kernels for both spatial and range domains.
    The filter is implemented using basic operations to ensure ONNX compatibility.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma_spatial: float = 1.0,
        sigma_range: float = 0.1,
    ):
        """Initialize bilateral filter.

        Args:
            kernel_size: Size of the filter kernel (must be odd)
            sigma_spatial: Standard deviation for spatial Gaussian kernel
            sigma_range: Standard deviation for range (intensity) Gaussian kernel
        """
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.padding = kernel_size // 2

        # Create spatial Gaussian kernel
        self.register_buffer('spatial_kernel', self._create_spatial_kernel())

    def _create_spatial_kernel(self) -> torch.Tensor:
        """Create spatial Gaussian kernel."""
        # Create coordinate grids
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - (self.kernel_size - 1) / 2

        # Create 2D spatial kernel
        y_coords = coords.view(-1, 1).expand(self.kernel_size, self.kernel_size)
        x_coords = coords.view(1, -1).expand(self.kernel_size, self.kernel_size)

        # Calculate spatial distances
        spatial_dist = (x_coords ** 2 + y_coords ** 2)

        # Apply Gaussian
        spatial_kernel = torch.exp(-spatial_dist / (2 * self.sigma_spatial ** 2))

        return spatial_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filter.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Filtered tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Pad input
        x_padded = F.pad(x, [self.padding] * 4, mode='reflect')

        # Initialize output
        output = torch.zeros_like(x)

        # Apply filter for each channel
        for c in range(C):
            channel_data = x_padded[:, c:c+1, :, :]  # Shape: (B, 1, H+pad, W+pad)

            # Process each position
            for i in range(H):
                for j in range(W):
                    # Extract local patch
                    patch = channel_data[
                        :, :,
                        i:i+self.kernel_size,
                        j:j+self.kernel_size
                    ]  # Shape: (B, 1, kernel_size, kernel_size)

                    # Center pixel value
                    center_val = channel_data[:, :, i+self.padding, j+self.padding]
                    center_val = center_val.view(B, 1, 1, 1)

                    # Compute range (intensity) weights
                    range_diff = patch - center_val
                    range_weights = torch.exp(
                        -range_diff ** 2 / (2 * self.sigma_range ** 2)
                    )

                    # Combine spatial and range weights
                    spatial_weights = self.spatial_kernel.view(1, 1, self.kernel_size, self.kernel_size)
                    weights = spatial_weights * range_weights

                    # Normalize weights
                    weights_sum = weights.sum(dim=(2, 3), keepdim=True)
                    weights_norm = weights / (weights_sum + 1e-8)

                    # Apply weights
                    filtered_val = (patch * weights_norm).sum(dim=(2, 3))
                    output[:, c, i, j] = filtered_val.squeeze()

        return output


class FastBilateralFilter(nn.Module):
    """Fast approximation of bilateral filter using separable convolutions.

    This version is more efficient and better suited for ONNX export.
    """

    def __init__(
        self,
        kernel_size: int = 5,
        sigma_spatial: float = 1.0,
        sigma_range: float = 0.1,
        num_iterations: int = 2,
    ):
        """Initialize fast bilateral filter.

        Args:
            kernel_size: Size of the filter kernel (must be odd)
            sigma_spatial: Standard deviation for spatial Gaussian kernel
            sigma_range: Standard deviation for range (intensity) Gaussian kernel
            num_iterations: Number of filtering iterations
        """
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.num_iterations = num_iterations
        self.padding = kernel_size // 2

        # Create 1D Gaussian kernel for separable convolution
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma_spatial ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Register as buffers
        self.register_buffer('kernel_h', kernel_1d.view(1, 1, 1, kernel_size))
        self.register_buffer('kernel_v', kernel_1d.view(1, 1, kernel_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fast bilateral filter.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Filtered tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Process each channel separately
        channels = []
        for c in range(C):
            channel = x[:, c:c+1, :, :]

            # Iterative filtering
            for _ in range(self.num_iterations):
                # Spatial filtering (separable convolution)
                filtered = F.conv2d(
                    channel,
                    self.kernel_h,
                    padding=(0, self.padding)
                )
                filtered = F.conv2d(
                    filtered,
                    self.kernel_v,
                    padding=(self.padding, 0)
                )

                # Range filtering approximation
                # Use local variance as a proxy for edge detection
                local_mean = filtered
                local_sq_mean = F.conv2d(
                    channel ** 2,
                    self.kernel_h,
                    padding=(0, self.padding)
                )
                local_sq_mean = F.conv2d(
                    local_sq_mean,
                    self.kernel_v,
                    padding=(self.padding, 0)
                )

                local_var = local_sq_mean - local_mean ** 2
                local_var = torch.clamp(local_var, min=0)

                # Adaptive weights based on local variance
                edge_weight = torch.exp(-local_var / (2 * self.sigma_range ** 2))

                # Blend original and filtered based on edge weight
                channel = edge_weight * filtered + (1 - edge_weight) * channel

            channels.append(channel)

        # Concatenate channels
        output = torch.cat(channels, dim=1)

        return output


class EdgePreservingFilter(nn.Module):
    """Edge-preserving filter using guided filtering approach.

    This is an alternative to bilateral filtering that's more ONNX-friendly.
    """

    def __init__(
        self,
        radius: int = 2,
        eps: float = 0.01,
    ):
        """Initialize edge-preserving filter.

        Args:
            radius: Filter radius
            eps: Regularization parameter
        """
        super().__init__()

        self.radius = radius
        self.eps = eps
        self.kernel_size = 2 * radius + 1

        # Create box filter kernel
        kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel / (self.kernel_size ** 2)
        self.register_buffer('box_kernel', kernel)

    def _box_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply box filter to each channel."""
        B, C, H, W = x.shape

        # Process each channel
        channels = []
        for c in range(C):
            filtered = F.conv2d(
                x[:, c:c+1, :, :],
                self.box_kernel,
                padding=self.radius
            )
            channels.append(filtered)

        return torch.cat(channels, dim=1)

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply edge-preserving filter.

        Args:
            x: Input tensor of shape (B, C, H, W)
            guide: Guide image (if None, uses input as guide)

        Returns:
            Filtered tensor of shape (B, C, H, W)
        """
        if guide is None:
            guide = x

        # Compute local statistics
        mean_x = self._box_filter(x)
        mean_g = self._box_filter(guide)

        corr_xg = self._box_filter(x * guide)
        corr_gg = self._box_filter(guide * guide)

        cov_xg = corr_xg - mean_x * mean_g
        var_g = corr_gg - mean_g * mean_g

        # Compute coefficients
        a = cov_xg / (var_g + self.eps)
        b = mean_x - a * mean_g

        # Apply filter
        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        output = mean_a * guide + mean_b

        return output


class BinaryMaskBilateralFilter(nn.Module):
    """Bilateral filter optimized for binary masks.
    
    This implementation is specifically designed for cleaning up binary segmentation masks
    while preserving edges. It uses morphological-like operations combined with 
    edge-aware smoothing.
    """
    
    def __init__(
        self,
        kernel_size: int = 7,
        sigma_spatial: float = 1.5,
        threshold: float = 0.5,
        num_iterations: int = 2,
    ):
        """Initialize binary mask bilateral filter.
        
        Args:
            kernel_size: Size of the filter kernel (must be odd)
            sigma_spatial: Standard deviation for spatial Gaussian kernel
            threshold: Threshold for binary decision (default 0.5)
            num_iterations: Number of filtering iterations
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
            
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.threshold = threshold
        self.num_iterations = num_iterations
        self.padding = kernel_size // 2
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2
        
        # 2D Gaussian kernel
        y_coords = coords.view(-1, 1).expand(kernel_size, kernel_size)
        x_coords = coords.view(1, -1).expand(kernel_size, kernel_size)
        
        gaussian_kernel = torch.exp(
            -(x_coords ** 2 + y_coords ** 2) / (2 * sigma_spatial ** 2)
        )
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bilateral filter to binary mask.
        
        Args:
            x: Input binary mask tensor of shape (B, C, H, W)
               Values should be in [0, 1] range
            
        Returns:
            Filtered binary mask tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Ensure input is in [0, 1] range
        x = torch.clamp(x, 0, 1)
        
        # Process each channel
        channels = []
        for c in range(C):
            mask = x[:, c:c+1, :, :]
            
            # Apply iterative filtering
            for _ in range(self.num_iterations):
                # Weighted average using Gaussian kernel
                filtered = F.conv2d(
                    mask,
                    self.gaussian_kernel,
                    padding=self.padding
                )
                
                # Edge-aware weighting
                # Compute local variance to detect edges
                mask_sq = mask ** 2
                local_mean = filtered
                local_mean_sq = F.conv2d(
                    mask_sq,
                    self.gaussian_kernel,
                    padding=self.padding
                )
                
                # Variance indicates edge strength
                local_var = local_mean_sq - local_mean ** 2
                local_var = torch.clamp(local_var, min=0)
                
                # High variance = edge, low variance = smooth region
                edge_weight = torch.exp(-local_var * 10)  # Amplify edge detection
                
                # Apply edge-preserving smoothing
                mask = edge_weight * filtered + (1 - edge_weight) * mask
                
            channels.append(mask)
            
        # Concatenate channels
        output = torch.cat(channels, dim=1)
        
        # Apply threshold to get binary output
        output = (output > self.threshold).float()
        
        return output


class MorphologicalBilateralFilter(nn.Module):
    """Morphological-inspired bilateral filter for binary masks.
    
    Combines morphological operations with bilateral filtering for 
    better structure preservation in binary masks.
    """
    
    def __init__(
        self,
        kernel_size: int = 5,
        sigma: float = 1.0,
        morph_size: int = 3,
    ):
        """Initialize morphological bilateral filter.
        
        Args:
            kernel_size: Size of bilateral filter kernel
            sigma: Standard deviation for Gaussian kernel
            morph_size: Size of morphological structuring element
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.morph_size = morph_size
        self.padding = kernel_size // 2
        self.morph_padding = morph_size // 2
        
        # Create Gaussian kernel for bilateral filtering
        coords = torch.arange(kernel_size, dtype=torch.float32)
        coords = coords - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # 2D kernel
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        kernel_2d = kernel_2d.view(1, 1, kernel_size, kernel_size)
        
        self.register_buffer('bilateral_kernel', kernel_2d)
        
        # Create morphological kernels
        morph_kernel = torch.ones(1, 1, morph_size, morph_size)
        self.register_buffer('morph_kernel', morph_kernel)
        
    def _morphological_close(self, x: torch.Tensor) -> torch.Tensor:
        """Apply morphological closing (dilation followed by erosion)."""
        # Dilation
        dilated = F.max_pool2d(x, self.morph_size, stride=1, padding=self.morph_padding)
        
        # Erosion
        eroded = -F.max_pool2d(-dilated, self.morph_size, stride=1, padding=self.morph_padding)
        
        return eroded
        
    def _morphological_open(self, x: torch.Tensor) -> torch.Tensor:
        """Apply morphological opening (erosion followed by dilation)."""
        # Erosion
        eroded = -F.max_pool2d(-x, self.morph_size, stride=1, padding=self.morph_padding)
        
        # Dilation
        dilated = F.max_pool2d(eroded, self.morph_size, stride=1, padding=self.morph_padding)
        
        return dilated
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply morphological bilateral filter.
        
        Args:
            x: Input binary mask tensor of shape (B, C, H, W)
            
        Returns:
            Filtered binary mask tensor of shape (B, C, H, W)
        """
        # Ensure binary input
        x = torch.clamp(x, 0, 1)
        
        # Apply morphological opening to remove small noise
        x_opened = self._morphological_open(x)
        
        # Apply bilateral filtering
        x_filtered = F.conv2d(
            x_opened,
            self.bilateral_kernel,
            padding=self.padding
        )
        
        # Apply morphological closing to fill small holes
        x_closed = self._morphological_close(x_filtered)
        
        # Final thresholding
        output = (x_closed > 0.5).float()
        
        return output