"""Binary mask edge smoothing implementation for segmentation outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class BinaryMaskEdgeSmoothing(nn.Module):
    """バイナリマスクのエッジ平滑化PyTorchモデル"""

    def __init__(self, threshold: float = 0.5, blur_strength: float = 3.0):
        super().__init__()
        self.threshold = threshold
        self.blur_strength = blur_strength

        # Laplacianカーネル（エッジ検出用）
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

        # ガウシアンカーネル（3x3）
        gaussian_kernel = torch.tensor([
            [1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('gaussian_kernel', gaussian_kernel)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply edge smoothing to binary mask.
        
        Args:
            mask: Binary mask tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
        
        Returns:
            Smoothed binary mask with same shape as input
        """
        # Store original shape and add batch/channel dims if needed
        original_shape = mask.shape
        if mask.dim() == 2:  # (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
        elif mask.dim() == 3:  # (C, H, W)
            mask = mask.unsqueeze(0)  # -> (1, C, H, W)
        
        B, C, H, W = mask.shape
        device = mask.device
        dtype = mask.dtype
        
        # Process each channel separately
        smoothed_channels = []
        for c in range(C):
            channel_mask = mask[:, c:c+1, :, :].float()
            
            # 1. エッジ検出
            edges = F.conv2d(channel_mask, self.laplacian_kernel.to(device), padding=1)
            
            # 2. エッジの絶対値
            edge_abs = torch.abs(edges)
            
            # 3. エッジマスク生成（Sigmoid使用）
            edge_scaled = edge_abs * self.blur_strength
            edge_mask = torch.sigmoid(edge_scaled)
            
            # 4. ガウシアンブラー適用
            blurred = F.conv2d(channel_mask, self.gaussian_kernel.to(device), padding=1)
            
            # 5. ブレンディング
            smoothed = channel_mask * (1 - edge_mask) + blurred * edge_mask
            
            # 6. 最終的な二値化
            binary_output = (smoothed > self.threshold).to(dtype)
            
            smoothed_channels.append(binary_output)
        
        # Concatenate channels
        result = torch.cat(smoothed_channels, dim=1)
        
        # Restore original shape
        if len(original_shape) == 2:
            result = result.squeeze(0).squeeze(0)
        elif len(original_shape) == 3:
            result = result.squeeze(0)
            
        return result


class MultiClassEdgeSmoothing:
    """Multi-class segmentation mask edge smoothing."""
    
    def __init__(self, threshold: float = 0.5, blur_strength: float = 3.0, 
                 iterations: int = 1, device: str = 'cuda'):
        self.smoother = BinaryMaskEdgeSmoothing(threshold, blur_strength)
        self.smoother.to(device)
        self.smoother.eval()
        self.iterations = iterations
        self.device = device
    
    def smooth_predictions(self, predictions: Union[torch.Tensor, np.ndarray],
                          apply_softmax: bool = False) -> Union[torch.Tensor, np.ndarray]:
        """
        Smooth multi-class segmentation predictions.
        
        Args:
            predictions: Predictions of shape (B, C, H, W) or (C, H, W)
            apply_softmax: Whether to apply softmax first
            
        Returns:
            Smoothed predictions with same shape and type as input
        """
        # Convert to torch if needed
        is_numpy = isinstance(predictions, np.ndarray)
        if is_numpy:
            predictions = torch.from_numpy(predictions).to(self.device)
        else:
            predictions = predictions.to(self.device)
            
        # Apply softmax if needed
        if apply_softmax:
            predictions = torch.softmax(predictions, dim=1 if predictions.dim() == 4 else 0)
        
        # Get binary masks for each class
        if predictions.dim() == 3:  # (C, H, W)
            predictions = predictions.unsqueeze(0)  # -> (1, C, H, W)
            
        B, C, H, W = predictions.shape
        
        # For 3-class segmentation, process each class
        smoothed_masks = []
        
        with torch.no_grad():
            for c in range(C):
                # Extract binary mask for this class
                if C == 3:
                    # For 3-class, use argmax
                    binary_mask = (predictions.argmax(dim=1) == c).float()
                else:
                    # For probability maps, threshold
                    binary_mask = (predictions[:, c] > 0.5).float()
                
                # Apply smoothing iterations
                for _ in range(self.iterations):
                    binary_mask = self.smoother(binary_mask)
                    
                smoothed_masks.append(binary_mask)
        
        # Stack masks and convert back to class predictions
        smoothed_stack = torch.stack(smoothed_masks, dim=1)
        
        # For 3-class segmentation, ensure mutual exclusivity
        if C == 3:
            # Use argmax to get final predictions
            smoothed_predictions = smoothed_stack
        else:
            smoothed_predictions = smoothed_stack
            
        # Remove batch dim if it was added
        if predictions.shape[0] == 1 and len(predictions.shape) == 4:
            smoothed_predictions = smoothed_predictions.squeeze(0)
            
        # Convert back to numpy if needed
        if is_numpy:
            smoothed_predictions = smoothed_predictions.cpu().numpy()
            
        return smoothed_predictions


def apply_edge_smoothing_to_masks(masks: Union[torch.Tensor, np.ndarray],
                                 threshold: float = 0.5,
                                 blur_strength: float = 3.0,
                                 iterations: int = 1,
                                 device: str = 'cuda') -> Union[torch.Tensor, np.ndarray]:
    """
    Convenience function to apply edge smoothing to segmentation masks.
    
    Args:
        masks: Input masks of shape (B, C, H, W), (C, H, W), or (H, W)
        threshold: Binary threshold after smoothing
        blur_strength: Strength of edge blur effect
        iterations: Number of smoothing iterations
        device: Device to run on
        
    Returns:
        Smoothed masks with same shape and type as input
    """
    smoother = MultiClassEdgeSmoothing(threshold, blur_strength, iterations, device)
    return smoother.smooth_predictions(masks)