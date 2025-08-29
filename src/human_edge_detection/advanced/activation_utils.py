"""Activation function utilities for hierarchical segmentation models.

This module provides utilities for selecting and using different activation functions
in the hierarchical segmentation models, including support for Swish activation.
"""

import torch
import torch.nn as nn
from typing import Optional, Union


class Swish(nn.Module):
    """Swish activation function: f(x) = x * sigmoid(x).
    
    This activation function has shown better performance than ReLU in some cases,
    providing smoother gradients and better optimization properties.
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize Swish activation.
        
        Args:
            beta: Scaling factor for sigmoid input. Default is 1.0.
                  Higher values make it more like ReLU, lower values make it smoother.
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        return x * torch.sigmoid(self.beta * x)


class SwishInplace(nn.Module):
    """Inplace version of Swish activation for memory efficiency.
    
    Note: This is not truly inplace due to PyTorch limitations, but minimizes
    memory usage where possible.
    """
    
    def __init__(self, beta: float = 1.0):
        """Initialize SwishInplace activation.
        
        Args:
            beta: Scaling factor for sigmoid input. Default is 1.0.
        """
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation with minimal memory usage.
        
        Args:
            x: Input tensor
            
        Returns:
            Activated tensor
        """
        # Compute sigmoid separately to allow garbage collection
        sigmoid_x = torch.sigmoid(self.beta * x)
        return x.mul_(sigmoid_x)


def get_activation(
    activation: str = "relu",
    inplace: bool = True,
    **kwargs
) -> nn.Module:
    """Get activation function module by name.
    
    Args:
        activation: Name of activation function. Options: "relu", "swish", "gelu", "silu"
        inplace: Whether to use inplace version for memory efficiency (where available)
        **kwargs: Additional arguments for specific activations (e.g., beta for Swish)
        
    Returns:
        Activation module
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activation = activation.lower()
    
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    elif activation == "swish":
        beta = kwargs.get("beta", 1.0)
        return SwishInplace(beta=beta) if inplace else Swish(beta=beta)
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        # SiLU is equivalent to Swish with beta=1.0
        # PyTorch has native SiLU which is more efficient
        return nn.SiLU(inplace=inplace)
    else:
        raise ValueError(f"Unknown activation function: {activation}")


def create_norm_activation(
    in_channels: int,
    activation: str = "relu",
    norm_type: str = "layernorm2d",
    **activation_kwargs
) -> nn.Sequential:
    """Create a normalization + activation sequence.
    
    Args:
        in_channels: Number of input channels for normalization
        activation: Name of activation function
        norm_type: Type of normalization ("layernorm2d", "batchnorm2d", "instancenorm2d")
        **activation_kwargs: Additional arguments for activation function
        
    Returns:
        Sequential module with normalization and activation
    """
    from ..model import LayerNorm2d
    
    # Select normalization
    if norm_type == "layernorm2d":
        norm = LayerNorm2d(in_channels)
    elif norm_type == "batchnorm2d":
        norm = nn.BatchNorm2d(in_channels)
    elif norm_type == "instancenorm2d":
        norm = nn.InstanceNorm2d(in_channels)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")
    
    # Get activation
    act = get_activation(activation, **activation_kwargs)
    
    return nn.Sequential(norm, act)


class ActivationConfig:
    """Configuration for activation functions in models.
    
    This class holds the configuration for activation functions to be used
    throughout a model, ensuring consistency.
    """
    
    def __init__(
        self,
        activation: str = "relu",
        norm_type: str = "layernorm2d",
        inplace: bool = True,
        **activation_kwargs
    ):
        """Initialize activation configuration.
        
        Args:
            activation: Name of activation function
            norm_type: Type of normalization to use
            inplace: Whether to use inplace operations
            **activation_kwargs: Additional activation-specific arguments
        """
        self.activation = activation
        self.norm_type = norm_type
        self.inplace = inplace
        self.activation_kwargs = activation_kwargs
    
    def get_activation(self) -> nn.Module:
        """Get configured activation module."""
        return get_activation(
            self.activation,
            inplace=self.inplace,
            **self.activation_kwargs
        )
    
    def get_norm_activation(self, in_channels: int) -> nn.Sequential:
        """Get configured norm + activation sequence."""
        return create_norm_activation(
            in_channels,
            activation=self.activation,
            norm_type=self.norm_type,
            inplace=self.inplace,
            **self.activation_kwargs
        )


# ONNX-compatible versions for export
def swish_onnx(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """ONNX-compatible Swish activation function.
    
    Args:
        x: Input tensor
        beta: Scaling factor for sigmoid
        
    Returns:
        Activated tensor
    """
    return x * torch.sigmoid(beta * x)


class SwishONNX(nn.Module):
    """ONNX-exportable Swish activation module."""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish_onnx(x, self.beta)