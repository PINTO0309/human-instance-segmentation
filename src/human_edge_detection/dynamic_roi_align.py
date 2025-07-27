"""Dynamic ROI Align implementation with variable output sizes.

This module performs ROI Align operation with dynamic output heights and widths,
allowing different ROI sizes to be processed in a single forward pass.
"""

import torch


class DynamicRoIAlign(torch.nn.Module):
    """Dynamic ROI Align layer that supports variable output sizes.

    This implementation allows ROI Align to produce outputs of different sizes
    for each ROI, which is useful for hierarchical segmentation models where
    different ROIs may require different levels of detail.

    Attributes:
        spatial_scale (float): Scale factor between input feature map and ROI coordinates.
                                Default is 1.0, meaning ROI coordinates are in feature map space.
        sampling_ratio (int): Number of sampling points in each bin. -1 means adaptive.
                                (Note: Currently not used in this implementation)
        aligned (bool): If True, use pixel-aligned grid_sample. Default is False.
    """

    def __init__(self, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
        """Initialize DynamicRoIAlign module.

        Args:
            spatial_scale (float or tuple): Scale factor to convert ROI coordinates to feature map space.
                                    Can be a single float for square images, or a tuple (scale_h, scale_w)
                                    for non-square images. For example:
                                    - Square: If ROIs are normalized [0,1] and features are 640x640, spatial_scale=640
                                    - Non-square: If ROIs are normalized [0,1] and features are 480x640,
                                      spatial_scale=(480, 640)
            sampling_ratio (int): Number of sampling points per bin. -1 for adaptive.
                                    (Currently not implemented in this version)
            aligned (bool): Whether to use pixel-aligned sampling. This affects how
                            coordinates are normalized for grid_sample.
        """
        super().__init__()
        # Handle both single value and tuple for spatial_scale
        if isinstance(spatial_scale, (list, tuple)):
            assert len(spatial_scale) == 2, "spatial_scale tuple must have 2 elements (height, width)"
            self.spatial_scale = spatial_scale
            self.spatial_scale_h = spatial_scale[0]
            self.spatial_scale_w = spatial_scale[1]
        else:
            self.spatial_scale = spatial_scale
            self.spatial_scale_h = spatial_scale
            self.spatial_scale_w = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input_feature_map, rois, output_height, output_width):
        """Perform dynamic ROI Align on input features.

        Args:
            input_feature_map (torch.Tensor): Input feature tensor of shape (N, C, H, W)
                                                where N is batch size, C is channels,
                                                H and W are spatial dimensions.
            rois (torch.Tensor): ROI tensor of shape (K, 5) where K is number of ROIs.
                                    Each ROI is [batch_idx, x1, y1, x2, y2].
                                    Coordinates should be in the same space as specified
                                    by spatial_scale.
            output_height (int or torch.Tensor): Desired output height for aligned features.
                                                Can be different for each forward pass.
            output_width (int or torch.Tensor): Desired output width for aligned features.
                                                Can be different for each forward pass.

        Returns:
            torch.Tensor: Aligned features of shape (K, C, output_height, output_width)
                            where K is the number of ROIs and C matches input channels.
        """
        num_rois = rois.shape[0]

        # Extract batch indices and scale ROI coordinates to feature map space
        batch_indices = rois[:, 0].long()
        # Scale x and y coordinates separately for non-square images
        # rois format: [batch_idx, x1, y1, x2, y2] where x,y are normalized [0,1]
        x1 = rois[:, 1] * self.spatial_scale_w
        y1 = rois[:, 2] * self.spatial_scale_h
        x2 = rois[:, 3] * self.spatial_scale_w
        y2 = rois[:, 4] * self.spatial_scale_h

        # Stack back into boxes tensor
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # ROI dimensions are already computed
        roi_width = x2 - x1
        roi_height = y2 - y1

        # Generate a grid for each ROI
        # This part is vectorized to avoid Python loops over ROIs

        # Create normalized coordinates [0, 1] for output grid points
        # These represent relative positions within each output bin
        # Ensure output_width and output_height are integers
        if isinstance(output_width, (list, tuple)):
            output_width = output_width[0] if len(output_width) > 0 else output_width
        if isinstance(output_height, (list, tuple)):
            output_height = output_height[0] if len(output_height) > 0 else output_height
        if torch.is_tensor(output_width):
            output_width = int(output_width.item())
        if torch.is_tensor(output_height):
            output_height = int(output_height.item())

        x_coords_normalized = torch.linspace(0, 1, output_width, device=rois.device)
        y_coords_normalized = torch.linspace(0, 1, output_height, device=rois.device)

        # Create 2D grid of normalized coordinates
        # grid_y, grid_x shape: (output_height, output_width)
        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing='ij')

        # Expand grid to process all ROIs in parallel
        # Shape: (num_rois, output_height, output_width)
        grid_x_expanded = grid_x.unsqueeze(0).expand(num_rois, -1, -1)
        grid_y_expanded = grid_y.unsqueeze(0).expand(num_rois, -1, -1)

        # Calculate coordinates in feature map space for each bin center
        # The +0.5 offset centers the sampling point within each bin
        # fx = x1 + (grid_x + 0.5) * bin_width
        # fy = y1 + (grid_y + 0.5) * bin_height

        # Note: bin dimensions are implicitly handled by the grid normalization
        # Each output pixel samples from the corresponding portion of the ROI

        # Calculate actual feature map coordinates for each sampling point
        # The grid already represents positions from 0 to 1 within the ROI
        # Shape: (num_rois, output_height, output_width)
        fx = x1.unsqueeze(1).unsqueeze(2) + grid_x_expanded * roi_width.unsqueeze(1).unsqueeze(2)
        fy = y1.unsqueeze(1).unsqueeze(2) + grid_y_expanded * roi_height.unsqueeze(1).unsqueeze(2)

        # Normalize coordinates to [-1, 1] range required by grid_sample
        # grid_sample expects -1 for top-left corner and 1 for bottom-right
        H_feat, W_feat = input_feature_map.shape[2], input_feature_map.shape[3]
        if self.aligned:
            # When align_corners=True, corners map to corners
            normalized_fx = (fx / (W_feat - 1)) * 2 - 1
            normalized_fy = (fy / (H_feat - 1)) * 2 - 1
        else:
            # When align_corners=False, pixel centers are used
            normalized_fx = (fx / W_feat) * 2 - 1
            normalized_fy = (fy / H_feat) * 2 - 1

        # Stack x and y coordinates to form sampling grid
        # Shape: (num_rois, output_height, output_width, 2)
        # Last dimension contains [x, y] coordinates for each sampling point
        grids_tensor = torch.stack([normalized_fx, normalized_fy], dim=-1)

        # Select the appropriate feature map for each ROI based on its batch index
        # This handles ROIs from different images in the batch
        # Shape: (num_rois, C, H, W)
        selected_feature_maps = torch.index_select(input_feature_map, 0, batch_indices)

        # Apply bilinear interpolation to sample features at grid points
        # grid_sample performs differentiable bilinear interpolation
        # Input: (num_rois, C, H_feat, W_feat)
        # Grid: (num_rois, output_height, output_width, 2)
        # Output: (num_rois, C, output_height, output_width)
        pooled_features = torch.nn.functional.grid_sample(
            selected_feature_maps,
            grids_tensor,
            mode='bilinear',  # Use bilinear interpolation for smooth gradients
            padding_mode='zeros',  # Pad with zeros outside feature map boundaries
            align_corners=self.aligned  # Controls pixel alignment behavior
        )

        return pooled_features


if __name__ == '__main__':
    """Test DynamicRoIAlign with example data and ONNX export."""

    # Create dummy input data for testing
    # Feature map: batch_size=2, channels=256, height=56, width=56
    input_feature_map = torch.randn(2, 256, 56, 56)

    # Define ROIs: [batch_idx, x1, y1, x2, y2]
    # Note: Coordinates are in feature map space (0-56 range)
    rois = torch.tensor([
        [0, 10, 10, 50, 50],  # ROI from first image: 40x40 region
        [0, 20, 20, 60, 60],  # Another ROI from first image (overlaps edge)
        [1, 5, 5, 45, 45]     # ROI from second image: 40x40 region
    ], dtype=torch.float32)

    # Define desired output size for aligned features
    output_height = 7
    output_width = 7

    # Create and test the module
    roi_align_module = DynamicRoIAlign()
    output = roi_align_module(input_feature_map, rois, output_height, output_width)

    print("Output shape:", output.shape)
    print(f"Expected: (num_rois={rois.shape[0]}, channels=256, height={output_height}, width={output_width})")

    # ONNX Export Test
    print("\nTesting ONNX export...")

    # Define dynamic axes for flexible input/output shapes
    # This allows the ONNX model to accept variable batch sizes, ROI counts, and output sizes
    dynamic_axes = {
        "input_feature_map": {
            0: "batch_size",    # Variable number of images
            1: "channels",      # Variable number of feature channels
            2: "H",             # Variable feature map height
            3: "W"              # Variable feature map width
        },
        "rois": {
            0: "num_rois"       # Variable number of ROIs
        },
        "output_height": {},    # Scalar, but can vary between calls
        "output_width": {},     # Scalar, but can vary between calls
        "output": {
            0: "num_rois",      # Matches number of input ROIs
            1: "channels",      # Matches input feature channels
            2: "output_H",      # Matches specified output_height
            3: "output_W"       # Matches specified output_width
        }
    }

    # Export the model to ONNX format
    torch.onnx.export(
        roi_align_module,
        # Convert height/width to tensors for ONNX compatibility
        (input_feature_map, rois, torch.tensor(output_height), torch.tensor(output_width)),
        "dynamic_roi_align.onnx",
        input_names=["input_feature_map", "rois", "output_height", "output_width"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=16  # Opset 16+ required for grid_sample with dynamic shapes
    )
    print("ONNX model exported successfully to: dynamic_roi_align.onnx")