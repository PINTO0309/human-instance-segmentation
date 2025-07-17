import torch

class DynamicRoIAlign(torch.nn.Module):
    def __init__(self, spatial_scale=1.0, sampling_ratio=-1, aligned=False):
        super().__init__()
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input_feature_map, rois, output_height, output_width):
        num_rois = rois.shape[0]

        batch_indices = rois[:, 0].long()
        boxes = rois[:, 1:] * self.spatial_scale

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        roi_width = x2 - x1
        roi_height = y2 - y1

        # Generate a grid for each ROI
        # This part is vectorized to avoid Python loops over ROIs

        # Create linspace for x and y coordinates for each output bin
        # Shape: (output_width,) and (output_height,)
        x_coords_normalized = torch.linspace(0, 1, output_width, device=rois.device)
        y_coords_normalized = torch.linspace(0, 1, output_height, device=rois.device)

        # Create meshgrid for a single ROI, then expand for all ROIs
        # grid_y, grid_x shape: (output_height, output_width)
        grid_y, grid_x = torch.meshgrid(y_coords_normalized, x_coords_normalized, indexing='ij')

        # Expand grid_x and grid_y to match num_rois
        # Shape: (num_rois, output_height, output_width)
        grid_x_expanded = grid_x.unsqueeze(0).expand(num_rois, -1, -1)
        grid_y_expanded = grid_y.unsqueeze(0).expand(num_rois, -1, -1)

        # Calculate coordinates in feature map space for each bin center
        # fx = x1 + (grid_x + 0.5) * bin_width
        # fy = y1 + (grid_y + 0.5) * bin_height

        # Calculate bin width and height for each ROI
        # Shape: (num_rois, 1, 1)
        bin_width_per_roi = (roi_width / output_width).unsqueeze(1).unsqueeze(2)
        bin_height_per_roi = (roi_height / output_height).unsqueeze(1).unsqueeze(2)

        # Calculate feature map coordinates for each sampling point
        # Shape: (num_rois, output_height, output_width)
        fx = x1.unsqueeze(1).unsqueeze(2) + (grid_x_expanded + 0.5) * bin_width_per_roi
        fy = y1.unsqueeze(1).unsqueeze(2) + (grid_y_expanded + 0.5) * bin_height_per_roi

        # Normalize to [-1, 1] for grid_sample
        H_feat, W_feat = input_feature_map.shape[2], input_feature_map.shape[3]
        normalized_fx = (fx / (W_feat - 1)) * 2 - 1
        normalized_fy = (fy / (H_feat - 1)) * 2 - 1

        # Stack to form grid (num_rois, output_height, output_width, 2)
        grids_tensor = torch.stack([normalized_fx, normalized_fy], dim=-1)

        # Prepare input_feature_map for grid_sample
        # We need to gather the correct feature map for each ROI based on batch_indices
        # This can be done using torch.index_select or by expanding the input_feature_map

        # Using torch.index_select to get the relevant feature maps for each ROI
        # Shape: (num_rois, C, H, W)
        selected_feature_maps = torch.index_select(input_feature_map, 0, batch_indices)

        # Apply grid_sample
        # grid_sample expects input (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
        # Output will be (N, C, H_out, W_out)
        pooled_features = torch.nn.functional.grid_sample(
            selected_feature_maps,
            grids_tensor,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=self.aligned
        )

        return pooled_features


if __name__ == '__main__':
    # Dummy input data
    input_feature_map = torch.randn(2, 256, 56, 56) # N=2, C=256, H=56, W=56
    rois = torch.tensor([
        [0, 10, 10, 50, 50], # batch_idx, x1, y1, x2, y2
        [0, 20, 20, 60, 60],
        [1, 5, 5, 45, 45] # ROI for the second image in the batch
    ], dtype=torch.float32)

    output_height = 7
    output_width = 7

    roi_align_module = DynamicRoIAlign()
    output = roi_align_module(input_feature_map, rois, output_height, output_width)

    print("Output shape:", output.shape)

    # ONNX Export Test
    # Define dynamic axes for ONNX export
    dynamic_axes = {
        "input_feature_map": {0: "batch_size", 1: "channels", 2: "H", 3: "W"},
        "rois": {0: "num_rois"},
        "output_height": {},
        "output_width": {},
        "output": {0: "num_rois", 1: "channels", 2: "output_H", 3: "output_W"} # Output channels are also dynamic
    }

    # Export the model to ONNX
    torch.onnx.export(
        roi_align_module,
        (input_feature_map, rois, torch.tensor(output_height), torch.tensor(output_width)),
        "dynamic_roi_align.onnx",
        input_names=["input_feature_map", "rois", "output_height", "output_width"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=16 # Opset version 16 or higher is recommended for grid_sample
    )
    print("ONNX model exported to dynamic_roi_align.onnx")