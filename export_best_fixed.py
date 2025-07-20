#!/usr/bin/env python3
"""Export best hierarchical model with fixed export function."""

from src.human_edge_detection.export_onnx_advanced import export_checkpoint_to_onnx_advanced
import json

# Load config
with open('experiments/hierarchical_segmentation/configs/config.json', 'r') as f:
    config = json.load(f)

# Export
success = export_checkpoint_to_onnx_advanced(
    checkpoint_path='experiments/hierarchical_segmentation/checkpoints/best_model.pth',
    output_path='experiments/hierarchical_segmentation/checkpoints/best_model_fixed.onnx',
    model_type='hierarchical',
    config=config,
    device='cuda',
    verify=True
)

if success:
    print("Export successful!")
    print("Check metadata at: experiments/hierarchical_segmentation/checkpoints/best_model_fixed.json")
else:
    print("Export failed!")