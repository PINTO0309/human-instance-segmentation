# ROI-based Instance Segmentation for Human Detection

This repository implements a lightweight ROI-based instance segmentation model for human detection using YOLOv9 features and a custom segmentation decoder.

## Overview

The model uses a 3-class segmentation approach:
- **Class 0**: Background
- **Class 1**: Target mask (primary instance in ROI)
- **Class 2**: Non-target mask (other instances in ROI)

This formulation helps the model better distinguish between multiple person instances in crowded scenes.

## Quick Start

1. **Environment Setup**
   ```bash
   uv sync
   ```

2. **Test Pipeline**
   ```bash
   uv run python test_pipeline.py
   ```

3. **Train Model (minimal dataset)**
   ```bash
   uv run python main.py --epochs 10
   ```

## Architecture

- **Feature Extraction**: YOLOv9 intermediate features (1024×80×80)
- **ROI Processing**: ROIAlign for extracting fixed-size features
- **Segmentation Head**: Lightweight decoder producing 56×56 masks
- **Loss Function**: Weighted CrossEntropy + Dice loss

## Project Structure

```
├── src/human_edge_detection/
│   ├── dataset.py          # COCO dataset loader
│   ├── feature_extractor.py # YOLO feature extraction
│   ├── model.py            # Segmentation model
│   ├── losses.py           # Loss functions
│   ├── train.py            # Training pipeline
│   ├── visualize.py        # Validation visualization
│   └── export_onnx.py      # ONNX export
├── data/
│   ├── annotations/        # COCO format annotations
│   └── images/            # Training/validation images
├── ext_extractor/         # YOLOv9 ONNX models
├── main.py               # Main training script
└── test_pipeline.py      # Component verification

```

## Training

See `CLAUDE.md` for detailed training instructions and command examples.

## License

MIT License - See LICENSE file for details.