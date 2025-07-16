# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a human edge detection project that appears to be in its initial stages. The repository contains:
- COCO dataset images and annotations filtered for person instances (no crowd annotations)
- YOLOv9 ONNX models for whole body detection with feature extraction capabilities

## Repository Structure

### Data Organization
- `data/annotations/`: Contains COCO format JSON files with person-only annotations
  - Full datasets: `instances_train2017_person_only_no_crowd.json`, `instances_val2017_person_only_no_crowd.json`
  - Smaller subsets: 100 and 500 image versions for development/testing
- `data/images/`: Contains train2017 and val2017 image directories from COCO dataset
- `ext_extractor/`: Contains YOLOv9 ONNX models
  - `yolov9_e_wholebody25_0100_1x3x640x640_featext.onnx`: Efficient model variant
  - `yolov9_n_wholebody25_0100_1x3x640x640_featext.onnx`: Nano model variant

## Development Status

Currently, this repository contains only data and pre-trained models. No implementation code exists yet. When implementing:

1. Consider the ONNX models are for whole body detection with 25 keypoints
2. The annotation files follow COCO format but are filtered to contain only person instances without crowd annotations
3. The models expect 640x640 input images with 3 channels

## Future Implementation Considerations

When implementing the edge detection system:
- The YOLOv9 models appear to be feature extraction variants (`featext` suffix)
- Models are designed for whole body detection with 25 keypoints
- Consider whether edge detection will be performed on the detected person regions or use the extracted features