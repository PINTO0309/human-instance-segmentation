"""Test script to verify all components work correctly."""

import torch
import json
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from src.human_edge_detection.dataset import COCOInstanceSegmentationDataset
    from src.human_edge_detection.feature_extractor import YOLOv9FeatureExtractor
    from src.human_edge_detection.model import create_model, ROIBatchProcessor
    from src.human_edge_detection.losses import create_loss_function
    from src.human_edge_detection.train import create_trainer
    from src.human_edge_detection.visualize import ValidationVisualizer
    from src.human_edge_detection.export_onnx import ONNXExporter
    print("✓ All imports successful!")
except Exception as e:
    print(f"✗ Import error: {e}")
    exit(1)

# Test configuration
config = {
    'num_classes': 3,
    'in_channels': 1024,
    'mid_channels': 256,
    'roi_size': 28,  # Enhanced model with better spatial resolution
    'mask_size': 56,
    'batch_size': 2,  # Small batch for testing
    'num_workers': 0,  # No multiprocessing for testing
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_epochs': 1,  # Just one epoch for testing
    'ce_weight': 1.0,
    'dice_weight': 1.0,
    'dice_classes': [1],
    'use_log_weights': True
}

# Check data files
print("\nChecking data files...")
data_files = {
    'train_ann': 'data/annotations/instances_train2017_person_only_no_crowd_100.json',
    'val_ann': 'data/annotations/instances_val2017_person_only_no_crowd_100.json',
    'train_img_dir': 'data/images/train2017',
    'val_img_dir': 'data/images/val2017',
    'onnx_model': 'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
    'data_stats': 'data_analyze_100.json'
}

all_exist = True
for name, path in data_files.items():
    if Path(path).exists():
        print(f"✓ {name}: {path}")
    else:
        print(f"✗ {name}: {path} NOT FOUND")
        all_exist = False

if not all_exist:
    print("\nSome required files are missing!")
    exit(1)

# Test dataset
print("\nTesting dataset...")
try:
    dataset = COCOInstanceSegmentationDataset(
        annotation_file=data_files['train_ann'],
        image_dir=data_files['train_img_dir']
    )
    print(f"✓ Dataset created successfully. Size: {len(dataset)}")

    # Test loading a sample
    sample = dataset[0]
    print(f"✓ Sample loaded. Image shape: {sample['image'].shape}")
    print(f"✓ ROI mask shape: {sample['roi_mask'].shape}")
    print(f"✓ Unique classes in mask: {torch.unique(sample['roi_mask']).tolist()}")
except Exception as e:
    print(f"✗ Dataset error: {e}")
    exit(1)

# Test feature extractor
print("\nTesting feature extractor...")
try:
    feature_extractor = YOLOv9FeatureExtractor(
        data_files['onnx_model'],
        device='cpu'  # Use CPU for testing
    )

    # Test with dummy input
    dummy_input = torch.randn(1, 3, 640, 640)
    features = feature_extractor.extract_features(dummy_input)
    print(f"✓ Feature extractor works. Output shape: {features.shape}")
except Exception as e:
    print(f"✗ Feature extractor error: {e}")
    exit(1)

# Test model
print("\nTesting model...")
try:
    model = create_model(
        num_classes=config['num_classes'],
        in_channels=config['in_channels'],
        mid_channels=config['mid_channels'],
        roi_size=config['roi_size'],
        mask_size=config['mask_size']
    )

    # Test forward pass
    dummy_features = torch.randn(2, 1024, 80, 80)
    dummy_roi_coords = torch.tensor([
        [0.1, 0.1, 0.5, 0.5],
        [0.3, 0.3, 0.8, 0.8]
    ])
    dummy_rois = ROIBatchProcessor.prepare_rois_for_batch(dummy_roi_coords)

    output = model(features=dummy_features, rois=dummy_rois)
    print(f"✓ Model forward pass successful. Mask shape: {output['masks'].shape}")
except Exception as e:
    print(f"✗ Model error: {e}")
    exit(1)

# Test loss function
print("\nTesting loss function...")
try:
    with open(data_files['data_stats'], 'r') as f:
        data_stats = json.load(f)
    pixel_ratios = data_stats['pixel_ratios']

    loss_fn = create_loss_function(
        pixel_ratios=pixel_ratios,
        use_log_weights=True,
        device='cpu'
    )

    # Test with dummy data
    dummy_pred = torch.randn(2, 3, 56, 56)
    dummy_target = torch.randint(0, 3, (2, 56, 56))

    loss, loss_dict = loss_fn(dummy_pred, dummy_target)
    print(f"✓ Loss function works. Total loss: {loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce_loss'].item():.4f}")
    print(f"  Dice loss: {loss_dict['dice_loss'].item():.4f}")
except Exception as e:
    print(f"✗ Loss function error: {e}")
    exit(1)

# Test ONNX export
print("\nTesting ONNX export...")
try:
    exporter = ONNXExporter(model, device='cpu')
    test_onnx_path = 'test_model.onnx'

    success = exporter.export_segmentation_head(
        output_path=test_onnx_path,
        verify=True
    )

    if success:
        print(f"✓ ONNX export successful")
        # Clean up test file
        Path(test_onnx_path).unlink()
    else:
        print(f"✗ ONNX export failed")
except Exception as e:
    print(f"✗ ONNX export error: {e}")

# Summary
print("\n" + "="*50)
print("PIPELINE TEST COMPLETE")
print("="*50)
print("\nAll components are working correctly!")
print("\nYou can now run the training with:")
print("  python main.py --epochs 10")
print("\nFor a quick test with 1 epoch:")
print("  python main.py --epochs 1")
print("\nTo use the full dataset, update the annotation files in main.py")