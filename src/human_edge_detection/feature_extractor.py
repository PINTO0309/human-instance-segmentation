"""Feature extractor for YOLO ONNX model intermediate layers."""

import numpy as np
import onnxruntime as ort
import torch
from typing import List, Optional
from pathlib import Path

# TensorRT setup function
def setup_tensorrt_providers(model_path: Path) -> List:
    """Setup TensorRT providers with optimization parameters.

    Args:
        model_path: Path to the ONNX model file

    Returns:
        List of providers with TensorRT optimization settings
    """
    # Use same directory as ONNX model for TensorRT cache
    model_dir_path = str(model_path.parent)

    # TensorRT provider with optimization settings
    trt_provider = (
        'TensorrtExecutionProvider',
        {
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': model_dir_path,
            'trt_fp16_enable': True,
        }
    )

    return [
        trt_provider,
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]


class YOLOFeatureExtractor:
    """Extract intermediate features from YOLO ONNX model.

    This class loads a YOLO ONNX model and extracts features from
    the intermediate layer specified (default: segmentation_model_34_Concat_output_0).
    """

    def __init__(
        self,
        onnx_path: str,
        target_layer: str = "segmentation_model_34_Concat_output_0",
        device: str = "cpu",
        providers: Optional[List[str]] = None
    ):
        """Initialize feature extractor.

        Args:
            onnx_path: Path to YOLO ONNX model
            target_layer: Name of the intermediate layer to extract features from
            device: Device to run inference on ('cpu' or 'cuda')
            providers: ONNX Runtime providers (auto-selected if None)
        """
        self.onnx_path = Path(onnx_path)
        self.target_layer = target_layer
        self.device = device

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Set providers with TensorRT optimization
        if providers is None:
            if device == "cuda" and torch.cuda.is_available():
                # Try TensorRT first, then CUDA, then CPU
                providers = setup_tensorrt_providers(self.onnx_path)
            else:
                providers = ['CPUExecutionProvider']

        # Create session with intermediate outputs
        self.session = self._create_session_with_intermediate_outputs(providers)

        # Get input info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

        # Verify target layer exists
        output_names = [out.name for out in self.session.get_outputs()]
        if self.target_layer not in output_names:
            raise ValueError(
                f"Target layer '{self.target_layer}' not found in model outputs. "
                f"Available outputs: {output_names}"
            )

        # Get feature shape
        for out in self.session.get_outputs():
            if out.name == self.target_layer:
                self.feature_shape = out.shape
                break

        print(f"Initialized YOLOFeatureExtractor:")
        print(f"  Model: {onnx_path}")
        print(f"  Target layer: {self.target_layer}")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Feature shape: {self.feature_shape}")

    def _create_session_with_intermediate_outputs(self, providers: List[str]) -> ort.InferenceSession:
        """Create ONNX Runtime session that outputs intermediate layers."""
        # For models with feature extraction (featext), intermediate outputs should already be available
        # Create session options
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # Create session
        session = ort.InferenceSession(
            str(self.onnx_path),
            session_options,
            providers=providers
        )

        # Check if the model already has the intermediate output we need
        available_outputs = [out.name for out in session.get_outputs()]
        print(f"Available model outputs: {available_outputs}")

        return session

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Input images tensor of shape (B, 3, H, W), normalized to [0, 1]

        Returns:
            Features tensor of shape (B, 1024, 80, 80)
        """
        # Ensure correct input shape
        if images.dim() == 3:
            images = images.unsqueeze(0)

        # Convert to numpy and ensure correct format
        if images.is_cuda:
            images_np = images.cpu().numpy()
        else:
            images_np = images.numpy()

        # Run inference
        outputs = self.session.run(
            [self.target_layer],
            {self.input_name: images_np}
        )

        # Get features
        features = outputs[0]

        # Convert back to torch tensor
        features_tensor = torch.from_numpy(features)

        # Move to same device as input
        if images.is_cuda:
            features_tensor = features_tensor.cuda()

        return features_tensor

    def extract_features_batch(self, image_list: List[np.ndarray]) -> torch.Tensor:
        """Extract features from a list of images.

        Args:
            image_list: List of images as numpy arrays (H, W, 3) in range [0, 255]

        Returns:
            Features tensor of shape (B, 1024, 80, 80)
        """
        # Preprocess images
        batch = []
        for img in image_list:
            # Normalize and transpose
            img_norm = img.astype(np.float32) / 255.0
            img_transposed = np.transpose(img_norm, (2, 0, 1))
            batch.append(img_transposed)

        # Stack into batch
        batch_np = np.stack(batch, axis=0)

        # Convert to tensor and extract features
        batch_tensor = torch.from_numpy(batch_np)
        return self.extract_features(batch_tensor)


class YOLOv9FeatureExtractor(YOLOFeatureExtractor):
    """Specialized feature extractor for YOLOv9 models.

    This implementation handles the specific requirements for YOLOv9
    models with feature extraction capabilities.
    """

    def __init__(
        self,
        onnx_path: str,
        model_variant: str = "e",  # "e" or "n"
        **kwargs
    ):
        """Initialize YOLOv9 feature extractor.

        Args:
            onnx_path: Path to YOLOv9 ONNX model
            model_variant: Model variant ("e" for efficient, "n" for nano)
            **kwargs: Additional arguments for parent class
        """
        self.model_variant = model_variant

        # For YOLOv9, the intermediate feature layer is typically at segmentation_model_34_Concat_output_0
        if 'target_layer' not in kwargs:
            kwargs['target_layer'] = "segmentation_model_34_Concat_output_0"

        super().__init__(onnx_path, **kwargs)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLOv9.

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            Preprocessed image (3, 640, 640) normalized to [0, 1]
        """
        # Resize to 640x640
        import cv2
        resized = cv2.resize(image, (640, 640))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        return transposed


def test_feature_extractor():
    """Test the feature extractor with a dummy input."""
    # Path to ONNX model
    onnx_path = "ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx"

    if not Path(onnx_path).exists():
        print(f"ONNX model not found at {onnx_path}")
        return

    # Create feature extractor
    extractor = YOLOv9FeatureExtractor(onnx_path)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640)

    # Extract features
    features = extractor.extract_features(dummy_input)

    print(f"\nFeature extraction test:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")


if __name__ == "__main__":
    test_feature_extractor()