"""Multi-scale feature extractor for YOLOv9."""

import torch
import numpy as np
import onnxruntime as ort
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class MultiScaleYOLOFeatureExtractor:
    """Extract multi-scale features from YOLOv9 model.
    
    Available outputs from YOLOv9:
    - segmentation_model_3_Concat_output_0: [batch, 256, 160, 160] - High res, shallow
    - segmentation_model_19_Concat_output_0: [batch, 256, 160, 160] - High res, mid
    - segmentation_model_5_Concat_output_0: [batch, 512, 80, 80] - Mid res, shallow
    - segmentation_model_22_Concat_output_0: [batch, 512, 80, 80] - Mid res, mid
    - segmentation_model_34_Concat_output_0: [batch, 1024, 80, 80] - Low res, deep
    """
    
    # Feature map specifications
    FEATURE_SPECS = {
        'layer_3': {
            'name': 'segmentation_model_3_Concat_output_0',
            'channels': 256,
            'resolution': 160,
            'stride': 4,
            'level': 'high'
        },
        'layer_19': {
            'name': 'segmentation_model_19_Concat_output_0',
            'channels': 256,
            'resolution': 160,
            'stride': 4,
            'level': 'high'
        },
        'layer_5': {
            'name': 'segmentation_model_5_Concat_output_0',
            'channels': 512,
            'resolution': 80,
            'stride': 8,
            'level': 'mid'
        },
        'layer_22': {
            'name': 'segmentation_model_22_Concat_output_0',
            'channels': 512,
            'resolution': 80,
            'stride': 8,
            'level': 'mid'
        },
        'layer_34': {
            'name': 'segmentation_model_34_Concat_output_0',
            'channels': 1024,
            'resolution': 80,
            'stride': 8,
            'level': 'low'
        }
    }
    
    def __init__(
        self,
        model_path: str,
        target_layers: List[str] = ['layer_3', 'layer_22', 'layer_34'],
        execution_provider: str = 'cuda'
    ):
        """Initialize multi-scale feature extractor.
        
        Args:
            model_path: Path to ONNX model
            target_layers: List of layer names to extract
            execution_provider: ONNX execution provider
        """
        self.model_path = Path(model_path)
        self.target_layers = target_layers
        self.execution_provider = execution_provider
        
        # Validate target layers
        for layer in target_layers:
            if layer not in self.FEATURE_SPECS:
                raise ValueError(f"Unknown layer: {layer}. Available: {list(self.FEATURE_SPECS.keys())}")
        
        # Initialize ONNX session
        self._init_session()
        
    def _init_session(self):
        """Initialize ONNX Runtime session."""
        providers = {
            'cpu': ['CPUExecutionProvider'],
            'cuda': ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            'tensorrt': ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        }
        
        # Get providers list
        provider_list = providers.get(self.execution_provider, ['CPUExecutionProvider'])
        
        # Special handling for TensorRT
        if self.execution_provider == 'tensorrt':
            # TensorRT provider options
            tensorrt_options = {
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(self.model_path.parent),
                'trt_fp16_enable': True,
            }
            provider_list = [
                ('TensorrtExecutionProvider', tensorrt_options),
                'CUDAExecutionProvider',
                'CPUExecutionProvider'
            ]
        
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=provider_list
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Map output names
        self.output_names = []
        self.output_map = {}
        
        for output in self.session.get_outputs():
            self.output_names.append(output.name)
            # Find matching layer spec
            for layer_id, spec in self.FEATURE_SPECS.items():
                if output.name == spec['name'] and layer_id in self.target_layers:
                    self.output_map[layer_id] = output.name
                    
        # Get actual provider being used
        actual_providers = self.session.get_providers()
        
        print(f"Initialized MultiScaleFeatureExtractor:")
        print(f"  Model: {self.model_path}")
        print(f"  Target layers: {self.target_layers}")
        print(f"  Output map: {self.output_map}")
        print(f"  Requested provider: {self.execution_provider}")
        print(f"  Actual providers: {actual_providers}")
        
    def extract_features(self, image_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from image batch.
        
        Args:
            image_batch: Input images (B, 3, H, W) normalized to [0, 1]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Convert to numpy
        if isinstance(image_batch, torch.Tensor):
            image_np = image_batch.cpu().numpy()
        else:
            image_np = image_batch
            
        # Run inference
        outputs = self.session.run(
            list(self.output_map.values()),
            {self.input_name: image_np}
        )
        
        # Convert outputs to tensors
        features = {}
        for i, (layer_id, output_name) in enumerate(self.output_map.items()):
            features[layer_id] = torch.from_numpy(outputs[i]).to(image_batch.device)
            
        return features
    
    def get_feature_info(self, layer_id: str) -> dict:
        """Get information about a specific feature layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            Dictionary with layer specifications
        """
        return self.FEATURE_SPECS.get(layer_id, {})
    
    def get_feature_pyramid_levels(self) -> Dict[str, List[str]]:
        """Group layers by resolution level.
        
        Returns:
            Dictionary mapping resolution levels to layer IDs
        """
        levels = {'high': [], 'mid': [], 'low': []}
        
        for layer_id in self.target_layers:
            spec = self.FEATURE_SPECS[layer_id]
            levels[spec['level']].append(layer_id)
            
        return levels
    
    def compute_feature_statistics(self, features: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        """Compute statistics for extracted features.
        
        Args:
            features: Dictionary of feature tensors
            
        Returns:
            Statistics for each feature layer
        """
        stats = {}
        
        for layer_id, feat in features.items():
            stats[layer_id] = {
                'shape': list(feat.shape),
                'mean': feat.mean().item(),
                'std': feat.std().item(),
                'min': feat.min().item(),
                'max': feat.max().item(),
                'sparsity': (feat == 0).float().mean().item()
            }
            
        return stats


class FeaturePyramidFusion(torch.nn.Module):
    """Fuse multi-scale features into a unified representation."""
    
    def __init__(
        self,
        feature_specs: Dict[str, dict],
        output_channels: int = 256,
        fusion_method: str = 'fpn'
    ):
        """Initialize feature pyramid fusion.
        
        Args:
            feature_specs: Dictionary of feature specifications
            output_channels: Number of output channels after fusion
            fusion_method: Method for fusing features ('fpn', 'concat', 'sum')
        """
        super().__init__()
        
        self.feature_specs = feature_specs
        self.output_channels = output_channels
        self.fusion_method = fusion_method
        
        # Create lateral connections (1x1 convs to unify channels)
        self.lateral_convs = torch.nn.ModuleDict()
        
        for layer_id, spec in feature_specs.items():
            self.lateral_convs[layer_id] = torch.nn.Conv2d(
                spec['channels'],
                output_channels,
                kernel_size=1
            )
            
        # Create upsampling layers if needed
        if fusion_method == 'fpn':
            self.smooth_convs = torch.nn.ModuleDict()
            for layer_id in feature_specs:
                self.smooth_convs[layer_id] = torch.nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1
                )
                
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Fuse multi-scale features.
        
        Args:
            features: Dictionary mapping layer IDs to feature tensors
            
        Returns:
            Dictionary of fused features
        """
        # Apply lateral connections
        lateral_features = {}
        for layer_id, feat in features.items():
            lateral_features[layer_id] = self.lateral_convs[layer_id](feat)
            
        if self.fusion_method == 'fpn':
            return self._fpn_fusion(lateral_features)
        elif self.fusion_method == 'concat':
            return self._concat_fusion(lateral_features)
        elif self.fusion_method == 'sum':
            return self._sum_fusion(lateral_features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
    def _fpn_fusion(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """FPN-style top-down fusion."""
        # Group by resolution
        levels = {}
        for layer_id, feat in features.items():
            res = self.feature_specs[layer_id]['resolution']
            if res not in levels:
                levels[res] = []
            levels[res].append((layer_id, feat))
            
        # Sort resolutions (high to low)
        sorted_res = sorted(levels.keys(), reverse=True)
        
        # Top-down path
        fused_features = {}
        prev_feat = None
        
        for res in sorted_res:
            # Sum features at same resolution
            res_feat = None
            for layer_id, feat in levels[res]:
                if res_feat is None:
                    res_feat = feat
                else:
                    res_feat = res_feat + feat
                    
            # Add top-down feature if available
            if prev_feat is not None:
                # Upsample previous feature
                upsampled = torch.nn.functional.interpolate(
                    prev_feat,
                    size=res_feat.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
                res_feat = res_feat + upsampled
                
            # Apply smoothing
            for layer_id, _ in levels[res]:
                fused_features[layer_id] = self.smooth_convs[layer_id](res_feat)
                
            prev_feat = res_feat
            
        return fused_features
        
    def _concat_fusion(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Concatenation-based fusion (all features resized to same resolution)."""
        # Use highest resolution as target
        target_size = max(
            self.feature_specs[lid]['resolution'] 
            for lid in features.keys()
        )
        
        # Resize all features
        resized_features = []
        for layer_id, feat in features.items():
            if feat.shape[2] != target_size:
                feat = torch.nn.functional.interpolate(
                    feat,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
            resized_features.append(feat)
            
        # Concatenate
        concat_feat = torch.cat(resized_features, dim=1)
        
        # Return same feature for all layers (simplified)
        return {lid: concat_feat for lid in features.keys()}
        
    def _sum_fusion(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Element-wise sum fusion."""
        # Group by resolution
        levels = {}
        for layer_id, feat in features.items():
            res = self.feature_specs[layer_id]['resolution']
            if res not in levels:
                levels[res] = []
            levels[res].append((layer_id, feat))
            
        # Sum features at each resolution
        fused_features = {}
        for res, layer_feats in levels.items():
            summed = None
            layer_ids = []
            
            for layer_id, feat in layer_feats:
                if summed is None:
                    summed = feat.clone()
                else:
                    summed = summed + feat
                layer_ids.append(layer_id)
                
            # Assign summed feature to all layers at this resolution
            for layer_id in layer_ids:
                fused_features[layer_id] = summed
                
        return fused_features


if __name__ == "__main__":
    # Test multi-scale feature extraction
    print("Testing MultiScaleYOLOFeatureExtractor...")
    
    # Create dummy input
    batch_size = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dummy_input = torch.randn(batch_size, 3, 640, 640).to(device)
    
    # Initialize extractor (will fail without actual model)
    try:
        extractor = MultiScaleYOLOFeatureExtractor(
            model_path="ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx",
            target_layers=['layer_3', 'layer_22', 'layer_34'],
            execution_provider='cpu'
        )
        
        # Extract features
        features = extractor.extract_features(dummy_input)
        
        # Print feature info
        print("\nExtracted features:")
        for layer_id, feat in features.items():
            print(f"  {layer_id}: {feat.shape}")
            
        # Test feature fusion
        print("\nTesting FeaturePyramidFusion...")
        fusion = FeaturePyramidFusion(
            {lid: extractor.FEATURE_SPECS[lid] for lid in features.keys()},
            output_channels=256,
            fusion_method='fpn'
        )
        
        fused = fusion(features)
        print("\nFused features:")
        for layer_id, feat in fused.items():
            print(f"  {layer_id}: {feat.shape}")
            
    except Exception as e:
        print(f"Test failed (expected without model file): {e}")