#!/usr/bin/env python3
"""
Export all edge smoothing PyTorch models to ONNX format.
Based on models defined in Refine_the_Binary_Mask.md
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path


class BinaryMaskEdgeSmoothing(nn.Module):
    """バイナリマスクのエッジ平滑化PyTorchモデル"""
    
    def __init__(self, threshold=0.5, blur_strength=3.0):
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
    
    def forward(self, mask):
        # 1. エッジ検出
        edges = F.conv2d(mask, self.laplacian_kernel, padding=1)
        
        # 2. エッジの絶対値
        edge_abs = torch.abs(edges)
        
        # 3. エッジマスク生成（Sigmoid使用）
        edge_scaled = edge_abs * self.blur_strength
        edge_mask = torch.sigmoid(edge_scaled)
        
        # 4. ガウシアンブラー適用
        blurred = F.conv2d(mask, self.gaussian_kernel, padding=1)
        
        # 5. ブレンディング
        smoothed = mask * (1 - edge_mask) + blurred * edge_mask
        
        # 6. 最終的な二値化
        binary_output = (smoothed > self.threshold).float()
        
        return binary_output


class DirectionalEdgeSmoothing(nn.Module):
    """エッジの方向性を考慮した平滑化PyTorchモデル"""
    
    def __init__(self, num_directions=4):
        super().__init__()
        self.num_directions = num_directions
        
        # Sobelフィルタで方向検出
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_y', sobel_y)
        
        # 方向別ブラーカーネル定義
        # 水平方向
        h_blur = torch.tensor(
            [[0.1, 0.2, 0.4, 0.2, 0.1]]
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('h_blur', h_blur)
        
        # 垂直方向
        v_blur = torch.tensor(
            [[0.1], [0.2], [0.4], [0.2], [0.1]]
        ).unsqueeze(0).unsqueeze(0)
        self.register_buffer('v_blur', v_blur)
        
        # 対角方向（45度）
        diag1_blur = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 0.1]
        ]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('diag1_blur', diag1_blur)
        
        # 対角方向（135度）
        diag2_blur = torch.tensor([
            [0.0, 0.0, 0.1],
            [0.0, 0.8, 0.0],
            [0.1, 0.0, 0.0]
        ]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('diag2_blur', diag2_blur)
    
    def forward(self, mask):
        # エッジ方向の検出
        edge_x = F.conv2d(mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask, self.sobel_y, padding=1)
        
        # エッジ強度計算
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        
        # エッジ方向計算（ラジアン）
        edge_angle = torch.atan2(edge_y, edge_x)
        
        # 方向別のブラー適用
        blur_h = F.conv2d(mask, self.h_blur, padding=(0, 2))
        blur_v = F.conv2d(mask, self.v_blur, padding=(2, 0))
        blur_d1 = F.conv2d(mask, self.diag1_blur, padding=1)
        blur_d2 = F.conv2d(mask, self.diag2_blur, padding=1)
        
        # 方向に基づく重み計算
        weight_h = torch.cos(edge_angle)**2
        weight_v = torch.sin(edge_angle)**2
        weight_d1 = torch.cos(edge_angle - np.pi/4)**2 * 0.5
        weight_d2 = torch.cos(edge_angle + np.pi/4)**2 * 0.5
        
        # 重みの正規化
        weight_sum = weight_h + weight_v + weight_d1 + weight_d2 + 1e-8
        weight_h = weight_h / weight_sum
        weight_v = weight_v / weight_sum
        weight_d1 = weight_d1 / weight_sum
        weight_d2 = weight_d2 / weight_sum
        
        # 方向別ブラーの合成
        blurred = (blur_h * weight_h + blur_v * weight_v +
                  blur_d1 * weight_d1 + blur_d2 * weight_d2)
        
        # エッジ強度に基づくブレンディング
        edge_mask = torch.sigmoid(edge_magnitude * 3.0)
        smoothed = mask * (1 - edge_mask) + blurred * edge_mask
        
        # 二値化
        binary_output = (smoothed > 0.5).float()
        
        return binary_output


class AdaptiveEdgeSmoothing(nn.Module):
    """適応的なエッジ平滑化PyTorchモデル（パラメータ動的調整）"""
    
    def __init__(self):
        super().__init__()
        
        # Laplacianカーネル（エッジ検出用）
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
        
    def forward(self, mask, blur_strength, edge_sensitivity, final_threshold):
        """
        Args:
            mask: バイナリマスク [B, 1, H, W]
            blur_strength: ブラーの強度（1.0-5.0） [B, 1]
            edge_sensitivity: エッジ検出の感度（0.5-2.0） [B, 1]
            final_threshold: 最終二値化の閾値（0.3-0.7） [B, 1]
        """
        B, C, H, W = mask.shape
        
        # エッジ検出
        edges = F.conv2d(mask, self.laplacian_kernel, padding=1)
        edges = torch.abs(edges)
        
        # 適応的な閾値でエッジマスク生成
        edge_threshold = 0.5 * edge_sensitivity.view(B, 1, 1, 1)
        edge_mask = (edges > edge_threshold).float()
        
        # blur_strengthに基づくスケーリング（動的カーネルサイズの代替）
        # 実際のONNXでは固定カーネルを使用し、blur_strengthで結果をスケーリング
        kernel_size = 5
        avg_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        avg_kernel = avg_kernel.to(mask.device)
        
        # 平均フィルタで近似的なブラー
        smoothed_base = F.conv2d(
            mask, 
            avg_kernel, 
            padding=kernel_size//2
        )
        
        # blur_strengthによるブラー効果の調整
        blur_factor = blur_strength.view(B, 1, 1, 1) / 3.0  # 正規化
        smoothed = mask * (1 - blur_factor) + smoothed_base * blur_factor
        
        # エッジ部分のみ平滑化
        result = mask * (1 - edge_mask) + smoothed * edge_mask
        
        # 最終的な二値化
        final_mask = (result > final_threshold.view(B, 1, 1, 1)).float()
        
        return final_mask


class OptimizedEdgeSmoothing(nn.Module):
    """最適化されたエッジ平滑化PyTorchモデル（FP16対応）"""
    
    def __init__(self, use_fp16=True):
        super().__init__()
        self.use_fp16 = use_fp16
        
        # 1. Depthwise Convolutionの使用
        # エッジ検出用のDepthwise Laplacianカーネル
        self.edge_detector = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            groups=1,  # Depthwise convolution
            bias=False
        )
        
        # Laplacianカーネルの重みを設定
        laplacian_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.edge_detector.weight.data = laplacian_kernel
        self.edge_detector.weight.requires_grad = False
        
        # 3. 最適化されたガウシアンブラー（Separable convolution）
        # 水平方向のガウシアンカーネル
        self.gaussian_h = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 5),
            padding=(0, 2),
            bias=False
        )
        gaussian_h_kernel = torch.tensor(
            [0.0625, 0.25, 0.375, 0.25, 0.0625], 
            dtype=torch.float32
        ).reshape(1, 1, 1, 5)
        self.gaussian_h.weight.data = gaussian_h_kernel
        self.gaussian_h.weight.requires_grad = False
        
        # 垂直方向のガウシアンカーネル
        self.gaussian_v = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(5, 1),
            padding=(2, 0),
            bias=False
        )
        gaussian_v_kernel = torch.tensor(
            [0.0625, 0.25, 0.375, 0.25, 0.0625], 
            dtype=torch.float32
        ).reshape(1, 1, 5, 1)
        self.gaussian_v.weight.data = gaussian_v_kernel
        self.gaussian_v.weight.requires_grad = False
        
        # 定数バッファ
        self.register_buffer('edge_scale', torch.tensor(3.0))
        self.register_buffer('threshold', torch.tensor(0.5))
        
    def forward(self, mask):
        """最適化された順伝播
        
        Args:
            mask: バイナリマスク [B, 1, H, W]
        
        Returns:
            smoothed_mask: 平滑化されたマスク [B, 1, H, W]
        """
        # FP16への変換（オプション）
        if self.use_fp16 and mask.dtype != torch.float16:
            mask = mask.half()
            
        # 1. エッジ検出（Depthwise Convolution）
        edges = self.edge_detector(mask)
        
        # 2. Fused Operations（複数の演算を融合）
        # エッジの絶対値とスケーリングを一度に
        edge_abs_scaled = torch.abs(edges) * self.edge_scale
        
        # 3. Separable Gaussian blur（2パスで効率化）
        blur_horizontal = self.gaussian_h(mask)
        blurred = self.gaussian_v(blur_horizontal)
        
        # 4. Sigmoid with approximation（高速近似）
        # sigmoid(x) ≈ clip((x + 0.5) * 0.5, 0, 1) for small x
        edge_mask = torch.clamp((edge_abs_scaled + 0.5) * 0.5, 0, 1)
        
        # 5. ブレンディング（SIMD最適化を意識）
        # FMA (Fused Multiply-Add) を活用できる形式
        # result = mask * (1 - edge_mask) + blurred * edge_mask
        smoothed = mask * (1 - edge_mask) + blurred * edge_mask
        
        # 6. 量子化対応の二値化
        binary_output = (smoothed > self.threshold).float()
        
        # FP16で返す（必要に応じて）
        if self.use_fp16:
            return binary_output.half()
        return binary_output


def export_all_models(output_dir="onnx_models", fp16=False):
    """すべてのエッジ平滑化モデルをONNXにエクスポート"""
    
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    
    print(f"Exporting edge smoothing models to {output_dir}")
    print(f"FP16 mode: {fp16}")
    print("="*60)
    
    # 1. 基本的なエッジ平滑化モデル
    print("\n1. Exporting BinaryMaskEdgeSmoothing...")
    basic_model = BinaryMaskEdgeSmoothing(threshold=0.5, blur_strength=3.0)
    basic_model.eval()
    
    dummy_input = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        basic_model,
        dummy_input,
        str(output_dir / "basic_edge_smoothing.onnx"),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mask'],
        output_names=['smoothed_mask'],
        dynamic_axes={
            'mask': {0: 'batch', 2: 'height', 3: 'width'},
            'smoothed_mask': {0: 'batch', 2: 'height', 3: 'width'}
        },
        verbose=False
    )
    print("✓ Exported: basic_edge_smoothing.onnx")
    
    # 2. 方向性エッジ平滑化モデル
    print("\n2. Exporting DirectionalEdgeSmoothing...")
    directional_model = DirectionalEdgeSmoothing(num_directions=4)
    directional_model.eval()
    
    torch.onnx.export(
        directional_model,
        dummy_input,
        str(output_dir / "directional_edge_smoothing.onnx"),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mask'],
        output_names=['smoothed_mask'],
        dynamic_axes={
            'mask': {0: 'batch', 2: 'height', 3: 'width'},
            'smoothed_mask': {0: 'batch', 2: 'height', 3: 'width'}
        },
        verbose=False
    )
    print("✓ Exported: directional_edge_smoothing.onnx")
    
    # 3. 適応的エッジ平滑化モデル
    print("\n3. Exporting AdaptiveEdgeSmoothing...")
    adaptive_model = AdaptiveEdgeSmoothing()
    adaptive_model.eval()
    
    # 適応的モデルは追加のパラメータが必要
    dummy_blur = torch.tensor([3.0])
    dummy_sensitivity = torch.tensor([1.0])
    dummy_threshold = torch.tensor([0.5])
    
    torch.onnx.export(
        adaptive_model,
        (dummy_input, dummy_blur, dummy_sensitivity, dummy_threshold),
        str(output_dir / "adaptive_edge_smoothing.onnx"),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mask', 'blur_strength', 'edge_sensitivity', 'final_threshold'],
        output_names=['smoothed_mask'],
        dynamic_axes={
            'mask': {0: 'batch', 2: 'height', 3: 'width'},
            'smoothed_mask': {0: 'batch', 2: 'height', 3: 'width'}
        },
        verbose=False
    )
    print("✓ Exported: adaptive_edge_smoothing.onnx")
    
    # 4. 最適化されたエッジ平滑化モデル（FP32）
    print("\n4. Exporting OptimizedEdgeSmoothing (FP32)...")
    optimized_model_fp32 = OptimizedEdgeSmoothing(use_fp16=False)
    optimized_model_fp32.eval()
    
    torch.onnx.export(
        optimized_model_fp32,
        dummy_input,
        str(output_dir / "optimized_edge_smoothing_fp32.onnx"),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mask'],
        output_names=['smoothed_mask'],
        dynamic_axes={
            'mask': {0: 'batch', 2: 'height', 3: 'width'},
            'smoothed_mask': {0: 'batch', 2: 'height', 3: 'width'}
        },
        verbose=False
    )
    print("✓ Exported: optimized_edge_smoothing_fp32.onnx")
    
    # 5. 最適化されたエッジ平滑化モデル（FP16）
    if fp16:
        print("\n5. Exporting OptimizedEdgeSmoothing (FP16)...")
        optimized_model_fp16 = OptimizedEdgeSmoothing(use_fp16=True)
        optimized_model_fp16.eval()
        optimized_model_fp16.half()
        
        dummy_input_fp16 = dummy_input.half()
        
        torch.onnx.export(
            optimized_model_fp16,
            dummy_input_fp16,
            str(output_dir / "optimized_edge_smoothing_fp16.onnx"),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['mask'],
            output_names=['smoothed_mask'],
            dynamic_axes={
                'mask': {0: 'batch', 2: 'height', 3: 'width'},
                'smoothed_mask': {0: 'batch', 2: 'height', 3: 'width'}
            },
            verbose=False
        )
        print("✓ Exported: optimized_edge_smoothing_fp16.onnx")
    
    print("\n" + "="*60)
    print(f"All models exported successfully to {output_dir}")
    
    # モデルサマリー
    print("\nExported models:")
    for onnx_file in output_dir.glob("*.onnx"):
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"  - {onnx_file.name}: {size_mb:.2f} MB")


def test_onnx_models(output_dir="onnx_models"):
    """エクスポートされたONNXモデルをテスト"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("Warning: onnxruntime not installed. Skipping model testing.")
        return
    
    output_dir = Path(output_dir)
    print("\nTesting exported ONNX models...")
    print("="*60)
    
    # テスト用のダミー入力
    test_input = np.random.rand(1, 1, 128, 128).astype(np.float32)
    
    for onnx_file in sorted(output_dir.glob("*.onnx")):
        print(f"\nTesting {onnx_file.name}...")
        
        try:
            # ONNXランタイムセッションの作成
            session = ort.InferenceSession(str(onnx_file))
            
            # 入力名の取得
            input_names = [inp.name for inp in session.get_inputs()]
            
            # モデルに応じた入力準備
            if "adaptive" in onnx_file.name:
                # 適応的モデルは追加パラメータが必要
                inputs = {
                    'mask': test_input,
                    'blur_strength': np.array([3.0], dtype=np.float32),
                    'edge_sensitivity': np.array([1.0], dtype=np.float32),
                    'final_threshold': np.array([0.5], dtype=np.float32)
                }
            else:
                inputs = {'mask': test_input}
            
            # 推論実行
            outputs = session.run(None, inputs)
            
            print(f"  ✓ Model loaded and inference successful")
            print(f"  - Input shape: {test_input.shape}")
            print(f"  - Output shape: {outputs[0].shape}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export edge smoothing PyTorch models to ONNX format"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='onnx_models',
        help='Output directory for ONNX models (default: onnx_models)'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Also export FP16 version of optimized model'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test exported models with onnxruntime'
    )
    
    args = parser.parse_args()
    
    # モデルのエクスポート
    export_all_models(args.output_dir, args.fp16)
    
    # テスト実行
    if args.test:
        test_onnx_models(args.output_dir)


if __name__ == "__main__":
    main()