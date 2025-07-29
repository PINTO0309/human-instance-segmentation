#!/usr/bin/env python3
"""Analyze the double normalization problem."""

import torch
import numpy as np

print("二重正規化問題の分析")
print("="*60)

# 1. 事前学習済みUNetの内部正規化
print("\n1. PreTrainedPeopleSegmentationUNetの内部正規化:")
print("   - mean = [0.5, 0.5, 0.5]")
print("   - std = [0.5, 0.5, 0.5]")
print("   - 効果: [0, 1] → [-1, 1] に変換")

# 2. Augmentationsの正規化
print("\n2. Augmentationsの正規化:")
print("   - 元の設定: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (ImageNet)")
print("   - ユーザー変更後: mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0] (正規化なし)")

# 3. データの流れを検証
print("\n3. データの流れ:")

# サンプル画像（[0, 1]範囲）
sample_image = np.array([0.5, 0.5, 0.5])

print("\n【元の設定（ImageNet正規化あり）】")
# Augmentationsでの正規化（ImageNet）
aug_normalized = (sample_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
print(f"   入力画像: {sample_image}")
print(f"   Augmentations後: {aug_normalized}")

# モデル内部での正規化
model_normalized = (aug_normalized - 0.5) / 0.5
print(f"   モデル内部後: {model_normalized}")
print(f"   → 異常な値！")

print("\n【ユーザー変更後（正規化なし）】")
# Augmentationsでの正規化（なし）
aug_no_norm = (sample_image - 0.0) / 1.0  # 変化なし
print(f"   入力画像: {sample_image}")
print(f"   Augmentations後: {aug_no_norm}")

# モデル内部での正規化
model_normalized = (aug_no_norm - 0.5) / 0.5
print(f"   モデル内部後: {model_normalized}")
print(f"   → 正常な値！")

print("\n4. 問題の根本原因:")
print("   - 事前学習済みモデルは[0, 1]範囲の入力を期待")
print("   - 内部で[-1, 1]に正規化")
print("   - Augmentationsで追加の正規化を行うと二重正規化になる")

print("\n5. なぜバリデーションが異常になったか:")
print("   - get_val_transforms()も修正が必要")
print("   - 現在のval_transformsはまだImageNet正規化を使用している可能性")

print("\n" + "="*60)
print("推奨対策:")
print("="*60)
print("\n1. すべての正規化を統一:")
print("   - get_val_transforms()もmean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]に変更")
print("   - get_train_transforms()、get_light_train_transforms()、get_heavy_train_transforms()も同様")
print("\n2. または、モデル側を調整:")
print("   - PreTrainedPeopleSegmentationUNetの正規化を無効化")
print("   - mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]を渡す")