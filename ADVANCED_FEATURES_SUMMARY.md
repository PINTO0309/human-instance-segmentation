# 段階的改善実装まとめ

## 実装完了項目

### 1. マルチスケール特徴抽出 ✅
**ファイル**: 
- `src/human_edge_detection/advanced/multi_scale_extractor.py`
- `src/human_edge_detection/advanced/multi_scale_model.py`

**特徴**:
- YOLOv9の5つの特徴マップ（160x160と80x80の異なる解像度）を活用
- 適応的特徴融合（adaptive fusion）による効率的な統合
- FPN風の特徴ピラミッド構造

**使用可能な特徴マップ**:
```python
- layer_3: 256ch, 160x160 (高解像度、浅い層)
- layer_19: 256ch, 160x160 (高解像度、中間層)  
- layer_5: 512ch, 80x80 (中解像度、浅い層)
- layer_22: 512ch, 80x80 (中解像度、中間層)
- layer_34: 1024ch, 80x80 (低解像度、深い層)
```

### 2. 距離認識型ロス関数 ✅
**ファイル**: `src/human_edge_detection/advanced/distance_aware_loss.py`

**特徴**:
- 境界領域への重み付け（boundary_width、boundary_weight）
- インスタンス間距離に基づく分離強化（instance_sep_weight）
- 適応的重み調整オプション

### 3. カスケード型セグメンテーション ✅
**ファイル**: `src/human_edge_detection/advanced/cascade_segmentation.py`

**特徴**:
- 3段階の段階的精緻化
  - Stage 1: 粗いセグメンテーション
  - Stage 2: 境界精緻化
  - Stage 3: インスタンス分離
- 各段階で異なる重み付け可能

### 4. 統合トレーニングパイプライン ✅
**ファイル**: 
- `train_advanced.py`
- `src/human_edge_detection/experiments/config_manager.py`
- `src/human_edge_detection/experiments/progressive_training.py`

**特徴**:
- 設定ベースの機能ON/OFF切り替え
- 事前定義された実験設定（baseline、multiscale、full等）
- 段階的な機能追加サポート

### 5. 段階的検証システム ✅
**ファイル**: `run_experiments.py`

**特徴**:
- 複数の設定での自動実験実行
- 結果の自動比較・可視化
- 各機能の効果測定

## 使用方法

### 基本的な学習実行

```bash
# ベースライン（シングルスケール）
python train_advanced.py --config baseline

# マルチスケール特徴のみ
python train_advanced.py --config multiscale

# マルチスケール + 距離認識ロス
python train_advanced.py --config multiscale_distance

# 全機能有効
python train_advanced.py --config full
```

### カスタム設定での実行

```bash
# 特定の層のみ使用
python train_advanced.py --config multiscale \
  --config_modifications '{"multiscale.target_layers": ["layer_5", "layer_34"]}'

# バッチサイズとエポック数の変更
python train_advanced.py --config multiscale_distance \
  --config_modifications '{"training.batch_size": 32, "training.num_epochs": 50}'
```

### 複数実験の比較

```bash
# 主要な設定で実験を実行して比較
python run_experiments.py --configs baseline multiscale multiscale_distance full --epochs 20

# 既存の結果のみ比較
python run_experiments.py --skip_training
```

## 推奨される検証手順

### Phase 1: ベースライン確立（1-2日）
```bash
# シンプルな設定で動作確認
python train_advanced.py --config baseline --config_modifications '{"training.num_epochs": 10}'
```

### Phase 2: マルチスケール検証（2-3日）
```bash
# マルチスケールの効果を確認
python train_advanced.py --config multiscale --config_modifications '{"training.num_epochs": 30}'

# 異なる層の組み合わせを試す
python train_advanced.py --config multiscale \
  --config_modifications '{"multiscale.target_layers": ["layer_3", "layer_34"]}'
```

### Phase 3: 距離ロス追加（2-3日）
```bash
# 距離認識ロスの効果を確認
python train_advanced.py --config multiscale_distance

# 境界重みの調整
python train_advanced.py --config multiscale_distance \
  --config_modifications '{"distance_loss.boundary_weight": 3.0, "distance_loss.instance_sep_weight": 4.0}'
```

### Phase 4: カスケード追加（3-4日）
```bash
# カスケードの効果を確認
python train_advanced.py --config multiscale_cascade

# 全機能での最終評価
python train_advanced.py --config full
```

## 設定ファイルの構造

各実験の設定は以下の構造で管理：

```
experiments/
├── <experiment_name>/
│   ├── checkpoints/       # モデルチェックポイント
│   ├── logs/             # TensorBoardログ
│   ├── configs/          # 実験設定
│   └── visualizations/   # 可視化結果
```

## パフォーマンス最適化のヒント

1. **メモリ不足の場合**:
   - バッチサイズを減らす
   - 使用する特徴層を減らす（例: layer_5とlayer_34のみ）
   - mixed_precisionを有効化

2. **学習が不安定な場合**:
   - 学習率を下げる
   - gradient_clipを調整
   - 段階的に機能を追加（progressive training）

3. **推論速度を上げたい場合**:
   - efficientコンフィグを使用
   - カスケードを無効化
   - 特徴層を2つに限定

## 未実装機能

### インスタンス間関係モデル（優先度: 低）
計算コストが高く、効果が限定的と予測されるため、現時点では未実装。
必要に応じて`src/human_edge_detection/advanced/relational_module.py`として実装可能。

## トラブルシューティング

### scipy未インストールエラー
```bash
uv add scipy
```

### ONNX Runtime エラー
```bash
# CUDAプロバイダーが使えない場合
--config_modifications '{"model.execution_provider": "cpu"}'
```

### メモリ不足
```bash
# バッチサイズとワーカー数を減らす
--config_modifications '{"training.batch_size": 8, "data.num_workers": 4}'
```

## 次のステップ

1. 各設定での完全な学習実行と性能評価
2. 最適なハイパーパラメータの探索
3. 実運用に向けた推論最適化
4. TensorRTやONNX最適化の適用