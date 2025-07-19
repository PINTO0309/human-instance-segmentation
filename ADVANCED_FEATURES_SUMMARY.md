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
uv run python train_advanced.py --config baseline

# マルチスケール特徴のみ
uv run python train_advanced.py --config multiscale

# マルチスケール + 距離認識ロス
uv run python train_advanced.py --config multiscale_distance

# 全機能有効
uv run python train_advanced.py --config full
```

### カスタム設定での実行

```bash
# 特定の層のみ使用
uv run python train_advanced.py --config multiscale \
  --config_modifications '{"multiscale.target_layers": ["layer_5", "layer_34"]}'

# バッチサイズとエポック数の変更
uv run python train_advanced.py --config multiscale_distance \
  --config_modifications '{"training.batch_size": 32, "training.num_epochs": 50}'
```

### 複数実験の比較

```bash
# 主要な設定で実験を実行して比較
uv run python run_experiments.py --configs baseline multiscale multiscale_distance full --epochs 20

# 既存の結果のみ比較
uv run python run_experiments.py --skip_training
```

## 推奨される検証手順

### Phase 1: ベースライン確立（1-2日）
```bash
# シンプルな設定で動作確認
uv run python train_advanced.py --config baseline --config_modifications '{"training.num_epochs": 10}'
```

### Phase 2: マルチスケール検証（2-3日）
```bash
# マルチスケールの効果を確認
uv run python train_advanced.py --config multiscale --config_modifications '{"training.num_epochs": 30}'

# 異なる層の組み合わせを試す
uv run python train_advanced.py --config multiscale \
  --config_modifications '{"multiscale.target_layers": ["layer_3", "layer_34"]}'
```

### Phase 3: 距離ロス追加（2-3日）
```bash
# 距離認識ロスの効果を確認
uv run python train_advanced.py --config multiscale_distance

# 境界重みの調整
uv run python train_advanced.py --config multiscale_distance \
  --config_modifications '{"distance_loss.boundary_weight": 3.0, "distance_loss.instance_sep_weight": 4.0}'
```

### Phase 4: カスケード追加（3-4日）
```bash
# カスケードの効果を確認
uv run python train_advanced.py --config multiscale_cascade

# 全機能での最終評価
uv run python train_advanced.py --config full
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
uv run python train_advanced.py --config baseline \
  --config_modifications '{"model.execution_provider": "cpu"}'
```

### メモリ不足
```bash
# バッチサイズとワーカー数を減らす
uv run python train_advanced.py --config multiscale \
  --config_modifications '{"training.batch_size": 8, "data.num_workers": 4}'
```

## Running Experiments

Run multiple experiments with different configurations:

```bash
# Run all configurations with default settings
uv run python run_experiments.py

# Run specific configurations with custom settings
uv run python run_experiments.py \
  --configs baseline multiscale multiscale_distance \
  --epochs 50 \
  --batch_size 16

# Compare existing results without training
uv run python run_experiments.py --skip_training

# Export ONNX models for existing experiments
uv run python run_experiments.py --export_onnx --configs baseline
```

### ONNX Export

Before each experiment starts, if a previous best model exists, it will be automatically exported to ONNX format. You can also export existing models manually:

```bash
# Export specific model to ONNX
uv run python -m src.human_edge_detection.export_onnx \
  --checkpoint experiments/baseline/checkpoints/best_model.pth \
  --output experiments/baseline/checkpoints/best_model.onnx

# Export all experiment models
uv run python run_experiments.py --export_onnx
```

Note: ONNX export is currently supported only for baseline single-scale models. Multi-scale models require custom export logic due to their multiple feature inputs.

## 7. 高度な検証とテスト推論

run_experiments.pyで学習したモデルを検証するための専用スクリプト `validate_advanced.py` を使用できます。

### 基本的な使用方法

```bash
# 最新のチェックポイントを検証
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/checkpoint_epoch_0013.pth

# ベストモデルを検証
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth
```

### 詳細なオプション

#### 1. 全検証画像での評価
デフォルトでは4枚のテスト画像のみを使用しますが、全検証画像（100枚）で評価することも可能：

```bash
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth --validate_all
```

#### 2. 可視化なしでメトリクスのみ取得
高速にメトリクスのみを取得したい場合：

```bash
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth --no_visualization
```

#### 3. 設定の上書き
チェックポイントに保存された設定を一時的に変更：

```bash
# バッチサイズを変更
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth \
  --override training.batch_size=16

# 複数の設定を変更
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth \
  --override training.batch_size=8 \
  --override data.num_workers=4
```

#### 4. CPUでの実行
GPUが利用できない環境での検証：

```bash
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth --device cpu
```

### 出力内容

スクリプトは以下の情報を出力します：

1. **実験情報**
   - 実験名と説明
   - 有効化された機能（マルチスケール、距離認識ロスなど）

2. **検証メトリクス**
   - Total Loss: 総合損失
   - CE Loss: クロスエントロピー損失
   - Dice Loss: Dice損失
   - mIoU: 平均IoU
   - 各クラスのIoU

3. **可視化画像**（--no_visualizationを指定しない場合）
   - `experiments/{実験名}/validation_results/` に保存
   - Ground Truthと予測結果の比較画像

4. **メトリクスファイル**
   - `{checkpoint_name}_validation_metrics.json` として保存
   - プログラムから読み込み可能なJSON形式

### 特徴

- **自動モデル再構築**: チェックポイントから完全な設定を読み込み、適切なモデルアーキテクチャを自動的に構築
- **高度な機能対応**: マルチスケール、距離認識ロス、カスケード、リレーショナルモジュールなど全ての機能に対応
- **柔軟な設定**: 実行時に設定を上書き可能
- **バッチ処理対応**: 複数のチェックポイントを一度に検証することも可能（スクリプトの拡張により）

### 使用例：実験の比較

異なる実験のベストモデルを比較：

```bash
# ベースライン
uv run python validate_advanced.py experiments/baseline/checkpoints/best_model.pth

# マルチスケール
uv run python validate_advanced.py experiments/multiscale/checkpoints/best_model.pth

# マルチスケール＋距離認識ロス
uv run python validate_advanced.py experiments/multiscale_distance/checkpoints/best_model.pth

# 全機能有効
uv run python validate_advanced.py experiments/all_features/checkpoints/best_model.pth
```

## 次のステップ

1. 各設定での完全な学習実行と性能評価
2. 最適なハイパーパラメータの探索
3. 実運用に向けた推論最適化
4. TensorRTやONNX最適化の適用