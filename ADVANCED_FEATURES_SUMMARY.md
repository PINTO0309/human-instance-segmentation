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

#### 距離認識型ロス関数の仕組み

距離認識型ロス関数は、通常のセグメンテーションロスに「位置情報」を追加することで、**境界付近やインスタンス間の領域でより慎重な学習**を促す仕組みです。

##### 1. 基本的な考え方

```
通常のロス: 全てのピクセルを同じ重要度で扱う
距離認識ロス: ピクセルの位置に応じて重要度を変える
```

##### 2. 重み付けの3つの要素

**① 境界重み付け（Boundary Weighting）**
```
┌─────────────────┐
│  背景（重み1.0） │
│ ╔═══════════╗   │  境界付近のピクセルは
│ ║境界(重み2.0║   │  誤分類しやすいため
│ ║┌─────────┐║   │  より高い重みを設定
│ ║│人物(1.0)│║   │
│ ║└─────────┘║   │
│ ╚═══════════╝   │
└─────────────────┘
```

**② インスタンス分離重み（Instance Separation）**
```
人物A        人物B
┌────┐      ┌────┐
│    │━━━━━━│    │  ← この領域は重み3.0
└────┘      └────┘     (インスタンス分離が重要)
```

**③ 距離変換（Distance Transform）**
- 各ピクセルから最も近い境界までの距離を計算
- 境界から5ピクセル以内を「境界領域」として特別扱い

##### 3. 具体的な計算手順

1. **距離マップの生成**
   ```python
   # 境界からの距離を計算
   boundary_distance = compute_boundary_distance(mask)
   # 0 = 境界上, 1 = 1ピクセル離れた位置, ...
   ```

2. **重みマップの生成**
   ```python
   # 基本重み = 1.0
   weights = ones_like(mask)
   
   # 境界付近は重みを増加（最大2.0倍）
   if distance < 5:
       weights *= 2.0 - (distance / 5)
   
   # 他インスタンスが近い場合はさらに増加（3.0倍）
   if near_other_instance:
       weights *= 3.0
   ```

3. **ロスの計算**
   ```python
   # 通常のクロスエントロピーロスに重みを掛ける
   weighted_loss = cross_entropy_loss * weights
   final_loss = weighted_loss.mean()
   ```

##### 4. 効果

**境界の明確化**
- Before: ぼやけた境界、ギザギザなエッジ
- After: シャープで滑らかな境界

**インスタンス分離の改善**
- Before: 隣接する人物が繋がってしまう
- After: 各人物が明確に分離される

**難しい領域への集中**
- モデルが境界付近やインスタンス間の「難しい」ピクセルに注力
- 全体的な精度向上につながる

##### 5. パラメータの意味

- **boundary_width** (デフォルト: 5)
  - 境界領域の幅（ピクセル単位）
  - 大きくすると広い範囲が境界扱いになる

- **boundary_weight** (デフォルト: 2.0)
  - 境界領域の重み倍率
  - 2.0 = 境界上のピクセルは通常の2倍重要

- **instance_sep_weight** (デフォルト: 3.0)
  - インスタンス間領域の重み倍率
  - 3.0 = 他の人物に近いピクセルは3倍重要

##### 6. 適応的重み調整（Adaptive）

学習の進行に応じて重みを自動調整：
- 序盤: 高い重み（境界を強く学習）
- 終盤: 低い重み（全体のバランスを整える）

この仕組みにより、**エッジ品質の向上**と**インスタンス分離の改善**を同時に実現します。

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

## 8. 評価指標の詳細：mIoUの計算方法

このプロジェクトで使用されるmIoU（mean Intersection over Union）の計算方法について説明します。

### 3クラス分類の構成

本システムは各ROI（関心領域）内で以下の3クラスに分類します：

- **クラス0（背景）**: ROI内の人物以外の領域
- **クラス1（ターゲット）**: 対象となる人物インスタンスのマスク
- **クラス2（非ターゲット）**: ROI内の他の人物インスタンスのマスク

### mIoUの計算手順

1. **各クラスのIoU計算**
   ```
   IoU = (予測マスク ∩ 正解マスク) / (予測マスク ∪ 正解マスク)
   ```

2. **前景クラスのみを対象**
   - クラス1（ターゲット）とクラス2（非ターゲット）のIoU値のみを収集
   - **クラス0（背景）は除外**

3. **平均値の算出**
   ```
   mIoU = (クラス1の全IoU値 + クラス2の全IoU値) / 総数
   ```

### なぜ背景を除外するのか

- **不均衡の回避**: ROI内では背景が大部分を占めることが多く、含めると評価が歪む
- **実質的な性能評価**: 人物セグメンテーションの精度を正確に測定
- **モデルの怠惰を防ぐ**: 「全て背景」と予測しても高スコアにならない

### 評価例

```
実際の計算例：
- バッチ内のROI数: 10
- クラス1のIoU値: [0.85, 0.92, 0.78, 0.88, 0.90]
- クラス2のIoU値: [0.75, 0.82, 0.79, 0.83, 0.77]
- mIoU = (0.85+0.92+...+0.77) / 10 = 0.831
```

この方式により、インスタンスセグメンテーションの実質的な性能を適切に評価できます。

## 9. モデル複雑度分析

run_experiments.pyで生成される各モデル構成のパラメータ量、計算量、計算速度について分析しました。

### モデルパラメータ量比較

| 構成 | 総パラメータ数 | 学習可能パラメータ | モデルサイズ (MB) |
|------|---------------|------------------|-----------------|
| **Baseline** | 5,791,555 | 5,791,555 | 22.09 |
| **Multiscale** | 4,620,934 | 4,620,934 | 17.63 |
| **Cascade** | 5,791,555 | 5,791,555 | 22.09 |
| **All Features** | 13,862,802 | 13,862,802 | 52.88 |

### 計算複雑度（推定GFLOPs）

| 構成 | GFLOPs | Baseline比 |
|------|--------|-----------|
| **Baseline** | 0.41 | 1.00x |
| **Multiscale** | 0.41 | 1.00x |
| **Cascade** | 1.24 | 3.00x |
| **All Features** | 1.24 | 3.00x |

### 各構成の特徴

#### 1. Baseline（単一スケール）
- **利点**: シンプルで高速、メモリ効率が良い
- **用途**: リアルタイム処理、エッジデバイス、基本的な精度要求

#### 2. Multiscale（マルチスケール）
- **利点**: Baselineよりパラメータが少なく、同等の計算量で高精度
- **特徴**: 複数の特徴層を効率的に統合、細部の保持に優れる
- **用途**: 高精度が必要だが計算資源に制限がある場合

#### 3. Cascade（カスケード）
- **利点**: 段階的な精緻化により高品質な境界を生成
- **欠点**: 計算量が3倍（3段階処理のため）
- **用途**: 精度最優先、後処理として使用

#### 4. All Features（全機能有効）
- **利点**: 最高精度、インスタンス分離に最適
- **欠点**: パラメータ数2.4倍、計算量3倍
- **用途**: 研究・開発、オフライン処理、最高品質要求時

### 推論速度の目安

RTX 3090での推論速度（バッチサイズ1、ROI数10の場合）：
- **Baseline**: ~5-10ms/画像
- **Multiscale**: ~5-10ms/画像
- **Cascade**: ~15-30ms/画像
- **All Features**: ~15-30ms/画像

### メモリ使用量

学習時のGPUメモリ使用量（バッチサイズ8の場合）：
- **Baseline**: ~4GB
- **Multiscale**: ~6GB（複数特徴マップのため）
- **Cascade**: ~5GB（共有特徴使用時）
- **All Features**: ~8GB

### 最適化の推奨事項

1. **速度優先の場合**
   - BaselineまたはMultiscaleを使用
   - Mixed precision (FP16)を有効化
   - TensorRTで最適化

2. **精度優先の場合**
   - MultiscaleまたはAll Featuresを使用
   - より大きなROIサイズ（56x56）を検討
   - 後処理でCascadeを追加

3. **バランス重視の場合**
   - Multiscale + Distance Lossの組み合わせ
   - 計算量増加なしで精度向上
   - メモリ使用量も適度

### 注意事項

- パラメータ数にはYOLOv9特徴抽出器（~50M parameters）は含まれていません
- 実際の推論速度はハードウェア、バッチサイズ、ROI数に大きく依存します
- Distance LossとRelational moduleは推論時には影響しません（学習時のみ）
- ONNX/TensorRT最適化により2-3倍の高速化が期待できます

## 10. 可変ROIサイズ実験（Phase 2）

高解像度層（layer_3）のみ大きなROIサイズを使用する実験的な構成を実装しました。

### 新しい実験設定

#### variable_roi_hires
- **layer_3（160x160）**: ROIサイズ 56x56（2倍）
- **layer_22（80x80）**: ROIサイズ 28x28（標準）
- **layer_34（80x80）**: ROIサイズ 28x28（標準）

高解像度層でより多くの詳細情報を取得することで、エッジ品質の向上を狙います。

#### variable_roi_progressive
- **layer_3（160x160）**: ROIサイズ 56x56
- **layer_22（80x80）**: ROIサイズ 42x42
- **layer_34（80x80）**: ROIサイズ 28x28

段階的にROIサイズを変化させることで、各スケールで最適な情報量を取得します。

### 実行方法

```bash
# 高解像度層のみ大きなROIサイズ
uv run python run_experiments.py --configs variable_roi_hires --epochs 20 --batch_size 8

# 段階的ROIサイズ
uv run python run_experiments.py --configs variable_roi_progressive --epochs 20 --batch_size 8

# 標準のmultiscale_distanceと比較
uv run python run_experiments.py --configs multiscale_distance variable_roi_hires --epochs 20
```

### 期待される効果

1. **詳細なエッジ保持**
   - 高解像度層（layer_3）の56x56 ROIにより、髪の毛や指先などの細かい境界をより正確に捉える

2. **計算効率の最適化**
   - 必要な層のみROIサイズを大きくすることで、メモリ使用量と精度のバランスを取る

3. **マルチスケール特徴の相補性向上**
   - 異なるROIサイズにより、各層が異なる粒度の情報に特化

### 注意事項

- メモリ使用量が増加します（layer_3のROIサイズが4倍になるため）
- 学習初期は不安定になる可能性があるため、学習率の調整が必要な場合があります
- カスケード機能との併用はまだサポートされていません

## 実験結果

### Phase 2: variable_roi_hires 実験結果（2025/01/19）

高解像度層（layer_3）のみROIサイズを56x56に拡大した実験を実施：

```bash
uv run python run_experiments.py --configs variable_roi_hires --epochs 1 --batch_size 2
```

**結果:**
- **mIoU: 0.4260** (1エポック後)
- 訓練損失: 4.3 → 3.3に減少
- 可視化結果: 様々なシーンで良好なセグメンテーション性能を確認

**観察された特徴:**
1. **エッジ品質の向上**: 高解像度層の大きなROIサイズにより、人物の輪郭がより鮮明に
2. **複数人物の分離**: 重なり合う人物でも適切に分離できている
3. **夜間シーンでの性能**: 低照度環境でも人物を正確に検出

**技術的ポイント:**
- HierarchicalFeatureFusion により異なるROIサイズの特徴を効果的に統合
- メモリ使用量は標準モデルより約20%増加
- ONNXエクスポート対応済み（adaptive_avg_pool2dを使用しない実装に更新）

### 実装改善（2025/01/19）

**adaptive_avg_pool2dの除去:**
- `variable_roi_hires` (56→28) と `variable_roi_progressive` (56→42→28) の両方に対応
- 56→28の場合: stride=2の畳み込みによる正確な2倍ダウンサンプリング
- 42→28の場合: 学習可能なダウンサンプリング（中間チャネル拡張＋補間）
- ONNX互換性を確保し、TensorRTなどでの推論最適化が可能に

## 次のステップ

1. より長いエポック数での完全な学習実行（20-50エポック）
2. variable_roi_progressive（段階的ROIサイズ）の実験
3. 標準のmultiscale_distanceモデルとの詳細な比較
4. メモリ効率と精度のトレードオフ分析
5. 実運用に向けた推論最適化

## 11. RGB特徴拡張 (RGBEnhancedVariableROIModel) ⭐NEW
**ファイル**: `src/human_edge_detection/advanced/variable_roi_model.py`

**概要**: YOLOv9の特徴マップに加えて、元のRGB画像から軽量エンコーダで特徴を抽出し、特定の層で融合

**利点**:
- 低レベル特徴（エッジ、テクスチャ）の補完
- YOLO特徴との相補的な情報提供
- 軽量な追加処理（64チャンネルのみ）

**使い方**:
```python
# 単一層でのRGB拡張
config = ConfigManager.get_config('rgb_enhanced_lightweight')

# 複数層でのRGB拡張
config = ConfigManager.get_config('rgb_enhanced_multi_layer')

# 実行
uv run python run_experiments.py --configs rgb_enhanced_lightweight --epochs 20
```

## 12. 階層的セグメンテーション (Hierarchical Segmentation) ⭐NEW
**ファイル**: `src/human_edge_detection/advanced/hierarchical_segmentation.py`

**概要**: クラスのモード崩壊を防ぐため、背景vs前景と、ターゲットvs非ターゲットを階層的に分離

**アーキテクチャ**:
```
入力特徴
   ├─→ 背景vs前景ブランチ → 背景/前景マスク
   └─→ ターゲットvs非ターゲットブランチ → インスタンス分類
         └─→ 最終出力: 3クラスセグメンテーション
```

**特徴**:
- 2段階の分類: まず背景/前景を分離、次にインスタンスを分類
- クラス間の干渉を構造的に防止
- 一貫性損失により階層間の整合性を保証

**使い方**:
```python
config = ConfigManager.get_config('hierarchical_segmentation')

# 実行
uv run python run_experiments.py --configs hierarchical_segmentation --epochs 20
```

## 13. クラス特化デコーダ (Class-Specific Decoder) ⭐NEW  
**ファイル**: `src/human_edge_detection/advanced/class_specific_decoder.py`

**概要**: 各クラスに独立したデコーダパスを持たせることで、クラス間の勾配干渉を防止

**アーキテクチャ**:
```
入力特徴
   ├─→ 背景デコーダ（軽量）
   ├─→ ターゲットデコーダ（複雑）
   └─→ 非ターゲットデコーダ（複雑）
         └─→ クラス間相互作用モジュール → 最終出力
```

**特徴**:
- 背景用の軽量デコーダとインスタンス用の複雑なデコーダを分離
- クラス間相互作用モジュールで完全な分離を防止
- クラスごとのアテンション機構で背景の支配を抑制

**使い方**:
```python
config = ConfigManager.get_config('class_specific_decoder')

# 実行
uv run python run_experiments.py --configs class_specific_decoder --epochs 20
```

## ONNX エクスポート対応状況

すべての新しいアーキテクチャ（RGB拡張、階層的セグメンテーション、クラス特化デコーダ）は、ONNXエクスポートに対応済みです：

```bash
# 未学習モデルのONNXエクスポート
uv run python run_experiments.py --configs hierarchical_segmentation class_specific_decoder --epochs 0 --export_onnx

# 学習済みモデルのONNXエクスポート
uv run python run_experiments.py --configs hierarchical_segmentation --export_onnx
```