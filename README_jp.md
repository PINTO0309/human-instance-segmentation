# 人物インスタンスセグメンテーション

[English](./README.md) / 日本語

EfficientNetベースの教師モデルからの知識蒸留を用いた、人物検出のための軽量ROIベース階層型インスタンスセグメンテーションモデル。2段階の階層型アーキテクチャと温度進行蒸留技術により、効率的なリアルタイム性能を実現します。

- インスタンスセグメンテーションモード

  <img width="640" height="424" alt="image" src="https://github.com/user-attachments/assets/10431a3c-0bba-422f-9d98-67af8e77b777" />

- バイナリマスクモード

  <img width="427" height="640" alt="000000229849_binary" src="https://github.com/user-attachments/assets/f73ef1b2-36d8-4eb0-b167-06764e05ebe3" />

## 目次
- [アーキテクチャ概要](#アーキテクチャ概要)
- [アーキテクチャ詳細](#アーキテクチャ詳細)
- [アーキテクチャ図](#アーキテクチャ図)
- [学習パイプライン](#学習パイプライン)
- [リファインメント機構](#リファインメント機構)
- [データセット構造](#データセット構造)
- [環境セットアップ](#環境セットアップ)
- [UNet蒸留コマンド](#unet蒸留コマンド)
- [ROIベース階層型学習](#roiベース階層型学習)
- [ONNXエクスポート](#onnxエクスポート)
- [推論テスト](#推論テスト)
- [ライセンス](#ライセンス)
- [引用と謝辞](#引用と謝辞)

## アーキテクチャ概要

人物インスタンスセグメンテーションモデルは、以下を組み合わせた洗練された階層型セグメンテーションアプローチを採用しています：
- **2段階アーキテクチャ**: 粗いバイナリセグメンテーションに続くROIベースのインスタンスリファインメント
- **マルチアーキテクチャサポート**: B0（軽量）、B1（バランス型）、B7（高精度）バリアント
- **知識蒸留**: 効率的な知識転送のための温度進行（10→1）
- **リアルタイム処理**: ONNX/TensorRTデプロイメントでエッジデバイス向けに最適化

### 主な特徴
- 個別の特徴抽出なしでの直接RGB入力処理
- 堅牢な前景/背景バイナリセグメンテーションのための事前学習済みUNet
- 正確なインスタンス分離のためのROIベースリファインメント
- 3クラス出力システム（背景、ターゲットインスタンス、非ターゲットインスタンス）
- 拡張とエッジスムージングによるオプションの後処理

## アーキテクチャ詳細

### モデル階層

#### B0アーキテクチャ（軽量）
- **エンコーダ**: EfficientNet-B0ベース（timm-efficientnet-b0）
- **パラメータ数**: 約5.3M
- **ONNXサイズ**: 約71MB
- **ROIサイズ**: 64×48（標準）、80×60（拡張）
- **マスクサイズ**: 128×96（標準）、160×120（拡張）
- **ユースケース**: リアルタイムエッジデプロイメント、モバイルデバイス

#### B1アーキテクチャ（バランス型）
- **エンコーダ**: EfficientNet-B1ベース（timm-efficientnet-b1）
- **パラメータ数**: 約7.8M
- **ONNXサイズ**: 約81MB
- **ROIサイズ**: 64×48（標準）、80×60（拡張）
- **マスクサイズ**: 128×96（標準）、160×120（拡張）
- **ユースケース**: パフォーマンス/精度のバランスの取れたトレードオフ

#### B7アーキテクチャ（高精度）
- **エンコーダ**: EfficientNet-B7ベース（timm-efficientnet-b7）
- **パラメータ数**: 約66M
- **ONNXサイズ**: 約90MB
- **ROIサイズ**: 64×48（標準）、80×60（拡張）、128×96（ウルトラ）
- **マスクサイズ**: 128×96（標準）、160×120（拡張）、256×192（ウルトラ）
- **ユースケース**: 最大精度、サーバーデプロイメント

### コアコンポーネント

#### 1. 事前学習済みUNetモジュール
- **アーキテクチャ**: 残差ブロック付き拡張UNet
- **正規化**: 安定した学習のためのLayerNorm2D
- **活性化**: ReLU/SiLU設定可能
- **出力**: バイナリ前景/背景マスク
- **学習**: インスタンスセグメンテーション学習中は凍結

#### 2. ROI抽出モジュール
- **入力**: COCOバウンディングボックス
- **正規化**: [0, 1]に正規化された座標
- **プーリング**: 設定可能な出力サイズでの動的RoI Align
- **バッチ処理**: 効率的なマルチインスタンス処理

#### 3. インスタンスセグメンテーションヘッド
- **アーキテクチャ**: アテンションモジュール付き階層型UNet V2
- **クラス**: 3クラスセグメンテーション（背景、ターゲット、非ターゲット）
- **特徴**:
  - 特徴リファインメントのための残差ブロック
  - 人物境界に焦点を当てるアテンションゲーティング
  - より良いインスタンス分離のための距離認識損失
  - 輪郭検出補助タスク

#### 4. 損失関数
- **主損失**: 重み付きクロスエントロピー + Dice損失
- **クラス重み**:
  - 背景: 0.538
  - ターゲット: 0.750
  - 非ターゲット: 1.712（1.2倍ブースト）
- **補助損失**:
  - 境界認識のための距離変換損失
  - エッジリファインメントのための輪郭検出損失
  - インスタンス区別のための分離認識重み付け

## アーキテクチャ図

```
             ┌─────────────────────────────┐           ┌──────────────────────────────┐
             │       Input RGB Image       │           │             ROIs             │
             │        [B, 3, H, W]         │           │            [N, 5]            │
             └──────────────┬──────────────┘           │ [batch_idx, x1, y1, x2, y2]  │
                            │                          │ (0-1 normalized coordinates) │
                            │                          └──────────────┬───────────────┘
                            │                                         │
             ┌──────────────▼──────────────┐                          │
             │   Pretrained UNet Module    │                          │
             │    (Frozen during training) │                          │
             │   Output: Binary FG/BG      │                          │
             └──────────────┬──────────────┘                          │
                            │                                         │
              ┌─────────────┴─────────────┐                           │
              │                           │                           │
  ┌───────────▼───────────┐   ┌───────────▼──────────┐                │
  │  Binary Mask Output   │   │   Feature Maps       │                │
  │   [B, 1, H, W]        │   │   for ROI Pooling    │                │
  └───────────┬───────────┘   └───────────┬──────────┘                │
              │                           │                           │
              └─────────────┬─────────────┘                           │
                            │◀────────────────────────────────────────┘
            ┌───────────────▼───────────────┐
            │   Dynamic RoI Align           │
            │  Output: [N, C, H_roi, W_roi] │
            └───────────────┬───────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
 ┌────────────▼───────────┐   ┌───────────▼────────────┐
 │      EfficientNet      │   │  Pretrained UNet Mask  │
 │      Encoder           │   │  (for each ROI)        │
 │      (B0/B1/B7)        │   │  [N, 1, H_roi, W_roi]  │
 └────────────┬───────────┘   └───────────┬────────────┘
              │                           │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │  Instance Segmentation    │
              │  Head (UNet V2)           │
              │  - Attention Modules      │
              │  - Residual Blocks        │
              │  - Distance-Aware Loss    │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   3-Class Output Logits   │
              │   [N, 3, mask_h, mask_w]  │
              │   Classes:                │
              │   0: Background           │
              │   1: Target Instance      │
              │   2: Non-target Instances │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   Post-Processing         │
              │   (Optional)              │
              │   - Mask Dilation         │
              │   - Edge Smoothing        │
              └───────────────────────────┘
```

## 学習パイプライン

### 知識蒸留パイプライン
1. **教師モデル学習**: B7アーキテクチャを高精度まで学習
2. **温度進行**: 段階的な温度低下（10→1）
3. **生徒の学習**: 特徴とロジットマッチングでB0/B1に蒸留
4. **ファインチューニング**: ターゲットデータセットでのオプションの直接学習

### 学習ステージ
1. **ステージ1: UNet事前学習**
   - COCOデータセットでのバイナリ人物セグメンテーション
   - 事前学習後、すべての後続ステージで凍結

2. **ステージ2: 知識蒸留**
   - 教師モデル（B7）がソフトターゲットを提供
   - スムーズな知識転送のための温度進行
   - 複数のデコーダレベルでの特徴マッチング

3. **ステージ3: インスタンスセグメンテーション学習**
   - 3クラス出力でのROIベース学習
   - インスタンス分離のための距離認識損失
   - 境界リファインメントのための補助タスク

## リファインメント機構

### 階層型リファインメントプロセス
1. **粗いセグメンテーション**: 事前学習済みUNetが初期バイナリマスクを提供
2. **ROI抽出**: 検出された人物周辺の領域を抽出
3. **特徴強化**: EfficientNetエンコーダでROIを処理
4. **インスタンスリファインメント**:
   - アテンションゲート付きリファインメントを適用
   - 背景抑制のための事前情報としてバイナリマスクを使用
   - 距離変換による重複インスタンスの分離

### 主要なリファインメント技術
- **アテンションゲーティング**: 人物境界への処理の集中
- **距離変換**: より良い分離のための空間関係のエンコード
- **輪郭検出**: エッジ保存のための補助タスク
- **分離認識重み付け**: より明確な境界のための非ターゲットクラスのブースト

## データセット構造

### ディレクトリレイアウト
```
data/
├── annotations/
│   ├── instances_train2017_person_only_no_crowd.json  # 完全な学習セット
│   ├── instances_val2017_person_only_no_crowd.json    # 完全な検証セット
│   ├── instances_train2017_person_only_no_crowd_100imgs.json  # 開発サブセット
│   └── instances_val2017_person_only_no_crowd_100imgs.json    # 開発サブセット
├── images/
│   ├── train2017/  # COCO学習画像
│   └── val2017/    # COCO検証画像
└── pretrained/
    ├── best_model_b0_*.pth  # 事前学習済みB0モデル
    ├── best_model_b1_*.pth  # 事前学習済みB1モデル
    └── best_model_b7_*.pth  # 事前学習済みB7モデル
```

### アノテーション形式
- **形式**: COCO JSON形式
- **カテゴリ**: 人物のみ（群衆アノテーションなし）
- **内容**: バウンディングボックスとセグメンテーションポリゴン
- **フィルタリング**: よりクリーンな学習のために群衆インスタンスを削除

### データセット統計
- **完全データセット**: 約64K学習画像、約2.7K検証画像
- **開発サブセット**: 100、500画像バージョン
- **クラス分布**:
  - 背景: 約53.8%のピクセル
  - ターゲットインスタンス: 約33.3%のピクセル
  - 非ターゲットインスタンス: 約12.9%のピクセル

## 環境セットアップ

### 前提条件
- Python 3.10
- CUDA 11.8+（GPUサポート用）
- uvパッケージマネージャー

### uvを使用したインストール

```bash
# uvがインストールされていない場合はインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境を作成
uv venv

# 環境をアクティベート
source .venv/bin/activate  # Linux/Mac の場合
# または
.venv\Scripts\activate  # Windows の場合

# 依存関係をインストール
uv pip install -r pyproject.toml

# 開発用依存関係をインストール（オプション）
uv pip install -e ".[dev]"
```

### インストールの確認
```bash
# PyTorchとCUDAをチェック
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# ONNX Runtimeをチェック
uv run python -c "import onnxruntime as ort; print(f'ONNX Runtime: {ort.__version__}')"
```

## UNet蒸留コマンド

### 蒸留設定ファイル
- `rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog`: B7→B0蒸留
- `rgb_hierarchical_unet_v2_distillation_b1_from_b7_temp_prog`: B7→B1蒸留
- `rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog`: B7自己蒸留

### 基本的な蒸留学習
```bash
# 温度進行を使用したB7からB0への蒸留
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--epochs 100 \
--batch_size 16

# B7からB1への蒸留
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b1_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--epochs 100 \
--batch_size 12
```

### 高度な蒸留オプション
```bash
# カスタム温度スケジュールを使用
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--initial_temperature 10.0 \
--final_temperature 1.0 \
--temperature_decay_epochs 50 \
--epochs 100

# チェックポイントから再開
uv run python train_distillation_staged.py \
--teacher_config rgb_hierarchical_unet_v2_distillation_b7_from_b7_temp_prog \
--student_config rgb_hierarchical_unet_v2_distillation_b0_from_b7_temp_prog \
--teacher_checkpoint ext_extractor/best_model_b7_0.9009.pth \
--resume checkpoints/distillation_epoch_050.pth \
--epochs 100
```

## ROIベース階層型学習

### 標準設定ファイル
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B0`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B1`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B7`

### 拡張設定ファイル
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B0_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B1_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B7_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B7_enhanced`
- `rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r128x96m256x192_disttrans_contdet_baware_from_B7_enhanced`

### 基本的な学習コマンド

```bash
# 標準ROIサイズでB0モデルを学習（開発データセット）
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--epochs 10 \
--batch_size 8

# 拡張ROIサイズでB1モデルを学習（完全データセット）
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r80x60m160x120_disttrans_contdet_baware_from_B1_enhanced \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--epochs 100 \
--batch_size 6

# ウルトラROIサイズでB7モデルを学習
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r128x96m256x192_disttrans_contdet_baware_from_B7_enhanced \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--epochs 100 \
--batch_size 4
```

### 高度な学習オプション

```bash
# チェックポイントから学習を再開
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--resume experiments/*/checkpoints/checkpoint_epoch_0050_640x640_0750.pth \
--epochs 100

# より小さい学習率でファインチューニング
uv run python train_advanced.py \
--config rgb_hierarchical_unet_v2_fullimage_pretrained_peopleseg_r64x48m128x96_disttrans_contdet_baware_from_B0 \
--pretrained_checkpoint experiments/*/checkpoints/best_model_*.pth \
--learning_rate 1e-5 \
--epochs 20
```

### 検証コマンド

```bash
# 単一のチェックポイントを検証
uv run python validate_advanced.py \
experiments/*/checkpoints/best_model_epoch_*_640x640_*.pth \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--batch_size 16

# 複数のチェックポイントを検証
uv run python validate_advanced.py \
"experiments/*/checkpoints/best_model*.pth" \
--multiple \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json

# 可視化付き検証
uv run python validate_advanced.py \
experiments/*/checkpoints/best_model_*.pth \
--visualize \
--num_visualize 20 \
--output_dir validation_results
```

## ONNXエクスポート

### エクスポートスクリプト
- `export_peopleseg_onnx.py`: 事前学習済みUNetモデルのエクスポート
- `export_hierarchical_instance_peopleseg_onnx.py`: 完全な階層型モデルのエクスポート
- `export_bilateral_filter.py`: バイラテラルフィルタ後処理のエクスポート
- `export_edge_smoothing_onnx.py`: エッジスムージングモジュールのエクスポート

### 基本的なエクスポートコマンド

```bash
# B0モデルをONNXにエクスポート
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b0_*.pth \
--output models/b0_model.onnx \
--image_size 640,640

# 1ピクセル拡張付きでB1モデルをエクスポート
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b1_*.pth \
--output models/b1_model_dil2.onnx \
--image_size 640,640 \
--dilation_pixels 1

# カスタムROIサイズでB7モデルをエクスポート
uv run python export_hierarchical_instance_peopleseg_onnx.py \
experiments/*/checkpoints/best_model_b7_*.pth \
--output models/b7_model_ultra.onnx \
--image_size 1024,1024 \
--roi_size 128,96 \
--mask_size 256,192
```

### 後処理モジュールのエクスポート

```bash
# エッジスムージングモジュールをエクスポート
uv run python export_edge_smoothing_onnx.py \
--output models/edge_smoothing.onnx \
--threshold 0.5 \
--blur_strength 3.0

# バイラテラルフィルタをエクスポート
uv run python export_bilateral_filter.py \
--output models/bilateral_filter.onnx \
--d 9 \
--sigma_color 75 \
--sigma_space 75
```

### ONNX最適化

```bash
# onnxsimでONNXモデルを最適化
uv run python -m onnxsim models/b0_model.onnx models/b0_model_opt.onnx

# 最適化されたモデルを検証
uv run python -c "import onnx; model = onnx.load('models/b0_model_opt.onnx'); onnx.checker.check_model(model); print('モデルは有効です')"
```

## 推論テスト

### テストスクリプト: `test_hierarchical_instance_peopleseg_onnx.py`

### 基本的なテスト

```bash
# 検証画像でONNXモデルをテスト
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd_100imgs.json \
--images_dir data/images/val2017 \
--num_images 5 \
--output_dir test_outputs

# CUDAプロバイダーでテスト
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b1_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--provider cuda \
--num_images 10 \
--output_dir test_outputs_cuda
```

### 高度なテストオプション

```bash
# バイナリマスク可視化でテスト（緑色オーバーレイ）
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 20 \
--binary_mode \
--alpha 0.7 \
--output_dir test_binary_masks

# カスタムスコア閾値でテスト
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b7_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 15 \
--score_threshold 0.5 \
--save_masks \
--output_dir test_high_confidence

# バッチ処理テスト
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 100 \
--batch_size 8 \
--output_dir batch_test_outputs
```

### パフォーマンスベンチマーク

```bash
# 推論速度をベンチマーク
uv run python test_hierarchical_instance_peopleseg_onnx.py \
--onnx models/b0_model_opt.onnx \
--annotations data/annotations/instances_val2017_person_only_no_crowd.json \
--num_images 50 \
--benchmark \
--provider cuda

# 異なるモデルバリアントを比較
for model in b0 b1 b7; do
  echo "Testing $model model..."
  uv run python test_hierarchical_instance_peopleseg_onnx.py \
    --onnx models/${model}_model_opt.onnx \
    --annotations data/annotations/instances_val2017_person_only_no_crowd_100imgs.json \
    --num_images 20 \
    --benchmark \
    --output_dir benchmark_${model}
done
```

## ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は以下を参照してください：

```
MIT License

Copyright (c) 2025 Katsuya Hyodo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 引用と謝辞

このプロジェクトは、コンピュータビジョンコミュニティにおけるいくつかの優れた作品に基づいています：

### People Segmentation
Vladimir Iglovikov (Ternaus)による人物セグメンテーションの作業に深く感謝します：
- リポジトリ: [https://github.com/ternaus/people_segmentation](https://github.com/ternaus/people_segmentation)
- 論文: "TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation"

### EfficientNet
```bibtex
@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

### COCOデータセット
```bibtex
@inproceedings{lin2014microsoft,
  title={Microsoft COCO: Common Objects in Context},
  author={Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and Hays, James and Perona, Pietro and Ramanan, Deva and Dollár, Piotr and Zitnick, C Lawrence},
  booktitle={European Conference on Computer Vision},
  pages={740--755},
  year={2014},
  organization={Springer}
}
```

### U-Netアーキテクチャ
```bibtex
@inproceedings{ronneberger2015u,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```

### 知識蒸留
```bibtex
@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year={2015}
}
```

### 特別な感謝
- 優れたディープラーニングフレームワークを提供するPyTorchチーム
- クロスプラットフォームモデルデプロイメントツールを提供するONNXコミュニティ
- 強力な拡張パイプラインを提供するAlbumentationsチーム
- 事前学習済みエンコーダを提供するSegmentation Models PyTorchコントリビューター
