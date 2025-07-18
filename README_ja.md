# 人物検出のためのROIベース・インスタンスセグメンテーション

[English](README.md) | [日本語](README_ja.md)

本リポジトリは、YOLOv9特徴量とカスタムセグメンテーションデコーダを使用した軽量なROIベースの人物検出用インスタンスセグメンテーションモデルを実装しています。

## 概要

<img width="1500" height="1000" alt="architecture_diagram" src="https://github.com/user-attachments/assets/4e235e0f-205b-44e2-9126-e5e954fee82e" />

本モデルは3クラスのセグメンテーションアプローチを採用しています：
- **クラス0**: 背景
- **クラス1**: ターゲットマスク（ROI内の主要インスタンス）
- **クラス2**: 非ターゲットマスク（ROI内の他のインスタンス）

この定式化により、混雑したシーンでの複数の人物インスタンスをより良く区別できるようになります。

## クイックスタート

1. **環境設定**
   ```bash
   uv sync
   
   # TensorRTサポートのためLD_LIBRARY_PATHを設定:
   # オプション1: アクティベーションスクリプトを使用
   source activate.sh
   
   # オプション2: direnvがインストールされている場合
   direnv allow
   
   # オプション3: 手動でエクスポート
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PWD}/.venv/lib/python3.10/site-packages/tensorrt_libs
   ```

2. **パイプラインテスト**
   ```bash
   uv run python test_pipeline.py
   ```

3. **モデル訓練（最小データセット）**
   ```bash
   uv run python main.py --epochs 10
   ```

## モデルアーキテクチャ

### 概要

モデルは3つの主要コンポーネントから構成されています：

1. **特徴抽出器**: 事前学習済みYOLOv9モデル（ONNX形式）
   - 入力: 640×640×3 RGBイメージ
   - 出力: 1024×80×80 中間特徴量
   - 特徴ストライド: 8 (640÷80)

2. **ROIセグメンテーションヘッド**: カスタム軽量デコーダ
   - 入力: YOLOv9特徴量 + ROI座標
   - 処理: DynamicRoIAlign → 畳み込みブロック → アップサンプリング
   - 出力: ROIごとに56×56×3のセグメンテーションマスク

3. **損失関数**: 結合重み付き損失
   - 全クラスに対する重み付きクロスエントロピー
   - ターゲットクラス（クラス1）専用のDice損失
   - クラス不均衡を処理する分離認識重み

### アーキテクチャの詳細

```
YOLOv9特徴量 (1024×80×80)
       ↓
DynamicRoIAlign (→ 1024×28×28)  ← 改善：2倍大きなROI
       ↓
Conv 1×1 + LayerNorm + ReLU (→ 256×28×28)
       ↓
残差ブロック1 (→ 256×28×28)
       ↓
残差ブロック2 (→ 256×28×28)
       ↓
ConvTranspose + Refine (→ 256×56×56)
       ↓
ConvTranspose + Refine (→ 128×112×112)
       ↓
マルチスケール融合 (→ 128×56×56)
       ↓
Conv 1×1 (→ 3×56×56)
```

### ROI処理の詳細

#### DynamicRoIAlign操作

DynamicRoIAlignモジュールは、可変サイズのROIから固定サイズの特徴量を抽出します：

1. **入力座標**: ROIは元画像空間（640×640）で`[batch_idx, x1, y1, x2, y2]`として指定されます

2. **座標変換**:
   - 画像空間座標は`spatial_scale = 1/8`を使用して特徴空間にスケーリングされます
   - 例: ROI [0, 160, 160, 320, 320] → 特徴空間 [0, 20, 20, 40, 40]

3. **28×28グリッドサンプリング**:
   - ROI領域は28×28のグリッドに分割されます
   - 各グリッドポイントはROI内のサンプリング位置を表します
   - 各ポイントで特徴量を抽出するために双線形補間が使用されます

#### グリッドの解釈

28×28の出力グリッドは、改善された解像度で各ROIの空間的に正規化された表現を提供します：

```
人物検出のための28×28グリッドマッピング：
┌─────────────────────────────┐
│ 行0-9:    頭/肩              │  各セル(i,j)は以下を表す：
│ 行10-18:  胴体/腕            │  - (0,0): ROIの左上
│ 行19-27:  脚/下半身           │  - (0,27): ROIの右上
├─────────────────────────────┤  - (27,0): ROIの左下
│ 列0-9:    左側               │  - (27,27): ROIの右下
│ 列10-18:  中央               │  - (13,13): ROIの中心
│ 列19-27:  右側               │
└─────────────────────────────┘
```

#### 非正方形ROIの処理

非正方形ROI（例：縦長の人物バウンディングボックス）の場合：
- 水平サンプリング: 幅 ÷ 28 ピクセル/サンプル
- 垂直サンプリング: 高さ ÷ 28 ピクセル/サンプル
- 28×28グリッドは一貫した出力サイズを維持しながらROIのアスペクト比に適応します

200×100 ROIの例：
```
元のROI (200×100ピクセル)      →    正規化グリッド (28×28)
┌────────────────────┐                 ┌─────────────┐
│ 疎な垂直            │                 │ ● ● ● ● ● ● │
│ サンプリング(~3.5px) │        →        │ ● ● ● ● ● ● │
│ 密な水平            │                 │ ● ● ● ● ● ● │
│ サンプリング(~7px)   │                 └─────────────┘
└────────────────────┘
```

### 主要な設計判断

1. **BatchNorm/InstanceNormの代わりにLayerNorm**: より良いONNX互換性と安定した推論

2. **強化された28×28 ROIサイズ**: より良い詳細キャプチャのための改善された空間解像度

3. **3クラス定式化**: 非ターゲットインスタンスを明示的にモデル化することで、混雑したシーンでの分離が改善

4. **DynamicRoIAlign**: opset 16でのより良いONNXエクスポートのためのカスタム実装

## マスク出力の解釈とオーバーレイ

### 出力の理解

モデルは`[num_rois, 3, 56, 56]`の形状でマスクを出力します：
- `num_rois`: 処理されたROIの数
- `3`: 3つのクラス（背景、ターゲット、非ターゲット）
- `56×56`: 固定マスク解像度

### 元画像へのマスクのオーバーレイ手順

#### 1. クラス予測の抽出

```python
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

# モデル出力: masks [N, 3, 56, 56]
# softmaxを適用して確率を取得
mask_probs = F.softmax(masks, dim=1)  # [N, 3, 56, 56]

# ターゲットマスク（クラス1）の確率を取得
target_masks = mask_probs[:, 1, :, :]  # [N, 56, 56]

# オプション：ピクセルごとの予測クラスを取得
predicted_classes = torch.argmax(mask_probs, dim=1)  # [N, 56, 56]
```

#### 2. マスクをROI寸法にリサイズ

各56×56マスクは対応するROIサイズに合わせてリサイズする必要があります：

```python
# ROIフォーマット: [batch_idx, x1, y1, x2, y2]
resized_masks = []

for i, roi in enumerate(rois):
    x1, y1, x2, y2 = roi[1:].int()
    roi_width = x2 - x1
    roi_height = y2 - y1

    # 56×56マスクをROIサイズにリサイズ
    mask = target_masks[i].unsqueeze(0).unsqueeze(0)  # [1, 1, 56, 56]
    resized_mask = F.interpolate(
        mask,
        size=(roi_height, roi_width),
        mode='bilinear',
        align_corners=False
    )
    resized_masks.append(resized_mask.squeeze())
```

#### 3. フル画像にマスクを配置

フル解像度のマスクを作成し、各ROIマスクを正しい位置に配置します：

```python
# フル画像用の空のマスクを作成
full_mask = torch.zeros((image_height, image_width))

for i, roi in enumerate(rois):
    x1, y1, x2, y2 = roi[1:].int()

    # リサイズされたマスクを正しい位置に配置
    full_mask[y1:y2, x1:x2] = resized_masks[i]
```

#### 4. 閾値適用とオーバーレイ

```python
# 閾値を適用してバイナリマスクを取得
threshold = 0.5
binary_mask = (full_mask > threshold).float()

# 可視化のためにnumpyに変換
mask_np = binary_mask.cpu().numpy()
image_np = np.array(image)  # imageがPIL Imageであると仮定

# 色付きオーバーレイを作成
overlay = image_np.copy()
mask_color = [255, 0, 0]  # ターゲットインスタンスには赤

# 透明度付きでマスクを適用
alpha = 0.5
overlay[mask_np > 0] = (
    alpha * np.array(mask_color) +
    (1 - alpha) * image_np[mask_np > 0]
).astype(np.uint8)
```

### 完全な関数例

```python
def overlay_masks_on_image(image, masks, rois, threshold=0.5, alpha=0.5):
    """
    元画像にセグメンテーションマスクをオーバーレイ。

    引数:
        image: PIL画像またはnumpy配列 (H, W, 3)
        masks: モデル出力テンソル [N, 3, 56, 56]
        rois: ROI座標テンソル [N, 5]
        threshold: バイナリマスクの信頼度閾値
        alpha: オーバーレイの透明度

    戻り値:
        マスクがオーバーレイされたPIL画像
    """
    # 画像をnumpyに変換
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    h, w = image_np.shape[:2]

    # マスク確率を取得
    mask_probs = F.softmax(masks, dim=1)

    # 各クラスのフル解像度マスクを作成
    full_masks = {
        'background': torch.zeros((h, w)),
        'target': torch.zeros((h, w)),
        'non_target': torch.zeros((h, w))
    }

    # 各ROIを処理
    for i, roi in enumerate(rois):
        _, x1, y1, x2, y2 = roi.int()
        roi_w = x2 - x1
        roi_h = y2 - y1

        # このROIのマスクをリサイズ
        roi_masks = F.interpolate(
            mask_probs[i].unsqueeze(0),  # [1, 3, 56, 56]
            size=(roi_h, roi_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [3, roi_h, roi_w]

        # フル画像に配置（重複ROIには最大値を使用）
        full_masks['background'][y1:y2, x1:x2] = torch.maximum(
            full_masks['background'][y1:y2, x1:x2],
            roi_masks[0]
        )
        full_masks['target'][y1:y2, x1:x2] = torch.maximum(
            full_masks['target'][y1:y2, x1:x2],
            roi_masks[1]
        )
        full_masks['non_target'][y1:y2, x1:x2] = torch.maximum(
            full_masks['non_target'][y1:y2, x1:x2],
            roi_masks[2]
        )

    # 色付きオーバーレイを作成
    overlay = image_np.copy()

    # ターゲットマスクを適用（赤）
    target_mask = (full_masks['target'] > threshold).cpu().numpy()
    overlay[target_mask] = (
        alpha * np.array([255, 0, 0]) +
        (1 - alpha) * image_np[target_mask]
    ).astype(np.uint8)

    # 非ターゲットマスクを適用（青）
    non_target_mask = (full_masks['non_target'] > threshold).cpu().numpy()
    overlay[non_target_mask] = (
        alpha * np.array([0, 0, 255]) +
        (1 - alpha) * image_np[non_target_mask]
    ).astype(np.uint8)

    return Image.fromarray(overlay)
```

### 複数の重複ROIの処理

複数のROIが重複する場合、マスクを結合する方法を決定する必要があります：

1. **最大値**: 最高の信頼度値を使用（推奨）
   ```python
   full_mask[y1:y2, x1:x2] = torch.maximum(
       full_mask[y1:y2, x1:x2],
       resized_mask
   )
   ```

2. **平均**: 重複する予測を平均化
   ```python
   full_mask[y1:y2, x1:x2] = (
       full_mask[y1:y2, x1:x2] + resized_mask
   ) / 2
   ```

3. **優先度ベース**: 信頼度の順序でROIを処理
   ```python
   # まず信頼度でROIをソート
   roi_confidences = get_roi_confidences()  # 検出モデルから
   sorted_indices = torch.argsort(roi_confidences, descending=True)

   # 順番に処理
   for idx in sorted_indices:
       # より高い信頼度のROIがより低いものを上書き
       full_mask[y1:y2, x1:x2] = resized_masks[idx]
   ```

### 重要な考慮事項

1. **座標系**: ROI座標が画像座標系と一致していることを確認（0インデックス、終了座標は排他的）

2. **補間モード**: スムーズなマスクエッジのために`align_corners=False`で`bilinear`を使用

3. **メモリ効率**: 大規模バッチ処理の場合、メモリ問題を避けるためROIをチャンクで処理

4. **エッジアーティファクト**: よりスムーズな遷移のため、マスクエッジにガウシアンブラーの適用を検討：
   ```python
   from scipy.ndimage import gaussian_filter
   smoothed_mask = gaussian_filter(mask_np, sigma=1.0)
   ```

## プロジェクト構造

```
├── src/human_edge_detection/
│   ├── dataset.py          # COCOデータセットローダー
│   ├── feature_extractor.py # YOLO特徴抽出
│   ├── model.py            # セグメンテーションモデル
│   ├── losses.py           # 損失関数
│   ├── train.py            # 訓練パイプライン
│   ├── visualize.py        # 検証時の可視化
│   └── export_onnx.py      # ONNXエクスポート
├── data/
│   ├── annotations/        # COCO形式のアノテーション
│   └── images/            # 訓練/検証画像
├── ext_extractor/         # YOLOv9 ONNXモデル
├── main.py               # メイン訓練スクリプト
└── test_pipeline.py      # コンポーネント検証

```

## 訓練

### 訓練機能

- **訓練の再開**: 任意のチェックポイントから訓練を継続
- **柔軟なエポック制御**: `--resume_epochs`で再開時に追加エポックを指定
- **動的プログレスバー**: ターミナル幅に自動調整
- **チェックポイント管理**: 指定間隔でモデルを保存
- **ベストモデル追跡**: 検証mIoUに基づいて最良のモデルを自動保存

### 推奨訓練パラメータ

広範な実験に基づく、異なる訓練シナリオの推奨パラメータ：

#### クイックテスト実行（100画像）
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd_100.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd_100.json \
--epochs 10 \
--batch_size 8 \
--lr 1e-4 \
--validate_every 1 \
--save_every 1
```

#### 小規模データセット訓練（500画像）
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd_500.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd_500.json \
--epochs 50 \
--batch_size 8 \
--lr 5e-4 \
--scheduler cosine \
--min_lr 1e-6 \
--validate_every 2 \
--save_every 5
```

#### フルデータセット訓練（推奨）
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--data_stats data_analyze_full.json \
--epochs 100 \
--batch_size 16 \
--lr 1e-3 \
--optimizer adamw \
--weight_decay 1e-4 \
--scheduler cosine \
--min_lr 1e-6 \
--gradient_clip 5.0 \
--num_workers 8 \
--validate_every 1 \
--save_every 1
```

#### 高性能訓練（マルチGPU）
```bash
uv run python main.py \
--train_ann data/annotations/instances_train2017_person_only_no_crowd.json \
--val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
--data_stats data_analyze_full.json \
--epochs 150 \
--batch_size 32 \
--lr 2e-3 \
--optimizer adamw \
--weight_decay 5e-5 \
--scheduler cosine \
--min_lr 1e-7 \
--gradient_clip 10.0 \
--num_workers 16 \
--ce_weight 1.0 \
--dice_weight 2.0 \
--validate_every 1 \
--save_every 1
```

#### チェックポイントからのファインチューニング
```bash
# 元のエポック数までチェックポイントから訓練を再開
uv run python main.py \
--resume checkpoints/best_model.pth \
--epochs 50 \
--batch_size 8 \
--lr 1e-4 \
--scheduler cosine \
--min_lr 1e-7 \
--dice_weight 3.0 \
--validate_every 1 \
--save_every 1

# チェックポイントから追加エポックで訓練
uv run python main.py \
--resume checkpoints/checkpoint_epoch_0010_640x640_0850.pth \
--resume_epochs 20 \
--batch_size 8 \
--lr 5e-5 \
--scheduler cosine \
--min_lr 1e-7 \
--dice_weight 3.0
```

### パラメータ選択ガイドライン

#### 学習率
- **小規模データセット（100-500画像）**: 1e-4 〜 5e-4
- **中規模データセット（500-5000画像）**: 5e-4 〜 1e-3
- **大規模データセット（5000画像以上）**: 1e-3 〜 2e-3
- **ファインチューニング**: 1e-5 〜 1e-4

#### バッチサイズ
- **GPUメモリ < 8GB**: 4-8
- **GPUメモリ 8-16GB**: 8-16
- **GPUメモリ > 16GB**: 16-32
- より大きなバッチサイズは一般により安定した訓練につながります

#### オプティマイザの選択
- **AdamW**（デフォルト）: ほとんどの場合に最適、良好な収束
- **Adam**: わずかに高速だが、正則化が少ない
- **SGD**: より安定だが収束が遅い、ファインチューニングに適している

#### スケジューラー戦略
- **Cosine**（推奨）: スムーズな減衰、ほとんどの場合でうまく機能
- **Step**: 学習が低下すべき特定のエポックがわかっている場合に良い
- **Exponential**: 連続的な減衰、長期訓練に適している
- **None**: 固定学習率、短い実験のみ

#### 損失の重み
- **CE重み**: 1.0（デフォルト）- 分類精度を制御
- **Dice重み**: 1.0-3.0 - より高い値はマスク品質を改善
  - バランスの取れた訓練には1.0から開始
  - マスクが十分シャープでない場合は2.0-3.0に増やす
  - マスク境界を改善するファインチューニングには3.0以上を使用

#### 勾配クリッピング
- **0**（無効）: 安定したデータセットの場合
- **5.0**: ほとんどの訓練に推奨
- **10.0**: より大きなバッチサイズまたは不安定な勾配の場合
- **1.0**: 非常に不安定な訓練の場合（まれ）

#### データ拡張（組み込み）
データセットは自動的に以下を適用します：
- ランダムな水平反転
- ランダムなわずかなスケーリング
- 色ジッター（明度、コントラスト、彩度）

### メモリ最適化

CUDA out of memoryエラーが発生した場合：

1. **バッチサイズを削減**: バッチサイズを半分にする
2. **勾配累積を有効にする**（コード修正が必要）：
   ```bash
   --batch_size 4 --gradient_accumulation 4  # 実効バッチサイズ16
   ```
3. **ワーカー数を削減**: `--num_workers 2`
4. **混合精度訓練を使用**（コード修正が必要）

### 訓練のモニタリング

訓練中に以下のメトリクスをモニタリング：
- **訓練損失**: 着実に減少すべき
- **検証mIoU**: 主要メトリクス、増加すべき
- **学習率**: 期待通りに減衰しているか確認
- **CE vs Dice損失**: 両方とも減少すべきだが、異なる速度で

TensorBoardを使用してモニタリング：
```bash
tensorboard --logdir logs
```

### 一般的な問題と解決策

1. **損失がNaNになる**
   - 学習率を下げる
   - 勾配クリッピングを有効にする
   - データセット内の破損した画像をチェック

2. **検証mIoUが改善しない**
   - 学習率を下げる
   - Dice重みを増やす
   - モデルが過学習していないかチェック（訓練損失 << 検証損失）

3. **訓練が遅すぎる**
   - GPUメモリが許せばバッチサイズを増やす
   - より多くのワーカーを使用: `--num_workers 8`
   - データがHDDではなくSSD上にあることを確認

4. **マスクがぼやけすぎる**
   - Dice重みを増やす: `--dice_weight 3.0`
   - より多くのエポックで訓練
   - 細部のために学習率を下げる

### 高度な訓練戦略

#### プログレッシブ訓練
小さなデータセットから始めて、次に拡張：
```bash
# ステージ1: 100画像、10エポック
uv run python main.py --epochs 10 [... その他のパラメータ]

# ステージ2: 500画像、ステージ1から再開
uv run python main.py --resume checkpoints/best_model.pth \
  --train_ann [..._500.json] --epochs 30

# ステージ3: フルデータセット
uv run python main.py --resume checkpoints/best_model.pth \
  --train_ann [..._full.json] --epochs 100
```

#### アンサンブル訓練
異なるシードで複数のモデルを訓練：
```bash
for seed in 42 123 456; do
  uv run python main.py --seed $seed \
    --checkpoint_dir checkpoints/seed_$seed \
    [... その他のパラメータ]
done
```

追加の訓練手順とコマンド例については`CLAUDE.md`を参照してください。

## 検証

### スタンドアロン検証

`validate.py`スクリプトを使用して、完全な訓練パイプラインを実行せずに訓練済みチェックポイントで検証を実行できます。

#### 単一チェックポイントの検証

```bash
# ベストモデルの検証
uv run python validate.py checkpoints/best_model.pth

# カスタム設定での特定チェックポイントの検証
uv run python validate.py checkpoints/checkpoint_epoch_0050_640x640_0850.pth \
  --val_ann data/annotations/instances_val2017_person_only_no_crowd.json \
  --data_stats data_analyze_full.json \
  --batch_size 16 \
  --num_workers 8
```

#### 複数チェックポイントの検証

```bash
# ディレクトリ内のすべてのチェックポイントを検証
uv run python validate.py "checkpoints/*.pth" --multiple

# パターンに一致するチェックポイントを検証
uv run python validate.py "checkpoints/checkpoint_epoch_00*.pth" --multiple \
  --no_visualization  # より高速な検証のために可視化生成をスキップ
```

#### コマンドライン引数

- `checkpoint`: チェックポイントファイルまたはglobパターンへのパス（--multipleと共に）
- `--val_ann`: 検証アノテーションファイル（デフォルト：100画像サブセット）
- `--val_img_dir`: 検証画像ディレクトリ（デフォルト：data/images/val2017）
- `--onnx_model`: YOLO ONNXモデルパス
- `--data_stats`: クラス重みのためのデータ統計ファイル
- `--batch_size`: 検証のバッチサイズ（デフォルト：8）
- `--num_workers`: データローダーワーカー数（デフォルト：4）
- `--device`: 使用するデバイス - cuda/cpu（デフォルト：cuda）
- `--no_visualization`: 可視化画像の生成をスキップ
- `--val_output_dir`: 可視化の出力ディレクトリ
- `--multiple`: globパターンを使用した複数チェックポイントの検証を有効にする

#### 出力

検証スクリプトは以下を提供します：
- 各チェックポイントの詳細メトリクス（損失、CE損失、Dice損失、mIoU）
- 複数チェックポイントを検証する際の比較表
- オプションの可視化画像（訓練中と同じ）
- mIoUに基づくベストチェックポイントの識別

### 訓練中の検証

検証は`--validate_every`パラメータに基づいて訓練中に自動的に実行されます。訓練なしで検証のみを実行するには：

```bash
# test_onlyフラグでmain.pyを使用
uv run python main.py --test_only --resume checkpoints/best_model.pth
```

## GPUアクセラレーションとTensorRTサポート

### インストール

プロジェクトはONNX Runtime GPUとTensorRTを介したGPUアクセラレーションをサポートしています：

```bash
# すでに環境に含まれています
uv add onnxruntime-gpu tensorrt
```

### パフォーマンス比較

セグメンテーションモデルでの推論テストに基づく：

| プロバイダー | 平均推論時間 | 高速化 |
|----------|----------------------|---------|
| CPU      | 74.57 ms            | 1.0x    |
| CUDA     | 10.07 ms            | 7.4x    |
| TensorRT（初回実行） | 19.6 s* | - |
| **TensorRT（キャッシュ済み）** | **4.04 ms** | **18.4x** |

*初回実行にはエンジン構築時間が含まれます。後続の実行はキャッシュされたエンジンを使用します。

### 使用方法

システムは利用可能な最適なプロバイダーを自動的に検出して使用します：

1. **TensorRT**（最速）- 利用可能な場合、エンジンキャッシングとFP16最適化付き
2. **CUDA**（高速）- CUDAが利用可能な場合
3. **CPU**（フォールバック）- 常に利用可能

**TensorRTの機能:**
- **エンジンキャッシング**: エンジンはONNXモデルと同じディレクトリにキャッシュされます
- **FP16最適化**: より良いパフォーマンスのための自動混合精度
- **初回実行**: エンジンの構築と最適化に時間がかかります
- **後続の実行**: キャッシュされたエンジンを使用した超高速推論

### プロバイダーの選択

必要に応じて手動でプロバイダーを指定できます：

```python
# 特定のプロバイダーでの特徴抽出
extractor = YOLOv9FeatureExtractor(
    'ext_extractor/yolov9_e_wholebody25_Nx3x640x640_featext_optimized.onnx',
    device='cuda',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)

# TensorRTを使用したONNX Runtimeセッション
import onnxruntime as ort
session = ort.InferenceSession(
    'test_model.onnx',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
)
```

### トラブルシューティング

TensorRTが動作しない場合：

1. **GPU互換性を確認**: GPUがTensorRTをサポートしていることを確認
2. **CUDAバージョン**: CUDAが適切にインストールされていることを確認
3. **メモリ**: TensorRTは最適化のために追加のGPUメモリが必要
4. **フォールバック**: システムは自動的にCUDAまたはCPUにフォールバック

## ライセンス

MITライセンス - 詳細はLICENSEファイルを参照してください。
