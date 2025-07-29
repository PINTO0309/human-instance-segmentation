# Binary Mask 精緻化手法

## 概要

このドキュメントでは、バイナリマスクの品質を向上させるための7つの精緻化手法について説明します。これらの手法は、特に境界の精度と滑らかさの改善に焦点を当てており、階層的セグメンテーションモデルのプラグイン可能なモジュールとして実装されています。

## 精緻化手法の詳細

### 1. Boundary Refinement Network (BRN)

**`use_boundary_refinement=True` で有効化**
マスクの境界部分を専門的に処理するサブネットワーク。エッジ検出とエッジ依存の精緻化を組み合わせて、境界領域を選択的に改善します。

```python
class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels=3, edge_channels=32):
        super().__init__()
        # エッジ検出と精緻化ネットワーク
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels, 3, padding=1),     # 3→32
            LayerNorm2d(edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, edge_channels, 3, padding=1),   # 32→32
            LayerNorm2d(edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, in_channels, 1)                 # 32→3
        )
        # 学習可能なブレンド重み（小さく初期化）
        self.blend_weight = nn.Parameter(torch.tensor(0.01))
        
        # 安定性のため小さく初期化
        for m in self.edge_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.1)

    def detect_edges(self, mask_logits):
        """Sobel風フィルタでエッジを検出"""
        probs = torch.softmax(mask_logits, dim=1)
        # 勾配計算
        dy = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        dx = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        # パディングとL2ノルム
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        edges = torch.sqrt(dy**2 + dx**2).mean(dim=1, keepdim=True)
        # 正規化（数値安定性考慮）
        edge_min, edge_max = edges.min(), edges.max()
        if edge_max - edge_min < 1e-6:
            edges = torch.zeros_like(edges)
        else:
            edges = (edges - edge_min) / (edge_max - edge_min + 1e-6)
        return edges

    def forward(self, mask_logits):
        # 1. エッジ検出
        edges = self.detect_edges(mask_logits)  # (B, 1, H, W)
        # 2. エッジ認識型の精緻化
        refined_edges = self.edge_conv(mask_logits)  # (B, 3, H, W)
        # 3. エッジ強度に基づいてブレンド（残差接続）
        refined = mask_logits + self.blend_weight * refined_edges * edges
        return refined
```

**動作の詳細：**
1. **エッジ検出**: Sobel風フィルタで境界領域を自動検出
   - 隣接ピクセル間の勾配を計算（dy, dx）
   - L2ノルムでエッジ強度を算出
   - 0-1の範囲に正規化

2. **エッジ依存の精緻化**: 検出されたエッジ領域に特化した調整を適用
   - 3層のConv2Dネットワークでエッジ領域を処理
   - LayerNorm2dで安定化

3. **適応的ブレンディング**: エッジ強度に基づいて精緻化の影響を調整
   - `refined = mask_logits + self.blend_weight * refined_edges * edges`
   - blend_weightは学習可能（0.01で初期化）

**主な特徴：**
- エッジ領域のみを選択的に処理（計算効率的）
- 残差接続により安定した学習と勾配流を実現
- 数値安定性のため小さく初期化（gain=0.1）

### 2. Active Contour Loss

**`use_active_contour_loss=True` で有効化**
境界の滑らかさを促進する損失関数：

```python
def active_contour_loss(pred_mask, smoothness_weight=0.01):
    """エネルギー最小化による境界の平滑化
    
    Active Contour Model (Snake) に基づく損失関数。
    境界の長さを最小化することで、ギザギザした境界を滑らかにする。
    """
    # ターゲットクラスの確率マップを取得
    if pred_mask.dim() == 4 and pred_mask.shape[1] > 1:
        pred_mask = pred_mask[:, 1:2, :, :]  # クラス1（ターゲット）

    # 勾配計算（隣接ピクセル間の差分）
    dy = pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]
    dx = pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]

    # 境界の長さ（L1ノルム）- 数値安定性のためクランプ
    dy_clamped = torch.clamp(torch.abs(dy), max=10.0)
    dx_clamped = torch.clamp(torch.abs(dx), max=10.0)
    boundary_length = torch.mean(dy_clamped) + torch.mean(dx_clamped)

    # 曲率（2次微分）- オプション
    ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    curvature = torch.mean(torch.abs(ddy)) + torch.mean(torch.abs(ddx))

    return boundary_length + smoothness_weight * curvature
```

**動作原理：**
- **境界長最小化**: マスクの境界線の総長を短くすることで、不要な凹凸を除去
- **曲率ペナルティ**: 急激な方向変化を抑制し、自然な曲線を促進
- **確率マップへの適用**: ソフトマックス後の連続値に対して作用

**効果：**
- ノイズによる細かいギザギザを除去
- 境界を滑らかな曲線に変換
- 過度の詳細を抑制し、主要な形状を保持

### 3. Progressive Upsampling

**`use_progressive_upsampling=True` で有効化**
段階的に解像度を上げることで、より滑らかな境界を生成：

```python
class ProgressiveUpsamplingDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.stages = nn.ModuleList([
            # 56x56 → 112x112
            nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels//2, 4, 2, 1),
                ResidualBlock(in_channels//2),
            ),
            # 112x112 → 224x224
            nn.Sequential(
                nn.ConvTranspose2d(in_channels//2, in_channels//4, 4, 2, 1),
                ResidualBlock(in_channels//4),
            ),
            # 224x224 → 56x56 (最終的にダウンサンプル)
            nn.Sequential(
                nn.Conv2d(in_channels//4, 3, 1),
                nn.AdaptiveAvgPool2d(56)
            )
        ])
```

### 4. Sub-pixel Convolution

**`use_subpixel_conv=True` で有効化**
ピクセルシャッフルを使用した高品質アップサンプリング：

```python
class SubPixelDecoder(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3 * upscale_factor**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
```

### 5. Contour Detection Branch

**`use_contour_detection=True` で有効化**
輪郭検出を明示的なタスクとして追加し、マルチタスク学習により境界精度を向上：

```python
class ContourDetectionBranch(nn.Module):
    def __init__(self, in_channels, contour_channels=64):
        super().__init__()
        self.contour_branch = nn.Sequential(
            nn.Conv2d(in_channels, contour_channels, 3, padding=1),  # 256→64
            LayerNorm2d(contour_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(contour_channels, contour_channels, 3, padding=1),  # 64→64
            LayerNorm2d(contour_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(contour_channels, 1, 1),  # 64→1
            nn.Sigmoid()  # 0-1の範囲に正規化
        )

    def forward(self, features):
        # 共有特徴量から輪郭を検出
        contours = self.contour_branch(features)
        return contours
```

**使用方法：**
```python
# Forward時
if self.use_contour_detection:
    contours = self.contour_branch(shared_features)
    # マスクサイズに合わせてリサイズ
    if contours.shape[2] != self.mask_size:
        contours = F.interpolate(contours, size=self.mask_size, 
                                mode='bilinear', align_corners=False)
    aux_outputs['contours'] = contours

# 損失計算時
if 'contours' in aux_outputs:
    contour_targets = self._generate_contour_targets(target)  # エッジ抽出
    contour_loss = F.binary_cross_entropy(aux_outputs['contours'], 
                                         contour_targets)
    total_loss += self.contour_loss_weight * contour_loss
```

**動作の詳細：**
1. **輪郭予測**: 共有特徴量から輪郭マップを直接予測
   - 256チャンネルの共有特徴を64チャンネルに圧縮
   - 2層のConv2Dで特徴抽出
   - 最終的に1チャンネルの輪郭マップを生成
   - Sigmoid活性化で0-1の範囲に正規化

2. **サイズ調整**: 必要に応じてmask_sizeに合わせてリサイズ
   - bilinear補間でスムーズな拡大縮小

3. **補助損失**: BCELossで輪郭予測の精度を最適化
   - ターゲットは`_generate_contour_targets`で生成
   - 勾配ベースのエッジ検出でグラウンドトゥルースを作成

**主な特徴：**
- 0-1の連続値として輪郭を表現（バイナリマスクより滑らか）
- メインのセグメンテーションタスクと同時学習
- 特徴表現の改善により全体的な性能向上
- 輪郭情報は後処理での境界精緻化にも活用可能

### 6. Distance Transform Prediction

**`use_distance_transform=True` で有効化**
物体境界からの距離を予測し、連続的な値でマスクを表現することで滑らかな境界を実現：

```python
class DistanceTransformDecoder(nn.Module):
    def __init__(self, in_channels, distance_channels=128):
        super().__init__()
        self.distance_head = nn.Sequential(
            nn.Conv2d(in_channels, distance_channels, 3, padding=1),  # 256→128
            LayerNorm2d(distance_channels),
            nn.ReLU(inplace=True),
            ResidualBlock(distance_channels),  # 残差ブロックで特徴抽出
            nn.Conv2d(distance_channels, 1, 1)  # 128→1 (距離マップ)
        )
        # 学習可能な閾値（マスク変換用）
        self.threshold = nn.Parameter(torch.tensor(0.3))

    def forward(self, features):
        # 境界からの距離を予測
        distance_map = self.distance_head(features)
        # 学習可能な閾値でマスクに変換（シャープな遷移）
        mask = torch.sigmoid((distance_map - self.threshold) * 10)
        return mask, distance_map
```

**使用方法：**
```python
# Forward時
if self.use_distance_transform:
    dist_mask, dist_map = self.distance_decoder(shared_features)
    # サイズ調整
    if dist_mask.shape[2] != self.mask_size:
        dist_mask = F.interpolate(dist_mask, size=self.mask_size, ...)
        dist_map = F.interpolate(dist_map, size=self.mask_size, ...)
    aux_outputs['distance_mask'] = dist_mask  # 変換後のマスク
    aux_outputs['distance_map'] = dist_map    # 生の距離マップ

# 損失計算時
if 'distance_map' in aux_outputs:
    distance_targets = self._generate_distance_targets(target)
    distance_loss = F.l1_loss(aux_outputs['distance_map'], distance_targets)
    # 安定性のためクランプ
    distance_loss = torch.clamp(distance_loss, max=10.0)
    total_loss += self.distance_loss_weight * distance_loss
```

**動作の詳細：**
1. **距離予測**: 境界からの距離を連続値として予測（回帰タスク）
   - 256→128チャンネルの特徴変換
   - ResidualBlockで深い特徴抽出
   - 最終的に1チャンネルの距離マップを生成

2. **マスク変換**: 学習可能な閾値でバイナリマスクに変換
   - `mask = torch.sigmoid((distance_map - self.threshold) * 10)`
   - thresholdは学習可能パラメータ（0.3で初期化）
   - ×10のスケーリングでシャープな遷移を実現

3. **二重出力**: 距離マップとマスクの両方を出力
   - `distance_map`: 生の距離予測値
   - `distance_mask`: 閾値変換後のマスク

**主な特徴：**
- 境界の不確実性を距離として表現
- 学習可能な閾値により最適な変換点を自動調整
- L1損失で距離予測の精度を最適化
- 後処理での閾値調整により、異なる要求に対応可能
- 距離情報により、境界付近の信頼度を定量化

### 7. Boundary-aware Loss

**`use_boundary_aware_loss=True` で有効化**
境界付近により大きな重みを与える損失関数：

```python
def boundary_aware_loss(pred, target, boundary_width=3, boundary_weight=2.0):
    """境界領域に焦点を当てた損失関数
    
    マスクの境界付近のピクセルにより大きな重みを与えることで、
    境界の精度を向上させる。
    """
    # ターゲットマスクから境界領域を検出
    # 膨張（dilation）操作で境界を特定
    kernel = torch.ones(1, 1, boundary_width, boundary_width).to(pred.device)
    
    # 各クラスの境界を検出
    boundaries = []
    for c in range(3):  # 3クラス分
        class_mask = (target == c).float().unsqueeze(1)
        dilated = F.conv2d(class_mask, kernel, padding=boundary_width//2)
        eroded = F.conv2d(class_mask, kernel, padding=boundary_width//2)
        # 膨張と収縮の差分が境界
        boundary = (dilated > 0) & (eroded < boundary_width**2)
        boundaries.append(boundary)
    
    # すべてのクラス境界を統合
    boundary_mask = torch.cat(boundaries, dim=1).any(dim=1)
    
    # 重み付きマップの作成
    weights = torch.ones_like(target, dtype=torch.float32)
    weights[boundary_mask] = boundary_weight  # 境界ピクセルの重みを増加
    
    # 重み付きクロスエントロピー損失
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    weighted_loss = ce_loss * weights
    
    return weighted_loss.mean()
```

**動作原理：**
- **境界検出**: 膨張・収縮演算により各クラスの境界ピクセルを特定
- **重み付け**: 境界ピクセルにより高い重み（デフォルト2.0倍）を適用
- **選択的学習**: モデルが境界付近の予測により注意を払うよう誘導

**効果：**
- 境界付近の分類精度が向上
- エッジのぼやけを減少
- 細かい構造の保持

## 統合アーキテクチャ

### RefinedHierarchicalSegmentationHead での統合

これらのモジュールは`RefinedHierarchicalSegmentationHead`内で協調動作します：

1. **共有特徴の活用**: 
   - すべてのモジュールが同じ共有特徴量（`shared_features`）を使用
   - 256チャンネルの中間特徴から各種の精緻化を実行

2. **独立した処理**: 
   - 各モジュールは独立して動作し、それぞれの補助出力を生成
   - Progressive UpsamplingとSub-pixel Convolutionは排他的使用

3. **統合された損失**: 
   - 各モジュールの損失が重み付けされて総損失に加算
   - すべての損失にクランプ（max=10.0）を適用して数値安定性を確保

### 実装時の処理フロー

```python
# 1. ベースの階層的セグメンテーション
base_masks, aux_outputs = self.base_head(features)

# 2. デコーダの置き換え（排他的）
if self.use_progressive_upsampling:
    refined_masks = self.progressive_decoder(shared_features, mask_size)
elif self.use_subpixel_conv:
    refined_masks = self.subpixel_decoder(shared_features)

# 3. 境界の精緻化（加算的）
if self.use_boundary_refinement:
    refined_masks = self.boundary_refiner(refined_masks)

# 4. 補助ブランチ（並列処理）
if self.use_contour_detection:
    aux_outputs['contours'] = self.contour_branch(shared_features)
if self.use_distance_transform:
    aux_outputs['distance_mask'], aux_outputs['distance_map'] = \
        self.distance_decoder(shared_features)
```

## 実装上の注意点

### 数値安定性の確保

1. **初期化戦略**:
   - Conv2D層: Xavier初期化with gain=0.1
   - 学習可能パラメータ: 小さな値（0.01〜0.3）で初期化
   - バイアス: ゼロ初期化

2. **損失のクランプ**:
   - すべての精緻化損失に`torch.clamp(loss, max=10.0)`を適用
   - 勾配爆発を防止

3. **正規化**:
   - LayerNorm2dを使用（BatchNormより安定）
   - ONNX互換性も確保

### メモリ効率

- Boundary Refinement: エッジ領域のみ処理
- Progressive Upsampling: 段階的な解像度変更
- 共有特徴量のキャッシュにより重複計算を回避

## 推奨される使用方法

### 段階的な有効化

複数のモジュールを同時に有効化すると不安定になる可能性があるため、以下の順序を推奨：

1. **Phase 1: 基本的な精緻化**
   - `use_boundary_refinement=True`のみで開始
   - 学習率: 1e-5
   - エポック数: 10-20

2. **Phase 2: 輪郭情報の追加**
   - `use_contour_detection=True`を追加
   - 学習率: 5e-6
   - エポック数: 10-20

3. **Phase 3: 距離情報の追加**
   - `use_distance_transform=True`を追加
   - 学習率: 1e-6
   - エポック数: 10-20

4. **Phase 4: 完全な精緻化**
   - すべてのモジュールを有効化
   - 低学習率（5e-7）で微調整
   - エポック数: 20-50

### 安定した設定例

```python
ModelConfig(
    name="rgb_hierarchical_unet_v2_attention_r64m64_refined_stable",
    # 基本設定
    learning_rate=5e-6,
    gradient_clip_val=1.0,
    mixed_precision=False,
    
    # 2つの精緻化モジュールのみ有効化
    use_boundary_refinement=True,
    use_contour_detection=True,
    
    # 他は無効化
    use_active_contour_loss=False,
    use_progressive_upsampling=False,
    use_subpixel_conv=False,
    use_distance_transform=False,
    use_boundary_aware_loss=False,
)
```

## Active Contour Loss と Contour Detection の違い

### Active Contour Loss（エネルギー最小化アプローチ）

**役割**: セグメンテーションマスクの境界を滑らかにする損失関数

```python
# 境界の長さを最小化（Snake algorithm inspired）
boundary_length = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))
```

**特徴**:
- 予測されたマスクの確率値に直接作用
- 境界線の総長を短くすることでノイズを除去
- 数学的なエネルギー最小化原理に基づく
- 追加のネットワーク構造は不要

**適用場面**:
- ノイズの多い予測結果の平滑化
- 医療画像など滑らかな境界が求められる場合
- 過剰にギザギザした境界の修正

### Contour Detection Branch（明示的な境界学習）

**役割**: 専用のニューラルネットワークで境界位置を予測

```python
# CNNで境界マップを生成
self.contour_branch = nn.Sequential(
    nn.Conv2d(in_channels, 64, 3, padding=1),
    nn.Sigmoid()  # 0-1の境界確率
)
```

**特徴**:
- 独立したネットワークブランチで境界を検出
- 境界位置を0-1の連続値として表現
- マルチタスク学習により特徴表現を改善
- 追加のパラメータと計算が必要

**適用場面**:
- 明確なエッジが重要な場合（建物、道路など）
- 境界の位置精度が求められる場合
- 後処理での境界情報活用

### 同時使用時の相互作用

**相補的効果**:
```
Active Contour Loss ← → Contour Detection
   (平滑化)              (精度向上)
        ↓                    ↓
    滑らかな境界  ＋  正確な位置
              ↓
        最適な境界表現
```

**推奨される組み合わせ**:
1. **人物セグメンテーション**: 両方を低重み（0.1）で使用
2. **医療画像**: Active Contour重視（0.2）、Contour Detection低め（0.05）
3. **建築物検出**: Contour Detection重視（0.2）、Active Contour低め（0.05）

**注意点**:
- 両方の重みが高いと競合する可能性
- 学習初期は片方のみ有効化を推奨
- タスクの性質に応じて重みバランスを調整

## トラブルシューティング

### NaN損失が発生する場合

1. 学習率を下げる（1e-6以下）
2. gradient_clip_valを小さくする（1.0以下）
3. mixed_precisionを無効化
4. 精緻化モジュールを1つずつ有効化

### 境界が不自然な場合

1. Active Contour Lossのsmoothness_weightを調整
2. Boundary Refinementのblend_weightを確認
3. Progressive Upsamplingの使用を検討

### 学習が収束しない場合

1. 精緻化損失の重みを小さくする（0.001〜0.01）
2. ベースモデルを先に学習してから精緻化を追加
3. データ拡張を減らす

### Active ContourとContour Detectionが競合する場合

1. どちらか一方の重みを下げる
2. 段階的に有効化（まずActive Contour、次にContour Detection）
3. タスクに応じて主要な手法を選択

## Boundary Refinement Network (BRN) と Contour Detection Branch の違い

### Boundary Refinement Network (BRN)

**役割**: 予測されたマスクの境界部分を後処理で精緻化

```python
# 処理フロー
mask_logits → エッジ検出 → エッジ領域の調整 → ブレンド
     ↓            ↓              ↓              ↓
 (B,3,H,W)    境界マップ    精緻化処理    元マスク+調整
```

**特徴**:
- **処理タイミング**: デコーダの最終段階（後処理）
- **入力**: 3クラスのマスクロジット
- **出力**: 精緻化された3クラスマスク
- **学習方法**: 残差接続による微調整

**動作の詳細**:
1. Sobel風フィルタで既存マスクから境界を検出
2. 境界領域に特化した3層CNNで調整値を計算
3. `refined = mask_logits + blend_weight * refined_edges * edges`
4. blend_weightは学習可能（0.01で初期化）

### Contour Detection Branch

**役割**: 共有特徴から境界位置を独立タスクとして学習

```python
# 処理フロー
shared_features → 専用CNN → 境界確率マップ → BCELoss
      ↓              ↓            ↓            ↓
  (B,256,H,W)    特徴抽出    (B,1,H,W)    境界学習
```

**特徴**:
- **処理タイミング**: 特徴抽出段階（並列処理）
- **入力**: 256チャンネルの共有特徴
- **出力**: 0-1の境界確率マップ
- **学習方法**: マルチタスク学習の補助タスク

**動作の詳細**:
1. エンコーダの共有特徴から境界情報を抽出
2. 独立した2層CNNで境界マップを生成
3. Sigmoid活性化で0-1の確率値に変換
4. グラウンドトゥルースの境界とBCELossで学習

### 主な違いの比較表

| 観点 | BRN（境界精緻化） | Contour Detection（境界検出） |
|------|-------------------|------------------------------|
| **処理段階** | 後処理（Post-process） | 特徴学習中（Feature-level） |
| **入力データ** | 最終マスク予測 | 中間特徴表現 |
| **出力形式** | 精緻化マスク（3ch） | 境界マップ（1ch） |
| **ネットワーク** | 3層Conv（軽量） | 2層Conv（独立） |
| **学習目的** | 境界の局所調整 | 境界の明示的検出 |
| **影響範囲** | エッジ領域のみ | 特徴表現全体 |

### 同時使用時の相互作用

**処理の流れ**:
```
入力画像
    ↓
エンコーダ（共有特徴抽出）
    ↓
    ├─→ Contour Detection → 境界マップ（補助出力）
    ↓                            ↓
デコーダ                    境界認識の改善
    ↓                            ↓
マスク予測 ←─────────────────────┘
    ↓
BRN（境界精緻化）
    ↓
最終出力
```

**相補的な効果**:
1. **Contour Detection**: 
   - 特徴レベルで境界を学習
   - エンコーダ全体の境界認識能力を向上
   - グローバルな境界構造を捉える

2. **BRN**: 
   - ピクセルレベルで境界を調整
   - 局所的な境界の歪みを修正
   - 最終的な境界品質を向上

**推奨される組み合わせ**:
```python
# 安定した設定
ModelConfig(
    # 先にContour Detectionで境界認識を強化
    use_contour_detection=True,
    contour_loss_weight=0.1,
    
    # 次にBRNで局所調整
    use_boundary_refinement=True,
    # blend_weightは自動学習（初期値0.01）
)
```

**注意点**:
- 両方有効でも基本的に競合しない（処理段階が異なる）
- 計算コストは増加するが、境界品質は大幅に向上
- 学習初期はContour Detectionのみ、安定後にBRNを追加すると効果的

### BRNとContour Detectionの使い分け

**BRNが適している場合**:
- 既存モデルの境界を改善したい
- 計算コストを抑えたい
- 後処理での調整で十分

**Contour Detectionが適している場合**:
- 境界の学習を根本的に改善したい
- マルチタスク学習の恩恵を受けたい
- 境界情報を他の用途にも活用したい

**両方使用が推奨される場合**:
- 最高の境界品質が必要
- 十分な計算リソースがある
- 複雑な境界形状を扱う（人物など）

## データ拡張（Augmentation）仕様

### ROI-safe Light Augmentation (`get_roi_safe_light_transforms`)

軽量なデータ拡張設定。ROI座標の整合性を保つため、幾何学的変換は水平反転のみに制限。

**構成要素**:

1. **幾何学的変換**:
   - `HorizontalFlip(p=0.5)`: 50%の確率で水平反転
   - ※ROI座標は別途調整が必要

2. **色彩変換** (80%の確率で以下のいずれか):
   - `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2)`: 
     - 明度を±20%の範囲で調整
     - コントラストを±20%の範囲で調整
   - `HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20)`:
     - 色相を±10度の範囲で調整
     - 彩度を±20%の範囲で調整
     - 明度を±20%の範囲で調整

3. **ぼかし効果**:
   - `GaussianBlur(blur_limit=(3, 5), p=0.1)`: 10%の確率で3-5ピクセルのガウシアンブラー

4. **正規化**:
   - `Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])`: 恒等変換（値の範囲はそのまま）

**特徴**:
- 計算負荷が軽い
- 基本的な色彩とぼかしの変化のみ
- ROI座標への影響は水平反転のみ
- 高速な学習とテストに適している

### ROI-safe Heavy Augmentation (`get_roi_safe_heavy_transforms`)

より強力なデータ拡張設定。多様な環境条件を模擬しつつ、ROI座標の整合性を維持。

**構成要素**:

1. **幾何学的変換**:
   - `HorizontalFlip(p=0.5)`: 50%の確率で水平反転
   - ※回転や平行移動は無効化されている

2. **色彩変換** (80%の確率で以下のいずれか):
   - `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)`: 
     総合的な色調整
   - `HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20)`: 
     より強い色相・彩度・明度の変化
   - `RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15)`: 
     各色チャンネルを±15の範囲でシフト

3. **照明条件** (50%の確率で以下のいずれか):
   - `RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)`: 
     より強い明度・コントラスト変化
   - `CLAHE(clip_limit=2.0, tile_grid_size=(8, 8))`: 
     適応的ヒストグラム均等化
   - `RandomGamma(gamma_limit=(80, 120))`: 
     ガンマ補正（0.8-1.2の範囲）

4. **環境効果** (10%の確率で以下のいずれか):
   - `RandomRain(rain_type='drizzle')`: 
     小雨効果（drop_length=20, brightness_coefficient=0.7）
   - `RandomFog(alpha_coef=0.1)`: 
     霧効果（透明度10%）
   - `RandomSunFlare`: 
     太陽フレア効果（上半分領域に白色光源）

5. **ぼかし効果** (5%の確率で以下のいずれか):
   - `MotionBlur(blur_limit=7)`: モーションブラー（最大7ピクセル）
   - `GaussianBlur(blur_limit=(3, 7))`: ガウシアンブラー（3-7ピクセル）
   - `MedianBlur(blur_limit=5)`: メディアンブラー（最大5ピクセル）

6. **ノイズ** (5%の確率で以下のいずれか):
   - `GaussNoise()`: ガウシアンノイズ
   - `ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5))`: 
     カメラのISOノイズをシミュレート

7. **画質劣化** (10%の確率で以下のいずれか):
   - `ImageCompression(quality_range=(70, 95))`: 
     JPEG圧縮による劣化（品質70-95%）
   - `Downscale(scale_range=(0.5, 0.9))`: 
     ダウンスケール後アップスケール（解像度50-90%）

8. **正規化**:
   - `Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])`: 恒等変換

**特徴**:
- 実世界の多様な撮影条件を模擬
- 天候効果（雨、霧、太陽光）を含む
- カメラの物理的特性（ノイズ、圧縮）を考慮
- より汎化性能の高いモデル学習が可能

### 使用上の注意点

1. **ROI座標の調整**:
   - 水平反転時は必ずROI座標も反転処理が必要
   - `bbox_params=A.BboxParams(format='pascal_voc', label_fields=[])`により自動調整

2. **計算コスト**:
   - Light: 高速処理、リアルタイム学習に適する
   - Heavy: 処理時間増加、バッチサイズの調整が必要な場合あり

3. **学習戦略**:
   - 初期段階: Light augmentationで安定した学習
   - 中期段階: Heavy augmentationで汎化性能向上
   - 微調整段階: augmentation無しまたはLightで精度追求

4. **パラメータ調整**:
   - 各変換の確率（p値）は経験的に設定
   - データセットの特性に応じて調整推奨
   - 過度な拡張は学習を不安定にする可能性あり