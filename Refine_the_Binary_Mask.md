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
def active_contour_loss(pred_mask, smoothness_weight=0.1):
    """境界の長さと滑らかさを最小化"""
    # 勾配計算
    dy = pred_mask[:, :, 1:, :] - pred_mask[:, :, :-1, :]
    dx = pred_mask[:, :, :, 1:] - pred_mask[:, :, :, :-1]

    # 境界の長さ
    boundary_length = torch.mean(torch.abs(dy)) + torch.mean(torch.abs(dx))

    # 曲率（2次微分）
    ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
    ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
    curvature = torch.mean(torch.abs(ddy)) + torch.mean(torch.abs(ddx))

    return boundary_length + smoothness_weight * curvature
```

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
def boundary_aware_loss(pred, target, boundary_width=3):
    # 境界マップ生成
    kernel = torch.ones(1, 1, boundary_width, boundary_width).to(pred.device)
    boundary = F.conv2d(target.float(), kernel, padding=boundary_width//2)
    boundary = (boundary > 0) & (boundary < boundary_width**2)

    # 重み付きCrossEntropy
    weights = torch.ones_like(pred)
    weights[boundary] = 5.0  # 境界付近の重みを増加

    return F.cross_entropy(pred, target, reduction='none') * weights
```

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