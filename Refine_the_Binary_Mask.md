## Binary Mask 精緻化手法

##### Boundary Refinement Network (BRN)
マスクの境界部分を専門的に処理する小さなサブネットワーク：

```python
class BoundaryRefinementModule(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, mask_logits):
        # エッジ検出
        edges = self.detect_edges(mask_logits)
        # エッジ周辺のみリファイン
        refined_edges = self.edge_conv(mask_logits)
        # 元のマスクとブレンド
        return mask_logits + refined_edges * edges
```

##### Active Contour Loss
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

##### Progressive Upsampling
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

##### Sub-pixel Convolution
ピクセルシャッフルを使用した高品質アップサンプリング：

```python
class SubPixelDecoder(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3 * upscale_factor**2, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
```

##### Contour Detection Branch
輪郭検出を明示的にタスクとして追加：

```python
class ContourAwareDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mask_branch = MaskDecoder(in_channels)
        self.contour_branch = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        mask = self.mask_branch(features)
        contour = self.contour_branch(features)
        # 輪郭情報でマスクを調整
        return mask, contour
```

##### Distance Transform Prediction
距離変換マップを予測してマスクを精緻化：

```python
class DistanceTransformDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.distance_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, features):
        # 境界からの距離を予測
        distance_map = self.distance_head(features)
        # 閾値処理でマスクに変換
        mask = torch.sigmoid(distance_map - self.threshold)
        return mask, distance_map
```

##### Boundary-aware Loss
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