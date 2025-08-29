# 人物エッジ検出モデル学習分析レポート

## 1. 現状分析

### 1.1 学習状況
- **学習設定**:
  - データセット: COCO 2017 (Training: 167,602サンプル, Validation: 7,688サンプル)
  - エポック数: 53/100 (継続中)
  - バッチサイズ: 32
  - 学習率: コサインスケジューラー (1e-3 → 1e-6)
  - オプティマイザ: AdamW (weight decay: 1e-4)

### 1.2 性能推移
- **mIoU推移**: 初期の改善後、エポック10以降継続的に低下（0.44 → 0.41）
- **クラス別IoU**:
  - Background (class 0): わずかな改善（+0.019）後、頭打ち状態
  - Target (class 1): エポック9でピーク（0.572）後、継続的に低下（-0.025）
  - Non-target (class 2): エポック9でピーク（0.318）後、わずかに低下（-0.007）

### 1.3 可視化結果から確認された問題点
1. **境界の不明瞭さ**: 人物の輪郭が不鮮明で、背景との分離が不十分
2. **インスタンス分離の不良**: 複数人物が密接している場合、マスクが混在
3. **小物体の検出精度低下**: 遠距離の小さな人物の検出が不正確
4. **複数人物間の境界混同**: 隣接する人物間の境界が曖昧

## 2. 問題の根本原因分析

### 2.1 データセットの特性による課題
- **高い多重インスタンス率**: 58%の画像に複数人物が存在
- **インスタンス間の重なり**: 40%の画像でインスタンス間にオーバーラップ
- **ROI内の複雑性**: 平均3つのインスタンスがROI内に存在、72%のROIで重なり発生

### 2.2 モデルアーキテクチャの限界
1. **固定ROIサイズ**: 28x28の固定サイズでは、様々なスケールの人物に対応困難
2. **単一スケール特徴**: YOLOv9の最終層特徴のみ使用、マルチスケール情報の欠如
3. **位置情報の不足**: 空間的な位置関係を明示的にモデル化していない

### 2.3 学習プロセスの問題
1. **クラス不均衡への対処不足**: Non-target classの重みが不十分の可能性
2. **過学習の兆候**: 早期（エポック9）でのピーク後の性能低下
3. **難易度適応の欠如**: 簡単なサンプルと困難なサンプルを同等に扱っている

## 3. 改善提案

### 3.1 短期的改善策（すぐに実装可能）

#### A. 学習戦略の改善
```python
# 1. 学習率の調整
# 現在のエポック53でmIoUが低下しているため、学習率を下げて再開
--lr 1e-4  # 1/10に削減
--min_lr 1e-7

# 2. 早期停止の実装
--early_stopping_patience 10
--early_stopping_metric "val/mIoU"

# 3. クラス重みの再調整
# Non-targetクラスの重みをさらに増加
separation_aware_weights = {
    "background": 0.538,
    "target": 0.750,
    "non_target": 2.054  # 1.712 × 1.2
}
```

#### B. データ拡張の強化
```python
# geometric_augmentations.py
class EnhancedAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            A.RandomScale(scale_limit=0.2, p=0.5),
            A.RandomCrop(height=512, width=512, p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            # 境界強調のための拡張
            A.ElasticTransform(p=0.3),
            A.GridDistortion(p=0.2),
            # コントラスト調整
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ])
```

### 3.2 中期的改善策（アーキテクチャ改良）

#### A. マルチスケール特徴の活用
```python
class MultiScaleROISegmentationHead(nn.Module):
    def __init__(self):
        super().__init__()
        # YOLOv9の複数層から特徴を抽出
        self.fpn = FeaturePyramidNetwork(
            in_channels=[256, 512, 1024],
            out_channels=256
        )
        
    def forward(self, multi_scale_features, rois):
        # 複数スケールでROIAlign実行
        roi_feats = []
        for level, features in enumerate(multi_scale_features):
            level_rois = self.assign_rois_to_level(rois, level)
            if len(level_rois) > 0:
                roi_feat = self.roi_align(features, level_rois)
                roi_feats.append(roi_feat)
        
        # 特徴を統合
        combined_features = self.combine_multi_scale_features(roi_feats)
        return self.decode(combined_features)
```

#### B. 位置エンコーディングの追加
```python
class PositionalEncodingModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pe = self.create_2d_positional_encoding(channels)
        
    def forward(self, x):
        # 空間的位置情報を特徴に追加
        return x + self.pe[:, :x.size(2), :x.size(3)]
```

#### C. インスタンス間関係のモデル化
```python
class RelationalReasoningModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8
        )
        
    def forward(self, roi_features):
        # ROI間の関係を学習
        N = roi_features.size(0)
        roi_flat = roi_features.view(N, -1)
        
        # Self-attention でROI間の関係をモデル化
        attended_features, _ = self.attention(
            roi_flat, roi_flat, roi_flat
        )
        
        return attended_features.view_as(roi_features)
```

### 3.3 長期的改善策（パイプライン全体の見直し）

#### A. 距離認識型ロス関数
```python
class DistanceAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_loss = SegmentationLoss()
        
    def forward(self, predictions, targets, instance_distances):
        # 基本ロス
        base_loss, loss_dict = self.base_loss(predictions, targets)
        
        # 境界領域の重み付け
        boundary_weights = self.compute_boundary_weights(targets)
        
        # 距離に基づく重み付け
        # 近いインスタンス間の境界により大きな重みを付与
        distance_weights = 1.0 / (instance_distances + 1e-6)
        
        # 重み付きロスの計算
        weighted_loss = base_loss * boundary_weights * distance_weights
        
        return weighted_loss.mean(), loss_dict
```

#### B. カスケード型セグメンテーション
```python
class CascadeSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: 粗いセグメンテーション
        self.coarse_head = ROISegmentationHead(mask_size=28)
        
        # Stage 2: 境界領域の精緻化
        self.refinement_head = BoundaryRefinementHead(mask_size=56)
        
        # Stage 3: インスタンス分離の改善
        self.separation_head = InstanceSeparationHead()
        
    def forward(self, features, rois):
        # 段階的な精緻化
        coarse_masks = self.coarse_head(features, rois)
        refined_masks = self.refinement_head(features, rois, coarse_masks)
        final_masks = self.separation_head(features, rois, refined_masks)
        
        return final_masks
```

#### C. アクティブラーニングの導入
```python
class ActiveLearningStrategy:
    def __init__(self, model, uncertainty_threshold=0.5):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
        
    def select_hard_samples(self, dataset, num_samples=1000):
        uncertainties = []
        
        for sample in dataset:
            # 予測の不確実性を計算
            with torch.no_grad():
                outputs = self.model(sample['image'])
                # エントロピーベースの不確実性
                uncertainty = self.compute_entropy(outputs)
                
            uncertainties.append({
                'idx': sample['idx'],
                'uncertainty': uncertainty,
                'has_overlap': sample['has_overlap'],
                'num_instances': sample['num_instances']
            })
        
        # 最も不確実なサンプルを選択
        hard_samples = sorted(
            uncertainties, 
            key=lambda x: x['uncertainty'], 
            reverse=True
        )[:num_samples]
        
        return hard_samples
```

## 4. 実装優先順位

### Phase 1 (即座に実装)
1. 学習率の調整と早期停止の実装
2. クラス重みの再調整（Non-targetクラスの重み増加）
3. 検証頻度の増加（エポック毎→5バッチ毎）

### Phase 2 (1週間以内)
1. データ拡張の強化（境界強調、弾性変形）
2. 難サンプルマイニングの実装
3. 境界領域への重み付けロス

### Phase 3 (2-3週間)
1. マルチスケール特徴の活用
2. 位置エンコーディングの追加
3. カスケード型セグメンテーション

## 5. 評価指標の改善

現在のmIoUに加えて、以下の指標も追跡することを推奨：

1. **Boundary IoU**: 境界領域（エッジから5ピクセル以内）のIoU
2. **Instance Separation Score**: インスタンス間の分離度
3. **Scale-aware mIoU**: オブジェクトサイズ別のmIoU
4. **Occlusion-aware metrics**: 重なり度合い別の性能評価

## 6. まとめ

現在の主な課題は、複数インスタンスの分離と境界の明瞭化です。短期的には学習戦略の改善とデータ拡張の強化により、ある程度の改善が期待できます。しかし、根本的な解決には、マルチスケール特徴の活用やインスタンス間関係のモデル化など、アーキテクチャレベルの改良が必要です。

特に重要なのは、単純なピクセル単位の分類問題として扱うのではなく、インスタンス間の空間的関係や距離を考慮したアプローチを取り入れることです。これにより、密集した人物群でも個々のインスタンスを正確に分離できるようになると期待されます。