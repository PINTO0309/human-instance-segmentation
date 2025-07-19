# 段階的改善実装ロードマップ

## 実装タスクリスト

### Phase 1: マルチスケール特徴実装 (優先度: 最高)

#### タスク1.1: マルチスケール特徴抽出器
- [ ] `MultiScaleYOLOFeatureExtractor`クラスの設計
- [ ] 複数の特徴マップを同時に抽出する機能
- [ ] 特徴マップのチャンネル数と解像度の管理
- [ ] メモリ効率的な実装

#### タスク1.2: マルチスケールROIAlign
- [ ] 異なる解像度での ROIAlign 実装
- [ ] 特徴マップレベルの自動選択機能
- [ ] 特徴の統合メカニズム（FPN風）

#### タスク1.3: 拡張セグメンテーションヘッド
- [ ] マルチスケール特徴を受け取る新しいヘッド
- [ ] 特徴融合レイヤーの実装
- [ ] 既存モデルとの互換性保持

### Phase 2: 距離認識型ロス関数 (優先度: 高)

#### タスク2.1: 距離マップ生成
- [ ] インスタンス間距離の計算
- [ ] 境界領域の自動検出
- [ ] 効率的な距離マップキャッシング

#### タスク2.2: 重み付きロス実装
- [ ] 距離ベースの重み計算
- [ ] 境界強調メカニズム
- [ ] 既存ロスとの統合

### Phase 3: カスケード型セグメンテーション (優先度: 中)

#### タスク3.1: ステージ設計
- [ ] 3段階のセグメンテーションステージ定義
- [ ] ステージ間の情報伝達メカニズム
- [ ] 各ステージの役割定義

#### タスク3.2: カスケードモデル実装
- [ ] Stage 1: 粗いセグメンテーション
- [ ] Stage 2: 境界精緻化
- [ ] Stage 3: インスタンス分離

### Phase 4: インスタンス間関係モデル (優先度: 低)

#### タスク4.1: アテンション機構
- [ ] ROI間のself-attention実装
- [ ] 効率的な注意機構の設計
- [ ] グラフニューラルネットワークの検討

### Phase 5: 統合システム

#### タスク5.1: 設定管理システム
- [ ] 各機能のON/OFF切り替え
- [ ] ハイパーパラメータ管理
- [ ] 実験設定の保存・読み込み

#### タスク5.2: 段階的学習パイプライン
- [ ] ベースライン学習
- [ ] 機能追加時の転移学習
- [ ] アブレーションスタディ自動化

## 実装スケジュール

### Week 1: 基盤構築
**Day 1-2**: 
- マルチスケール特徴抽出器の実装
- YOLOv9の複数出力対応

**Day 3-4**:
- マルチスケールROIAlignの実装
- 特徴統合メカニズム

**Day 5-7**:
- 距離認識型ロス関数の実装
- 初期検証とデバッグ

### Week 2: 拡張と最適化
**Day 8-10**:
- カスケード型セグメンテーション実装
- 段階的学習の仕組み構築

**Day 11-12**:
- 統合パイプラインの構築
- 実験管理システム

**Day 13-14**:
- 総合テストと最適化
- ドキュメント作成

## ファイル構造

```
src/human_edge_detection/
├── advanced/
│   ├── __init__.py
│   ├── multi_scale_extractor.py    # マルチスケール特徴抽出
│   ├── multi_scale_model.py         # 拡張モデル
│   ├── distance_aware_loss.py      # 距離認識型ロス
│   ├── cascade_segmentation.py     # カスケード型
│   └── relational_module.py        # 関係モデル
├── experiments/
│   ├── config_manager.py            # 実験設定管理
│   ├── ablation_study.py            # アブレーション
│   └── progressive_training.py      # 段階的学習
└── train_advanced.py                # 新しいメインスクリプト
```

## 実験計画

### 実験1: ベースライン確立
```bash
python train_advanced.py --config baseline
```

### 実験2: マルチスケール特徴
```bash
python train_advanced.py --config multiscale --features "3,22,34"
```

### 実験3: マルチスケール + 距離認識ロス
```bash
python train_advanced.py --config multiscale_distance --features "3,22,34" --distance_weight 0.5
```

### 実験4: フルモデル
```bash
python train_advanced.py --config full --enable_cascade --enable_relation
```

## 評価指標

各実験で以下を記録：
1. mIoU（全体、クラス別）
2. Boundary IoU
3. Instance Separation Score
4. 学習時間
5. 推論速度（FPS）
6. メモリ使用量

## チェックポイント

- [ ] Phase 1完了: マルチスケール動作確認
- [ ] Phase 2完了: 距離ロス統合確認
- [ ] Phase 3完了: カスケード動作確認
- [ ] Phase 4完了: 全機能統合確認
- [ ] 最終評価: 性能改善の定量評価