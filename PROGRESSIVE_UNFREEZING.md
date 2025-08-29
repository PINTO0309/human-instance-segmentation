# Progressive Unfreezing（段階的解凍）実装ガイド

## 概要

エンコーダーの凍結を段階的に解除し、徐々にファインチューニングすることで、より安定した学習と高い精度を実現します。

## 実装内容

### 1. DistillationUNetWrapperの拡張

```python
# 初期化時にprogressive_unfreezeフラグを設定
model = DistillationUNetWrapper(
    student_encoder="timm-efficientnet-b0",
    teacher_checkpoint_path="ext_extractor/2020-09-23a.pth",
    progressive_unfreeze=True  # 段階的解凍を有効化
)
```

### 2. 解凍スケジュール

```python
# 解凍スケジュールの取得
schedule = model.get_progressive_unfreeze_schedule(
    total_epochs=50,           # 総エポック数
    unfreeze_start_epoch=10,   # 解凍開始エポック
    unfreeze_rate=5            # N エポックごとに1ブロック解凍
)

# スケジュール例：
# Epoch 0-9:   全エンコーダー凍結（デコーダーのみ学習）
# Epoch 10-14: 最後の1ブロック解凍
# Epoch 15-19: 最後の2ブロック解凍
# Epoch 20-24: 最後の3ブロック解凍
# ...
```

### 3. 訓練ループでの使用

```python
def train_with_progressive_unfreeze(model, optimizer, epochs):
    # 初期オプティマイザー（デコーダーのみ）
    optimizer = torch.optim.AdamW(
        model.student.get_decoder_parameters(),
        lr=1e-4
    )

    # 解凍スケジュールを取得
    unfreeze_schedule = model.get_progressive_unfreeze_schedule(
        total_epochs=epochs,
        unfreeze_start_epoch=10,
        unfreeze_rate=5
    )

    for epoch in range(epochs):
        # 現在のエポックでの解凍ブロック数を確認
        blocks_to_unfreeze = unfreeze_schedule[epoch]

        # 前エポックと異なる場合は更新
        if epoch > 0 and blocks_to_unfreeze != unfreeze_schedule[epoch-1]:
            # エンコーダーブロックを解凍
            unfrozen_params = model.unfreeze_encoder_blocks(
                num_blocks=blocks_to_unfreeze,
                learning_rate_scale=0.1  # エンコーダーの学習率は1/10
            )

            # オプティマイザーに新しいパラメータグループを追加
            if unfrozen_params:
                # デコーダーパラメータ
                decoder_params = model.student.get_decoder_parameters()

                # 新しいオプティマイザーを作成（差別的学習率）
                optimizer = torch.optim.AdamW([
                    {'params': decoder_params, 'lr': 1e-4},
                    {'params': unfrozen_params, 'lr': 1e-5}  # エンコーダーは低学習率
                ])

                print(f"Epoch {epoch}: Updated optimizer with {blocks_to_unfreeze} unfrozen blocks")

        # 通常の訓練処理
        train_one_epoch(model, optimizer, ...)
```

## 利点

1. **安定性向上**: 最初はデコーダーのみ学習し、その後徐々にエンコーダーも調整
2. **破壊的忘却の防止**: ImageNet事前学習の知識を保持しながら適応
3. **精度向上**: タスク特化の特徴抽出が可能に
4. **柔軟な制御**: 解凍速度やタイミングを自由に調整可能

## 推奨設定

### 小規模データセット（<1000枚）
```python
unfreeze_start_epoch=20    # 遅めに開始
unfreeze_rate=10           # ゆっくり解凍
learning_rate_scale=0.01   # 非常に低い学習率
```

### 中規模データセット（1000-10000枚）
```python
unfreeze_start_epoch=10    # 標準的なタイミング
unfreeze_rate=5            # 標準的な速度
learning_rate_scale=0.1    # 標準的な学習率スケール
```

### 大規模データセット（>10000枚）
```python
unfreeze_start_epoch=5     # 早めに開始
unfreeze_rate=3            # 速めに解凍
learning_rate_scale=0.3    # より高い学習率も可能
```

## 注意事項

1. **メモリ使用量**: 解凍するとメモリ使用量が増加
2. **学習率調整**: エンコーダーには必ず低い学習率を使用
3. **検証**: 各解凍後に検証損失を確認し、過学習を監視
4. **Early Stopping**: 過学習の兆候があれば解凍を停止

## train_distillation_staged.pyへの統合例

```python
# コマンドライン引数を追加
parser.add_argument('--progressive_unfreeze', action='store_true',
                   help='Enable progressive unfreezing')
parser.add_argument('--unfreeze_start', type=int, default=10,
                   help='Epoch to start unfreezing')
parser.add_argument('--unfreeze_rate', type=int, default=5,
                   help='Unfreeze one block every N epochs')

# モデル作成時
model = DistillationUNetWrapper(
    student_encoder=config.distillation.student_encoder,
    teacher_checkpoint_path=config.distillation.teacher_checkpoint,
    progressive_unfreeze=args.progressive_unfreeze
)

# 訓練ループ内
if args.progressive_unfreeze and epoch >= args.unfreeze_start:
    # 段階的解凍の処理
    ...
```

## 実験結果の期待値

| エポック | 状態 | 期待mIoU改善 |
|---------|------|-------------|
| 0-9 | デコーダーのみ | ベースライン |
| 10-19 | 深層1-2ブロック解凍 | +1-2% |
| 20-29 | 中間層3-4ブロック解凍 | +2-3% |
| 30-39 | 浅層5-6ブロック解凍 | +1-2% |
| 40-50 | 全層解凍 | +0-1% |

合計で3-8%の精度向上が期待できます。