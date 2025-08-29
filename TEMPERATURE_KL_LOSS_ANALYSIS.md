# Temperature Scheduling and KL Loss Analysis

## 概要
Temperature（温度パラメータ）がepoch経過とともに減衰することで、KL Lossの挙動が大きく変化します。

## Temperature の役割

### 数式での影響
```python
# Temperature適用後のソフト確率
student_soft = sigmoid(student_logit / T)
teacher_soft = sigmoid(teacher_logit / T)

# KL Divergence
KL(teacher || student) = teacher_soft * log(teacher_soft/student_soft) + 
                         (1-teacher_soft) * log((1-teacher_soft)/(1-student_soft))
```

## Epoch経過による影響

### 1. 訓練初期（T=10, 高温度）
- **確率分布が平滑化**: sigmoid(logit/10) → 0.5に近い値
- **KL Lossが小さく**: 予測の違いに対して寛容
- **効果**:
  - Studentのミスに対してペナルティが小さい
  - 大まかなパターンを学習しやすい
  - 勾配が安定し、学習が安定

### 2. 訓練中期（T=5, 中温度）
- **確率分布が適度にシャープ化**
- **KL Lossが中程度**: バランスの取れた学習
- **効果**:
  - ソフトターゲットとハードターゲットのバランス
  - 細部の学習を開始

### 3. 訓練後期（T=1, 低温度）
- **確率分布がシャープ化**: sigmoid(logit/1) → 0 or 1に近い値
- **KL Lossが大きく**: 予測の違いに対して厳格
- **効果**:
  - Teacherの出力に正確に一致することを要求
  - ファインチューニング段階
  - 最終的な精度向上

## 実際の訓練への影響

### 利点
1. **段階的学習**: 粗から密への自然な学習プロセス
2. **過学習防止**: 初期の過度な適合を防ぐ
3. **収束の安定性**: 急激な変化を避け、スムーズな最適化

### 観察される現象
- **Epoch 1-20**: KL Lossは小さいが、徐々に増加
- **Epoch 20-80**: KL Lossが安定または微増
- **Epoch 80-100**: KL Lossが増加し、精密な調整

## コード内での実装
```python
# train_distillation_staged.py
if enable_temp_scheduling:
    new_temperature = loss_fn.update_temperature(
        current_epoch=epoch,
        total_epochs=config.training.num_epochs,
        final_temperature=final_temperature,
        schedule_type=schedule_type
    )
```

## 最適な設定
- **初期温度**: T=10（ソフトな学習開始）
- **最終温度**: T=1（ハードな最終調整）
- **スケジュール**: Linear（線形減衰）が安定

## まとめ
Temperature減衰により、KL Lossは：
- **初期**: 小さく寛容 → 基本パターン学習
- **中期**: 適度 → バランスの取れた学習
- **後期**: 大きく厳格 → 精密なマッチング

これにより、より効果的な知識蒸留が実現されます。