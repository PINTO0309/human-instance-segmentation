
インスタンスセグメンテーションモデルを作成したい。uv と python3.10 を使用した環境構築を前提としたい。

# 要件
3クラスとし、0:背景、1: 推定対象マスク、2: 推定対象外マスク、の３クラスにして、教師ROI（バウンディングボックス）の内側に含まれる教師マスクには推定対象とするマスクと推定対象外のマスクを両方含めるようにすることでアテンションが効きやすい状況を作る。

教師入力１：YOLO ONNX の中間層 (/model.34/Concat_output_0) から抽出した特徴マップ 1x1024x80×80
教師入力２：インスタンスセグメンテーションの推定領域を示す ROI 領域の正規化座標 X1, Y1, X2, Y2 あるいは cx, cy, w, h
教師入力３：インスタンスセグメンテーションの推定対象マスクと推定対象外マスク含む正解マスク（推定対象マスクと推定対象外マスクのみアノテーションされており、この２つ以外の領域は全て背景とする）

このようなアイデアでインスタンスセグメンテーションのデコーダー部分のみを３クラスのインスタンスセグメンテーションタスクと定義した場合、学習とバリデーションのパイプラインを作成したい。


# データの種類
data/images/train2017 : 教師データ用画像(.jpg)
data/images/val2017 : バリデーションデータ用画像(.jpg)
data/annotations/instances_train2017_person_only_no_crowd.json : 教師データ用アノテーションのフルサイズ
data/annotations/instances_train2017_person_only_no_crowd_500.json : 教師データ用アノテーションのフルサイズ
data/annotations/instances_train2017_person_only_no_crowd_100.json : 教師データ用アノテーションの最小サイズ（プログラムの検証用最小セット）
data/annotations/instances_val2017_person_only_no_crowd.json : バリデーションデータ用アノテーションのフルサイズ
data/annotations/instances_val2017_person_only_no_crowd_500.json : バリデーションデータ用アノテーションの中サイズ
data/annotations/instances_val2017_person_only_no_crowd_100.json : バリデーションデータ用アノテーションの最小サイズ（プログラムの検証用最小セット）

学習パイプラインの設計を始める前に最適なLoss選択とLossのパラメータ配分に使用するため、 data/annotations/instances_train2017_person_only_no_crowd.json を分析し、推定対象マスクピクセル, 推定対象外マスクピクセル, 背景ピクセルの構成比率の集計結果を data_analyze.json へ出力すること


# アノテーションデータフォーマット
- MS-COCO 形式
- "category_id": 1 人のみ
- "segmentation" は画像に対するピクセル座標になっているので、データローダーで 0.0 - 1.0 の範囲に正規化して学習に使用する

# 学習
- 学習対象のモデルはデコーダ部のみ、つまりセグメンテーションヘッドのみとする
- 学習後の重みを使用してONNXにエクスポート可能になることを考慮する
- ext_extractor/yolov9_e_wholebody25_0100_1x3x640x640_featext.onnx を使用して推論し /model.34/Concat_output_0 の [1,1024,80,80] の特徴マップを得る
- checkpoints フォルダの配下に 1 epoch ごとに最新のチェックポイント.pthを保存する
- チェックポイント.pth のファイル名は checkpoint_epoch_{4桁に先頭ゼロ埋めしたepoch数}_{Height}x{Width}_{mIoUの小数点以下を4桁にゼロ埋めした整数}.pth
- モデルの最終出力形状は入力解像度と同じ解像度のマスクデータとする

# バリデーション
- 下記の画像を使用して 1 epoch ごとに学習後の重みを使用したテスト推論結果のマスク重畳画像.pngを生成する
000000020992.jpg: 1 person
000000413552.jpg: 2 person
000000109118.jpg: 3 person
000000162732.jpg: 5 person
- マスク重畳画像は上下２段構成とし、上段にGT、下段にテスト推論結果をレンダリングする
- GTには、ROIのバウンディングボックスを赤色、インスタンス番号、アノテーションデータから抽出したマスクを重畳する
- マスクの配色はインスタンスごとに異なる色を採用する
- テスト推論結果にはバウンディングボックスをレンダリングせず、マスクのみを重畳する
- 生成結果のサンプル画像 sample/validation_all_images_epoch_0078.png
- テスト結果生成画像のファイル名は validation_all_images_epoch_{4桁に先頭ゼロ埋めしたepoch数}.png とする

# モデル設計の事前調査・参考情報

以下に、**「YOLO中間特徴 + ROI座標 + 3クラス（背景／推定対象／推定対象外）マスク」構成でインスタンスセグメンテーション用デコーダを学習するモデル設計と Loss 設計の選択肢**を整理します。
「推定対象外」クラスを ROI 内で明示的に教えることで、対象と非対象を対比的に学習させ、注意（アテンション）誘導・誤検出抑制・マスク境界の洗練を狙う設計方針です。

---

# 0. 前提再確認（あなたの設計意図）

* 入力画像は 640×640（例）。YOLO 中間層特徴: `[B, 1024, 80, 80]`（stride=8換算）。
* 教師 ROI: 正規化 (x1,y1,x2,y2) または (cx,cy,w,h)。
* 教師マスク: ROI 内に

  * `1: 推定対象マスク (positive instance)`
  * `2: 推定対象外マスク (distractor / ignore-target)`
    ROI 外および未アノテーション領域は `0: 背景`。
* YOLO 本体は凍結。デコーダのみ学習。

---

# 1. ラベル表現とテンソル整形

## 1.1 解像度整合

* 元マスク (640×640) を stride=8 でダウンサンプル → 80×80 の教師ラベル（学習安定＆速い）。
* もしくはデコーダは ROI 域のみ高解像度アップサンプリングしてフル 640×640 出力。

## 1.2 3クラス one-hot / index

* PyTorch なら `CrossEntropyLoss` に整数クラスIDマップ `[B, H, W]` を渡せばOK（logitsは `[B,3,H,W]`）。
* 「推定対象外」クラスを厳密にラベル付けできない領域（境界不確実、未アノテーション）には ignore\_index を設定する選択肢もあり。

---

# 2. ROI 情報のモデルへの渡し方（重要）

ROI をどう「条件」として渡すかで性能が大きく変わります。代表的に以下。

| 方針                                       | 説明                                                                       | 長所                                    | 短所                       |
| ---------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------- | ------------------------ |
| **A: ROI Crop Head (Mask R-CNN風)**       | 特徴マップから ROIAlign → 小パッチ (e.g., 14×14) → Conv stack → Upsample → 3ch mask | 計算軽い／ROI依存明確                          | ROI外コンテキスト喪失。複数ROIバッチ処理要 |
| **B: ROI Positional Encoding Broadcast** | ROIを 80×80 上にブロードキャスト（距離, inside-flag, norm coords等）→ 特徴にconcat          | 全体文脈維持／単一forwardで複数ROI可 (マスク畳み込みで繰返し) | 入力ch増加／学習に工夫必要           |
| **C: Dynamic Filter (CondInst系)**        | ROIから埋め込みベクトル→動的畳み込みフィルタ生成→特徴へ適用→マスク                                     | 多インスタンス拡張性／柔軟                         | 実装やデバッグ難                 |

**まずは A（ROIAlign型）が実装コストと安定性のバランス◎。**

---

# 3. モデルブロック例（A: ROIAlign型の最小構成）

```text
YOLO frozen backbone/head  ─┐
                            ├─> F ∈ ℝ[B,1024,80,80]
ROI boxes (in image coords) ┘

For each ROI:
  1. Scale ROI to feature coords:
       fx1 = x1 / stride, fx2 = x2 / stride, etc.
  2. ROIAlign(F, ROI) -> R ∈ ℝ[B_roi, 1024, h_r, w_r] (e.g., 14×14)
  3. Small ConvNet:
       Conv(1024→256) + GN/BN + ReLU
       Conv(256→256) × n
       Upsample progressive to fixed high-res (e.g., 56×56 or 112×112)
  4. Final 3-ch Conv → logits_3
  5. (Optional) Paste back into image canvas at ROI loc; else compare to cropped GT mask resized to same ROI resolution.
```

---

# 4. 教師マスクの ROI 切り出し

2通り:

### (a) ROI内クロップ学習（推奨第一歩）

* GTマスクを ROI でクロップし、ROI座標で正規化リスケール（学習ヘッド出力と同サイズ）。
* Loss は ROI内のみ計算（pixel数少なく高速／ROI中心学習）。

### (b) フルイメージ出力 + ROI重み付け

* デコーダがフル 80×80（または640×640）マップを出力。
* Lossは全域計算だが、ROI内画素の重み↑、ROI外は軽く（もしくは背景のみ supervise）。
* ROI外まで対象外クラスを引きずりたくない場合に。

---

# 5. Loss 設計オプション

以下を組み合わせる形で設計します。

## 5.1 ベース損失（必須）

**3クラス排他 → `CrossEntropyLoss`。**

* `weight=[w_bg, w_pos, w_neg]` でクラス不均衡補正。
* ROI内画素が少ない場合、`w_pos`/`w_neg` を上げる。

## 5.2 Dice（またはSoftDice/Tversky）補助

* 特に pos(=1) マスク境界が重要なら CE + Dice(pos)。
* distractor(=2) の描写精度も欲しければ class-wise Dice を両方に適用。
* 係数例: `Loss = λ_ce * CE + λ_dice_pos * Dice_pos + λ_dice_neg * Dice_neg`.

## 5.3 Boundary / Contour Loss（任意）

* マスク境界をラペシアンや距離変換で細線化し、BCEまたは L1 で境界整合性を促す。
* 特に「推定対象外」を境界沿いサンプルとして使う設計も可（positives at edges）。

## 5.4 Contrastive / Margin 的拡張（高度）

ROI内で「pos vs neg」特徴埋め込みを距離分離する（pixel embedding or region embedding）。

* Pixel-level embedding networkから pos/neg センター距離を分離 (InfoNCE, Triplet)。
* 主損失とは別の補助 head として追加。

## 5.5 Focal化（不均衡強い場合）

CEをFocalCE化（γ>0）。または pos/neg 画素のみ focal。
例：`FocalCE(logits, target, gamma=2, alpha_classwise)`。

---

# 6. クラス重み設計の実務フロー

1. データ統計: ROIクロップマスク内で各クラス画素数を集計。
2. 逆頻度重み: `w_c = 1 / (freq_c + ε)` 正規化。
3. 上限クリップ（極端不安定回避）。
4. ログスケーリング（`w_c = log(K/freq_c)`）でマイルド化。

---

# 7. 実装スケッチ（ROIクロップ学習 + CE+Dice）

```python
class RoiSegHead(nn.Module):
    def __init__(self, in_ch=1024, mid_ch=256, num_classes=3, out_size=56):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(32, mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.GroupNorm(32, mid_ch),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.Upsample(size=(out_size, out_size), mode='bilinear', align_corners=False),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, feats_roi):
        x = self.block(feats_roi)
        x = self.upsample(x)
        return self.classifier(x)  # [B_roi,3,out_size,out_size]
```

### Loss 部分

```python
def ce_dice_loss(logits, target, class_weights=None, dice_classes=(1,)):
    # logits: [B,3,H,W], target: [B,H,W] int
    ce = F.cross_entropy(logits, target, weight=class_weights, reduction='mean')

    # Dice for selected classes only (e.g., pos=1, neg=2)
    dice_total = 0.0
    for c in dice_classes:
        # one-hot slice
        tgt_c = (target == c).float()
        prob_c = F.softmax(logits, dim=1)[:, c, ...]
        inter = (prob_c * tgt_c).sum()
        denom = prob_c.sum() + tgt_c.sum() + 1e-6
        dice = 1 - (2 * inter / denom)
        dice_total = dice_total + dice
    dice_total = dice_total / max(len(dice_classes), 1)

    return ce + dice_total
```

---

# 8. ROI 情報を特徴に注入する改良（Bプラン）

ROI を単に切り出すのではなく、\*\*「YOLO特徴全体 + ROIマップをチャネル追加」\*\*でコンテキストを生かす方法：

生成する補助チャネル例（80×80）:

* binary mask: inside ROI=1, else 0
* 距離到中心: 正規化 (dx, dy)
* スケール: ROI幅/高さを全ピクセルにブロードキャスト
* relative coords: ( (x - cx)/w, (y - cy)/h )

これらを concat → `[B,1024+K,80,80]` → U-Net風デコーダ → 3ch logits。
ROI依存の空間注意が自然に学習される。

---

# 9. 推定対象外（クラス2）の使いどころ

推定対象外は単なる「背景の一部」とせず、\*\*「ROI内の競合インスタンス」「ノイズ」「似たパターン」\*\*を明示教師する点に価値があります。

期待効果:

* 誤った大域領域膨張を抑止（posとnegを分離）。
* ROI中心へのアテンション強化（特に複数物体密集時）。
* 後段で pos のみ抜き出したいとき信頼度向上。

---

# 10. 学習バッチ構成の注意

| 設定                   | 推奨                                                        |
| -------------------- | --------------------------------------------------------- |
| ROI サンプリング           | 各画像から K 個（正例GT box, hard負例, IoU thresholdで混ぜる）            |
| Hard Negative Mining | 推定対象外が多いROIを優先サンプリング                                      |
| Data Aug             | ROI jitter / scale jitter / random crop to encourageロバスト性 |

---

# 11. 推論フロー例（ROI提供前提）

1. 画像入力 → YOLO推論（凍結）→ bbox提案 (または外部ROI)。
2. 中間特徴取り出し。
3. 各 ROI についてデコーダ forward → 3chマスク (ROI解像度)。
4. posチャンネル確率マップを select（必要ならnegも利用して信頼度再計算）。
5. 画像座標に戻して貼り付け／閾値化／ポリゴン化。

---

# 12. アブレーション（やると効果検証が早い）

| 実験                                  | 比較目的      |
| ----------------------------------- | --------- |
| 2クラス (背景/推定対象) vs 3クラス              | neg導入効果   |
| ROIクロップ vs 全体+ROIチャネル               | コンテキスト重要度 |
| CEのみ vs CE+Dice vs CE+Dice+Boundary | 境界精度と安定性  |
| クラス重みあり/なし                          | 不均衡耐性     |

---

# 13. 最小ステップで始める推奨構成（実装優先）

1. **ROIAlign型クロップヘッド**（A）。
2. 出力 56×56（高速・十分）。
3. **CrossEntropyLoss(weighted) + Dice(pos)**。
4. Distractor クラスは CE のみ（初期はDice対象外）で学習安定化。
5. 学習収束後、Diceをnegにも拡張、Boundary loss追加で精緻化。

---

# 14. クイック疑似コード（学習ループ断片）

```python
# feats: [B,1024,80,80] from YOLO (frozen)
# rois:  [R,5] (batch_idx, x1,y1,x2,y2 in image px)
# gt_masks_full: list of [H,W] int masks per image (0/1/2)

roi_feats = roi_align(feats, rois, output_size=(14,14), spatial_scale=1/stride)

# build roi-level gt targets
roi_targets = []
for (b,x1,y1,x2,y2) in rois:
    gt = gt_masks_full[b][y1:y2, x1:x2]
    gt = resize(gt, (56,56), interp='nearest')
    roi_targets.append(torch.from_numpy(gt))
roi_targets = torch.stack(roi_targets).to(device)  # [R,56,56]

logits = seg_head(roi_feats)  # [R,3,56,56]

loss = ce_dice_loss(logits, roi_targets, class_weights=torch.tensor([w0,w1,w2]).to(device))
loss.backward()
```
