# Prototypical Contrastive Learning (PCL) — 時系列版

論文 "Prototypical Contrastive Learning of Unsupervised Representations" (ICLR 2021) の時系列データへの適用。

---

## ファイル構成と役割

```
.
├── train.py            # メインの学習スクリプト
├── visualize_tsne.py   # チェックポイントからt-SNE画像を生成
├── requirements.txt    # 依存パッケージ
├── datasets/           # ベンチマークデータ (ETT-small/ETTh1.csv など)
└── pcl/
    ├── model.py        # PCLモデル (Transformer/CNNエンコーダ + Momentumエンコーダ + キュー)
    ├── loss.py         # ProtoNCE損失関数
    ├── clustering.py   # k-meansクラスタリング + 濃度推定 φ
    └── dataset.py      # 時系列Dataset・Augmentationユーティリティ
```

---

## 各ファイルの詳細

### `pcl/model.py` — PCLモデル

**クラス: `PCL`**

| 要素 | 説明 |
|---|---|
| `encoder_q` | 学習対象のQueryエンコーダ (Transformer or CNN) |
| `encoder_k` | 重みが固定されたMomentumエンコーダ (EMA更新) |
| `queue` | インスタンスワイズ対照学習用の負例キュー (shape: `dim × queue_size`) |

- `forward(x_q, x_k)`: 2つのaugmented viewを受け取り、queryとkey特徴量を返す。Momentumエンコーダの重みをEMAで更新し、keyをキューに追加する
- `get_features(loader, device)`: Momentumエンコーダで全訓練データの特徴量を抽出する（クラスタリング用）

```
encoder_q ──→ L2-normalize ──→ q
encoder_k ──→ L2-normalize ──→ k ──→ queue に追加
(EMAで更新)
```

**エンコーダ (`--encoder`)**

| エンコーダ | 入力 | 処理 | 出力 |
|---|---|---|---|
| `transformer` (デフォルト) | `(B, C, L)` | Linear投影 → 位置エンコーディング → Transformer → 平均プーリング → Linear | `(B, dim)` |
| `cnn` | `(B, C, L)` | Conv1d×3 (stride=2) → AdaptiveAvgPool1d(1) → Linear | `(B, dim)` |

---

### `pcl/loss.py` — ProtoNCE損失

**クラス: `ProtoNCELoss`**

論文 Eq.11 を実装。2つの項の和：

```
L_ProtoNCE = L_InfoNCE (インスタンスワイズ) + (1/M) * Σ L_proto_m (プロトタイプ対照)
```

| メソッド | 役割 |
|---|---|
| `_info_nce` | MoCo方式のInfoNCE損失。qとkの正例ペア vs キュー内の負例 |
| `_proto_nce_single` | 1粒度分のプロトタイプ対照損失。qと割当プロトタイプ(正例) vs ランダムサンプルしたrプロトタイプ(負例) |
| `forward` | warm_up=True のときはInfoNCEのみ、Falseのときは両方の和を返す |

---

### `pcl/clustering.py` — クラスタリング (E-step)

EMフレームワークのE-stepに対応。

| 関数 | 役割 |
|---|---|
| `_run_kmeans` | faiss (Linux) または sklearn (macOS) でk-meansを実行 |
| `_concentration` | 各プロトタイプの濃度 φ を計算 (論文 Eq.12) |
| `cluster_features` | k_listの各kに対してクラスタリング＋φ計算をまとめて実行 |

**濃度推定 φ (Eq.12)**:

```
φ_c = Σ||v'_z - c||₂ / (Z * log(Z + α))
```

- クラスタが疎 → φ大 → 類似度をダウンスケール（崩壊防止）
- クラスタが密 → φ小 → 類似度をアップスケール
- 最後に `mean(φ) = τ` になるよう正規化

戻り値は `(centroids, assignments, phi)` のリスト（粒度M個分）。

---

### `pcl/dataset.py` — 時系列データセットユーティリティ

| クラス | 役割 |
|---|---|
| `Jitter` | ガウスノイズを付加するAugmentation |
| `WindowSlicing` | ランダムクロップ＋リサイズ（画像のRandomCropに相当） |
| `TimeSeriesAugmentation` | Jitter + WindowSlicing の組み合わせ |
| `TwoViewTransform` | 同じウィンドウに異なるAugmentationを2回適用し、2つのviewを生成 |
| `IndexedDataset` | 既存Datasetをラップし、`(data, label, index)` を返すようにする |
| `TimeSeriesDataset` | CSVから時系列を読み込み、スライディングウィンドウでサンプルを生成 |

**`TimeSeriesDataset` の動作**:

1. CSVを読み込み、日付列を除外
2. `univariate`: `--target-col` のカラムのみ使用 → shape `(T, 1)`
3. `multivariate`: 全数値カラムを使用 → shape `(T, C)`
4. train/val/test に分割 (60% / 20% / 20%)
5. Z-score正規化 (`std + 1e-8` でゼロ除算を防止)
6. スライディングウィンドウで `(C, seq_len)` のサンプルを生成

`IndexedDataset` でサンプルのインデックスを返すのは、クラスタリング結果の `assignments[i]` とバッチ内サンプルを対応付けるために必要。

---

### `train.py` — 学習スクリプト (EMループ)

EMフレームワークの全体ループを実装。

```
for epoch in range(epochs):
    if epoch >= warm_up_epochs:
        # E-step: Momentumエンコーダで全特徴抽出 → k-means → φ推定
        features = model.get_features(cluster_loader)
        cluster_results = cluster_features(features, K_list)

    # M-step: ProtoNCE損失でエンコーダを更新
    for (x_q, x_k), _, indices in train_loader:
        q, k = model(x_q, x_k)
        loss = ProtoNCELoss(q, k, queue, cluster_results, indices, warm_up)
        loss.backward()
        optimizer.step()
```

| フェーズ | エポック | 内容 |
|---|---|---|
| warm-up | 0 〜 warm_up_epochs-1 | InfoNCEのみ（クラスタリングなし） |
| PCL | warm_up_epochs 〜 end | E-step (k-means) + M-step (ProtoNCE) |

LRスケジュール: CosineAnnealingLR（最終エポックまで余弦的に減衰）

---

### `visualize_tsne.py` — t-SNE可視化（論文 Figure 4 相当）

チェックポイントを渡すと、Momentumエンコーダが抽出した特徴量をt-SNEで2次元に圧縮し、時系列上の時刻位置で色付けした散布図をPNGで保存する。

- 青 (0): 時系列の最初に位置するウィンドウ
- 赤 (1): 時系列の最後に位置するウィンドウ
- 学習が進むにつれて、時系列の構造を反映したクラスタが現れる

`train.py` の `--tsne-freq` でも同じ処理をエポックごとに自動実行できる。

---

## ファイル間の依存関係

```
train.py
├── pcl/model.py       (PCLモデル: Transformer/CNN エンコーダ)
├── pcl/loss.py        (ProtoNCE損失)
├── pcl/clustering.py  (E-step: k-means + φ)
│   └── faiss / sklearn
└── pcl/dataset.py     (TimeSeriesDataset, TwoViewTransform, IndexedDataset)

visualize_tsne.py
├── pcl/model.py
├── pcl/dataset.py
├── sklearn.manifold.TSNE
└── matplotlib
```

---

## 実行方法

### インストール

```bash
pip3 install torch numpy pandas scikit-learn matplotlib --break-system-packages
```

---

### コマンドの読み方

すべての設定は `--オプション名 値` の形式でコマンドラインから指定する。
**省略した場合はデフォルト値が使われる**（後述のオプション一覧参照）。

```
python3 train.py --encoder cnn --epochs 100 --batch-size 128
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 これらはすべてオプション。書かなければデフォルト値が適用される。
```

`python3 train.py` だけで実行した場合は、全オプションがデフォルト値になる。
→ `./datasets/ETT-small/ETTh1.csv` の `OT` カラムを Transformer で200エポック学習する。

---

### train.py のオプション一覧

重要度: ★★★ 必ず確認する　／　★★☆ 状況によって変える　／　★☆☆ 基本触らなくていい

#### データ

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--data-path` | `./datasets/ETT-small/ETTh1.csv` | 使うCSVファイルのパス |
| ★★★ | `--variables` | `univariate` | `univariate`: target-colのみ / `multivariate`: 全数値カラム |
| ★★☆ | `--target-col` | `OT` | univariateのときに使うカラム名 |
| ★★☆ | `--seq-len` | `96` | 1サンプルのウィンドウ長（タイムステップ数） |
| ★☆☆ | `--stride` | `1` | スライディングウィンドウのストライド（1=最大サンプル数） |

#### エンコーダ

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★☆ | `--encoder` | `transformer` | `transformer` または `cnn` |
| ★☆☆ | `--d-model` | `64` | Transformerの埋め込み次元（`--encoder transformer` のみ有効） |
| ★☆☆ | `--nhead` | `4` | Transformerのアテンションヘッド数 |
| ★☆☆ | `--num-layers` | `2` | Transformerのレイヤー数 |
| ★☆☆ | `--dim` | `128` | エンコーダ出力の特徴量次元数 |

#### Augmentation

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★☆☆ | `--jitter-sigma` | `0.03` | Jitterのノイズ強度 |
| ★☆☆ | `--slice-ratio` | `0.9` | WindowSlicingの切り取り割合（0〜1） |
| ★☆☆ | `--no-jitter` | （なし） | このフラグを付けるとJitterを無効化 |
| ★☆☆ | `--no-slicing` | （なし） | このフラグを付けるとWindowSlicingを無効化 |

#### 対照学習・クラスタリング

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--num-clusters` | `50 200 500` | クラスタ数（複数指定で階層的プロトタイプ）。訓練ウィンドウ数より小さくする必要がある |
| ★★☆ | `--queue-size` | `4096` | 負例キューのサイズ。訓練ウィンドウ数より大きくしても意味がない |
| ★★☆ | `--r` | `500` | 1ステップでサンプルする負例プロトタイプ数。`--num-clusters` の最小値より小さくする必要がある |
| ★☆☆ | `--tau` | `0.1` | 温度パラメータ |
| ★☆☆ | `--momentum` | `0.999` | Momentumエンコーダの更新係数 |
| ★☆☆ | `--alpha` | `10.0` | 濃度推定の平滑化係数 |

#### 学習

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--epochs` | `200` | 総エポック数 |
| ★★★ | `--output-dir` | `./checkpoints` | チェックポイントとt-SNE画像の保存先 |
| ★★☆ | `--batch-size` | `256` | メモリが足りなければ減らす |
| ★★☆ | `--warm-up-epochs` | `20` | 最初のNエポックはクラスタリングなしで学習 |
| ★★☆ | `--tsne-freq` | `0` | Nエポックごとにt-SNE画像を保存。`0`で無効 |
| ★★☆ | `--resume` | （なし） | 途中から再開する場合にチェックポイントのパスを指定 |
| ★☆☆ | `--lr` | `1e-4` | 学習率（CosineAnnealingLRで減衰） |
| ★☆☆ | `--weight-decay` | `1e-4` | 重み減衰 |
| ★☆☆ | `--workers` | `2` | データロードの並列数 |
| ★☆☆ | `--save-freq` | `10` | Nエポックごとにチェックポイントを保存 |
| ★☆☆ | `--tsne-samples` | `500` | t-SNEで使うウィンドウ数 |

---

### 実行例

#### 最小構成（デフォルト値のまま）

```bash
python3 train.py
```

`./datasets/ETT-small/ETTh1.csv` の `OT` カラムを Transformer で200エポック学習する。

#### ETTh1 / CNNエンコーダ

```bash
python3 train.py \
  --encoder cnn \
  --epochs 100 \
  --batch-size 128 \
  --tsne-freq 20 \
  --output-dir ./checkpoints_cnn
```

#### ETTh1 / 多変量

```bash
python3 train.py \
  --variables multivariate \
  --encoder transformer \
  --d-model 64 \
  --num-clusters 50 200 500 \
  --epochs 200
```

#### 別データセット（Weather）

```bash
python3 train.py \
  --data-path ./datasets/weather/weather.csv \
  --variables multivariate \
  --seq-len 96 \
  --num-clusters 50 200 500
```

#### 途中から再開

```bash
python3 train.py --resume ./checkpoints/pcl_epoch0100.pth
```

チェックポイントに保存された引数が自動で復元されるが、`--epochs` など追加指定することもできる。

---

### visualize_tsne.py のオプション一覧

学習済みチェックポイントからt-SNE画像を生成する単体スクリプト。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | （必須） | 読み込むチェックポイントのパス |
| `--data-path` | `./datasets/ETT-small/ETTh1.csv` | データセットのパス |
| `--variables` | `univariate` | `univariate` または `multivariate` |
| `--target-col` | `OT` | univariateのときに使うカラム名 |
| `--seq-len` | `96` | ウィンドウ長（学習時と同じ値を指定） |
| `--samples` | `500` | t-SNEで使うウィンドウ数 |
| `--perplexity` | `30.0` | t-SNEのperplexityパラメータ |
| `--output` | `./tsne.png` | 出力画像のパス |

#### 実行例

```bash
python3 visualize_tsne.py \
  --checkpoint ./checkpoints/pcl_epoch0200.pth \
  --output ./tsne_epoch200.png
```

---

## 論文との対応

| 論文の要素 | 実装箇所 |
|---|---|
| EMフレームワーク | `train.py` の学習ループ |
| ProtoNCE損失 Eq.11 | `pcl/loss.py:ProtoNCELoss` |
| 濃度推定 φ Eq.12 | `pcl/clustering.py:_concentration` |
| φの正規化 (mean=τ) | `pcl/clustering.py:cluster_features` |
| Momentumエンコーダ EMA更新 | `pcl/model.py:_update_momentum_encoder` |
| 負例キュー | `pcl/model.py:queue`, `_enqueue` |
| 階層的プロトタイプ (M粒度) | `cluster_features` の `k_list` |
| ウォームアップ | `train.py` の `warm_up` フラグ |
| t-SNE可視化 (Figure 4) | `visualize_tsne.py` / `train.py --tsne-freq` |

---

## 画像版との主な違い

| 項目 | 画像版 | 時系列版 |
|---|---|---|
| エンコーダ | ResNet (torchvision) | Transformer / CNN1D |
| Augmentation | RandomCrop, ColorJitter, etc. | Jitter, WindowSlicing |
| サンプル単位 | 画像1枚 | スライディングウィンドウ |
| t-SNEの色 | クラスラベル | 時刻位置 (0=始め → 1=終わり) |
| オプティマイザ | SGD | Adam |
| LRスケジュール | MultiStepLR (epoch 120, 160 で×0.1) | CosineAnnealingLR |
| デフォルトLR | `0.03` | `1e-4` |

---

## デバイス対応

`cuda → mps (Apple Silicon) → cpu` の順に自動選択。`pin_memory` はCUDAのみ有効（MPSは非対応）。
