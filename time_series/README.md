# Prototypical Contrastive Learning (PCL) — 時系列版

論文 "Prototypical Contrastive Learning of Unsupervised Representations" (ICLR 2021) の時系列データへの適用。

---

## ファイル構成と役割

```
.
├── train.py            # メインの学習スクリプト
├── visualize_umap.py   # チェックポイントからUMAP画像を生成
├── requirements.txt    # 依存パッケージ
├── datasets/           # ベンチマークデータ (electricity, ETT-small など)
└── pcl/
    ├── model.py        # PCLモデル (Transformer/CNN/Chronos-Boltエンコーダ + Momentumエンコーダ + キュー)
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
| `encoder_q` | 学習対象のQueryエンコーダ |
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
| `chronos` (デフォルト) | `(B, C, L)` | Chronos-Bolt encoder → REG トークン → 2層MLP projection head | `(B, dim)` |
| `transformer` | `(B, C, L)` | Linear投影 → 位置エンコーディング → Transformer → 平均プーリング → Linear | `(B, dim)` |
| `cnn` | `(B, C, L)` | Conv1d×3 (stride=2) → AdaptiveAvgPool1d(1) → Linear | `(B, dim)` |

**`chronos` エンコーダの特徴**

- Amazon の事前学習済み Chronos-Bolt モデルを使用。バックボーンは完全凍結（勾配なし）。
- encoder_q と encoder_k でバックボーンを共有しているためメモリは1コピー分のみ。
- **学習されるのは projection head のみ**（2層MLP: `d_model → d_model → dim`）。バックボーンは更新されない。
- `--chronos-model` でモデルサイズを選択可能（下記参照）。デフォルトは `chronos-bolt-small`。

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

**`TimeSeriesDataset` の `--variables` モード**:

| モード | 説明 | サンプル shape | label |
|---|---|---|---|
| `univariate` | `--target-col` のカラムのみ | `(1, seq_len)` | 0 |
| `multivariate` | 全数値カラムを1サンプルとして扱う | `(C, seq_len)` | 0 |
| `individuals` (デフォルト) | 列ごとを独立した個体の単変量系列として扱う | `(1, seq_len)` | 個体ID |

**`individuals` モードの動作**:

```
electricity.csv: 321クライアント × 26,304タイムステップ

個体0の系列 → windows (1, seq_len) × n個、label=0
個体1の系列 → windows (1, seq_len) × n個、label=1
...
個体320の系列 → windows (1, seq_len) × n個、label=320
```

fMRI における「被験者ごとのROI信号」と同じ構造。個体差を反映したクラスタリングの検証に使う。

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

### `visualize_umap.py` — UMAP可視化

チェックポイントを渡すと、Momentumエンコーダが抽出した特徴量をUMAPで2次元に圧縮し、時系列上の時刻位置で色付けした散布図をPNGで保存する。

- 青 (0): 時系列の最初に位置するウィンドウ
- 赤 (1): 時系列の最後に位置するウィンドウ
- t-SNEより高速で、全サンプルを対象に実行できる

---

## ファイル間の依存関係

```
train.py
├── pcl/model.py       (PCLモデル: Transformer / CNN / ChronosBolt エンコーダ)
│   └── chronos        (--encoder chronos のとき: chronos-forecasting パッケージ)
├── pcl/loss.py        (ProtoNCE損失)
├── pcl/clustering.py  (E-step: k-means + φ)
│   └── faiss / sklearn
└── pcl/dataset.py     (TimeSeriesDataset, TwoViewTransform, IndexedDataset)

visualize_umap.py
├── pcl/model.py
├── pcl/dataset.py
├── umap-learn
└── matplotlib
```

---

## 実行方法

### インストール

```bash
pip3 install torch numpy pandas scikit-learn matplotlib umap-learn --break-system-packages
# Chronos-Bolt を使う場合
pip3 install chronos-forecasting --break-system-packages
```

---

### train.py のオプション一覧

重要度: ★★★ 必ず確認する　／　★★☆ 状況によって変える　／　★☆☆ 基本触らなくていい

#### データ

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--data-path` | `./datasets/electricity/electricity.csv` | 使うCSVファイルのパス |
| ★★★ | `--variables` | `individuals` | `univariate` / `multivariate` / `individuals` |
| ★★☆ | `--target-col` | `OT` | univariateのときに使うカラム名 |
| ★★☆ | `--seq-len` | `512` | 1サンプルのウィンドウ長（タイムステップ数）。Chronos-Boltのデフォルトコンテキスト長に合わせた値 |
| ★☆☆ | `--stride` | `seq-lenと同じ` | スライディングウィンドウのストライド（デフォルトは非重複ウィンドウ） |

#### エンコーダ

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★☆ | `--encoder` | `chronos` | `chronos` / `transformer` / `cnn` |
| ★★☆ | `--chronos-model` | `amazon/chronos-bolt-small` | `--encoder chronos` のみ有効 |
| ★☆☆ | `--d-model` | `64` | Transformerの埋め込み次元 |
| ★☆☆ | `--nhead` | `4` | Transformerのアテンションヘッド数 |
| ★☆☆ | `--num-layers` | `2` | Transformerのレイヤー数 |
| ★☆☆ | `--dim` | `128` | エンコーダ出力の特徴量次元数（全エンコーダ共通） |

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
| ★★★ | `--num-clusters` | `50 200 500` | クラスタ数（複数指定で階層的プロトタイプ） |
| ★★☆ | `--cluster-samples` | `50000` | E-stepで特徴抽出に使うサンプル数（0=全件）。大規模データでE-stepを高速化する |
| ★★☆ | `--queue-size` | `4096` | 負例キューのサイズ |
| ★★☆ | `--r` | `500` | 1ステップでサンプルする負例プロトタイプ数 |
| ★☆☆ | `--tau` | `0.1` | 温度パラメータ |
| ★☆☆ | `--momentum` | `0.999` | Momentumエンコーダの更新係数 |
| ★☆☆ | `--alpha` | `10.0` | 濃度推定の平滑化係数 |

#### 学習

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--epochs` | `200` | 総エポック数 |
| ★★★ | `--output-dir` | `./checkpoints` | チェックポイントの保存先 |
| ★★☆ | `--batch-size` | `128` | メモリが足りなければ減らす |
| ★★☆ | `--warm-up-epochs` | `10` | 最初のNエポックはクラスタリングなしで学習 |
| ★★☆ | `--resume` | （なし） | 途中から再開する場合にチェックポイントのパスを指定 |
| ★☆☆ | `--lr` | `1e-4` | 学習率（CosineAnnealingLRで減衰） |
| ★☆☆ | `--weight-decay` | `1e-4` | 重み減衰 |
| ★☆☆ | `--workers` | `2` | データロードの並列数 |
| ★☆☆ | `--save-freq` | `10` | Nエポックごとにチェックポイントを保存 |

---

### 実行例

#### 最小構成（デフォルト値のまま）

```bash
python3 train.py
```

electricity の321クライアントを individuals モードで Chronos-Bolt-small により100エポック学習する。

#### Chronos-Bolt エンコーダ（デフォルト）

```bash
python3 train.py
```

モデルサイズを変える場合:

```bash
python3 train.py --chronos-model amazon/chronos-bolt-base
```

| モデル名 | d_model | projection head (dim=128) |
|---|---|---|
| `amazon/chronos-bolt-mini` | 64 | 64→64→128 |
| `amazon/chronos-bolt-small` (���フォルト) | 256 | 256→256→128 |
| `amazon/chronos-bolt-base` | 768 | 768→768→128 |

#### ETTh1 単変量

```bash
python3 train.py \
  --data-path ./datasets/ETT-small/ETTh1.csv \
  --variables univariate \
  --target-col OT
```

#### 途中から再開

```bash
python3 train.py --resume ./checkpoints/pcl_epoch0100.pth
```

---

### visualize_umap.py のオプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | （`--checkpoints-dir` と排他） | 単一チェックポイントのパス |
| `--checkpoints-dir` | `./checkpoints` | ディレクトリ内の全チェックポイントを処理（引数省略時のデフォルト） |
| `--data-path` | `./datasets/electricity/electricity.csv` | データセットのパス |
| `--variables` | `individuals` | `univariate` / `multivariate` / `individuals` |
| `--seq-len` | チェックポイントから自動取得 | 省略時はチェックポイントに保存された値を使用 |
| `--stride` | チェックポイントから自動取得 | 省略時はチェックポイントに保存された値を使用 |
| `--output-dir` | `./umap_all` | 出力画像の保存先ディレクトリ |
| `--n-neighbors` | `15` | UMAPのn_neighbors |
| `--min-dist` | `0.1` | UMAPのmin_dist |

#### 実行例

```bash
# 引数なし（./checkpoints内の全チェックポイントを処理）
python3 visualize_umap.py

# 単一チェックポイント
python3 visualize_umap.py --checkpoint ./checkpoints/pcl_epoch0200.pth

# ディレクトリ指定
python3 visualize_umap.py --checkpoints-dir ./checkpoints
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

---

## 画像版との主な違い

| 項目 | 画像版 | 時系列版 |
|---|---|---|
| エンコーダ | ResNet (torchvision) | Transformer / CNN1D / Chronos-Bolt |
| Augmentation | RandomCrop, ColorJitter, etc. | Jitter, WindowSlicing |
| サンプル単位 | 画像1枚 | スライディングウィンドウ |
| データモード | クラスラベルあり | individuals（個体ごとの系列） |
| 可視化の色 | クラスラベル | 時刻位置 (0=始め → 1=終わり) |
| オプティマイザ | SGD | Adam |
| LRスケジュール | MultiStepLR (epoch 120, 160 で×0.1) | CosineAnnealingLR |
| デフォルトLR | `0.03` | `1e-4` |

---

## デバイス対応

`cuda → mps (Apple Silicon) → cpu` の順に自動選択。`pin_memory` はCUDAのみ有効（MPSは非対応）。
