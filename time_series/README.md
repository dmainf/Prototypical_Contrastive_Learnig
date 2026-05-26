# Prototypical Contrastive Learning (PCL) — 時系列版

論文 "Prototypical Contrastive Learning of Unsupervised Representations" (ICLR 2021) の時系列データへの適用。

---

## ファイル構成と役割

```
.
├── train.py                              # メインの学習スクリプト
├── visualize_umap.py                    # チェックポイントからUMAP画像を生成
├── visualize_alignment_uniformity.py   # Alignment & Uniformity 可視化 (Wang & Isola, ICML 2020)
├── requirements.txt                     # 依存パッケージ
├── datasets/           # ベンチマークデータ (electricity, ETT-small など)
└── pcl/
    ├── model.py        # PCLモデル (Transformer/CNN/Chronos-Boltエンコーダ + Momentumエンコーダ + キュー)
    ├── loss.py         # HybridContrastiveLoss（InfoNCE / Align&Uniform × ProtoNCE の4モード）
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

- `forward(x_q, x_k)`: 2つのaugmented viewを受け取り、queryとkey特徴量を返す。Momentumエンコーダの重みをEMAで更新する。キューへの追加は損失計算後に `enqueue(k)` を呼び出すことで行う
- `enqueue(keys)`: keyをキューに追加する。`train.py` で損失計算・`optimizer.step()` の後に呼び出す
- `get_features(loader, device)`: Momentumエンコーダで全訓練データの特徴量を抽出する（クラスタリング用）

```
encoder_q ──→ L2-normalize ──→ q ──┐
                                    ├── loss() ── optimizer.step() ── enqueue(k)
encoder_k ──→ L2-normalize ──→ k ──┘
(EMAで更新)
```

**エンコーダ (`--encoder`)**

| エンコーダ | 入力 | 処理 | 出力 |
|---|---|---|---|
| `chronos` (デフォルト) | `(B, C, L)` | Chronos-Bolt encoder → 時間軸 Mean Pooling → チャンネル Flatten → 2層MLP projection head | `(B, dim)` |
| `transformer` | `(B, C, L)` | Linear投影 → 位置エンコーディング → Transformer → 平均プーリング → Linear | `(B, dim)` |
| `cnn` | `(B, C, L)` | Conv1d×3 (stride=2) → AdaptiveAvgPool1d(1) → Linear | `(B, dim)` |

**`chronos` エンコーダの特徴**

- Amazon の事前学習済み Chronos-Bolt モデルを使用。バックボーンは完全凍結（勾配なし）。
- encoder_q と encoder_k でバックボーンを共有しているためメモリは1コピー分のみ。
- **学習されるのは projection head のみ**（2層MLP: `d_model * C → d_model → dim`）。バックボーンは更新されない。
- `--chronos-model` でモデルサイズを選択可能（下記参照）。デフォルトは `chronos-bolt-small`。

---

### `pcl/loss.py` — 損失関数

**クラス: `HybridContrastiveLoss`**

`--base-loss`（ベース損失の種類）と `--use-proto`（ProtoNCEの有無）を直交する2軸として管理し、以下の4モードを1クラスで実現する。

| `--base-loss` | `--use-proto` | 損失の構成 |
|---|---|---|
| `infonce` | なし | pure InfoNCE（MoCo準拠） |
| `infonce` | あり | PCL（InfoNCE + ProtoNCE）← 従来手法 |
| `align_uniform` | なし | pure Align & Uniform |
| `align_uniform` | あり | ハイブリッド（Align & Uniform + ProtoNCE）← 提案手法 |

**ベース損失の数式**

```
# InfoNCE
L_InfoNCE = CrossEntropy([q・k/τ, q・k⁻/τ, ...])

# Align & Uniform (Wang & Isola, ICML 2020)
L_align   = E_{(x,y)~p_pos} [ ||f(x) - f(y)||_2^alpha ]
L_uniform = log E_{x,y~p_data} [ exp(-t * ||f(x) - f(y)||_2^2) ]
base_loss = L_align + λ * (L_uniform(q) + L_uniform(k)) / 2
```

**ProtoNCE 項（`--use-proto` のとき加算）**

```
L_ProtoNCE = (1/M) * Σ_m CrossEntropy(q・C_m/φ_m)
total_loss = base_loss + L_ProtoNCE
```

| メソッド | 役割 |
|---|---|
| `_align` / `_uniform` | Alignment・Uniformity の各項を計算 |
| `_info_nce` | MoCo方式のInfoNCE。qとkの正例ペア vs キュー内の負例 |
| `_proto_nce_single` | 1粒度分のプロトタイプ対照損失（PCL Eq.10）。全Kプロトタイプとの類似度を行列積で一括計算し、割当クラスタIDをターゲットとしたクロスエントロピー |
| `forward` | `is_warmup=True` のときはProtoNCEをスキップ。`breakdown` dict にベース損失・各項の値を格納して返す |

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
| `Jitter` | ガウスノイズを付加。周期性・位相を保持 |
| `Scaling` | 信号全体にランダムなスカラーを乗算。振幅依存性を排除し波形ダイナミクスに着目させる |
| `ContinuousMasking` | 系列中間の連続区間をゼロでマスク。時間軸伸縮なしで周波数特性を完全保持 |
| `TimeSeriesAugmentation` | Scaling + ContinuousMasking + Jitter の組み合わせ |
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
    # E-step: --use-proto のときのみ実行
    if use_proto and epoch >= warm_up_epochs:
        features = model.get_features(cluster_loader)
        cluster_results = cluster_features(features, K_list)

    # M-step: HybridContrastiveLoss でエンコーダを更新
    for (x_q, x_k), _, indices in train_loader:
        q, k = model(x_q, x_k)
        loss, breakdown = criterion(q, k, queue, cluster_results, indices, is_warmup)
        loss.backward()
        optimizer.step()
        if base_loss == "infonce":
            model.enqueue(k)   # キュー更新は InfoNCE のときのみ
```

| フェーズ | エポック | 内容 |
|---|---|---|
| warm-up | 0 〜 warm_up_epochs-1 | ベース損失のみ（ProtoNCEなし） |
| 通常 | warm_up_epochs 〜 end | `--use-proto` のとき E-step + ベース損失 + ProtoNCE |

- `--use-proto` なしのときはE-stepが一切走らず、warm-upの概念も実質不要
- キュー（`model.queue`）は `--base-loss infonce` のときのみ更新・使用される

LRスケジュール: CosineAnnealingLR（最終エポックまで余弦的に減衰）

---

### `visualize_umap.py` — UMAP可視化

チェックポイントを渡すと、Momentumエンコーダが抽出した特徴量をUMAPで2次元に圧縮し、時系列上の時刻位置で色付けした散布図をPNGで保存する。

- 青 (0): 時系列の最初に位置するウィンドウ
- 赤 (1): 時系列の最後に位置するウィンドウ
- t-SNEより高速で、全サンプルを対象に実行できる

---

### `visualize_alignment_uniformity.py` — Alignment & Uniformity 可視化

Wang & Isola (ICML 2020) のFigure 3に倣い、表現空間の品質を2つの指標で可視化する。

- **左列 — Alignment**: 正例ペアの L2 距離ヒストグラム。値が小さいほど同一サンプルの2つのaugmented viewが近い表現を持つことを示す
- **右列 — Uniformity**: 特徴量をPCAで2次元に圧縮し単位円に正規化した後、KDEで分布を可視化。特徴量が球面上に均一に分散しているほど情報が豊か

出力: `{checkpoints_dir}/align_uniform/au_epoch****.png`（エポックごとに1枚）

---

## ファイル間の依存関係

```
train.py
├── pcl/model.py       (PCLモデル: Transformer / CNN / ChronosBolt エンコーダ)
│   └── chronos        (--encoder chronos のとき: chronos-forecasting パッケージ)
├── pcl/loss.py        (HybridContrastiveLoss)
├── pcl/clustering.py  (E-step: k-means + φ)
│   └── faiss / sklearn
└── pcl/dataset.py     (TimeSeriesDataset, TwoViewTransform, IndexedDataset)

visualize_umap.py
├── pcl/model.py
├── pcl/dataset.py
├── umap-learn
└── matplotlib

visualize_alignment_uniformity.py
├── pcl/model.py
├── pcl/dataset.py
├── scikit-learn   (PCA)
├── scipy          (gaussian_kde)
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
| ★☆☆ | `--scale-range` | `0.8 1.2` | Scalingの振幅スケール範囲（LOW HIGH の2値） |
| ★☆☆ | `--mask-ratio` | `0.2` | ContinuousMaskingでマスクする区間長の割合（0〜1） |
| ★☆☆ | `--no-jitter` | （なし） | Jitterを無効化 |
| ★☆☆ | `--no-scaling` | （なし） | Scalingを無効化 |
| ★☆☆ | `--no-masking` | （なし） | ContinuousMaskingを無効化 |

#### 損失関数（2軸で独立制御）

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--base-loss` | `align_uniform` | ベース損失: `infonce`（MoCo方式） / `align_uniform`（Wang & Isola, ICML 2020） |
| ★★★ | `--use-proto` | （なし） | フラグを立てると ProtoNCE をベース損失に加算（EMクラスタリングを有効化） |
| ★☆☆ | `--align-alpha` | `2.0` | L_align のべき乗パラメータ α（`align_uniform` のみ有効） |
| ★☆☆ | `--uniform-t` | `2.0` | L_uniform のガウスカーネルパラメータ t（`align_uniform` のみ有効） |
| ★☆☆ | `--lam` | `1.0` | L_uniform の重み λ: `L_align + λ * L_uniform`（`align_uniform` のみ有効） |

#### 対照学習・クラスタリング

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--num-clusters` | `50 200 500` | クラスタ数（複数指定で階層的プロトタイプ）。`--use-proto` のときのみ使用 |
| ★★☆ | `--queue-size` | `4096` | 負例キューのサイズ。`--base-loss infonce` のときのみ使用 |
| ★☆☆ | `--tau` | `0.1` | 温度パラメータ（InfoNCE・ProtoNCEの softmax スケール） |
| ★☆☆ | `--momentum` | `0.999` | Momentumエンコーダの更新係数 |
| ★☆☆ | `--alpha` | `10.0` | 濃度推定の平滑化係数（`--use-proto` のときのみ有効） |

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

#### 4つの損失モード

```bash
# 1. pure Align & Uniform（デフォルト）
python3 train.py

# 2. ハイブリッド: Align & Uniform + ProtoNCE（提案手法）
python3 train.py --use-proto

# 3. pure InfoNCE（MoCo準拠）
python3 train.py --base-loss infonce

# 4. 従来の PCL（InfoNCE + ProtoNCE）
python3 train.py --base-loss infonce --use-proto
```

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

### visualize_alignment_uniformity.py のオプション一覧

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoints-dir` | `./checkpoints_s128` | チェックポイントを含むディレクトリ |
| `--data-path` | `./datasets/electricity/electricity.csv` | データセットのパス |
| `--variables` | `individuals` | `univariate` / `multivariate` / `individuals` |
| `--target-col` | `OT` | univariateのときに使うカラム名 |
| `--output-dir` | `./align_uniform` | 出力画像の保存先ディレクトリ |
| `--epochs` | （全エポック） | 可視化するエポック番号を空白区切りで指定 |
| `--n-pairs` | `3000` | Alignment計算に使う正例ペア数 |
| `--n-feats` | `5000` | Uniformity可視化に使うサンプル数 |
| `--workers` | `0` | データロードの並列数 |

#### 実行例

```bash
# ディレクトリ内の全チェックポイントを処理
python3 visualize_alignment_uniformity.py --checkpoints-dir ./checkpoints_s128

# 特定エポックのみ
python3 visualize_alignment_uniformity.py --checkpoints-dir ./checkpoints_s128 --epochs 0 50 100
```

---

## 論文との対応

| 論文の要素 | 実装箇所 |
|---|---|
| EMフレームワーク | `train.py` の学習ループ |
| InfoNCE（MoCo方式）| `pcl/loss.py:HybridContrastiveLoss`（`--base-loss infonce`） |
| ProtoNCE損失 Eq.10 | `pcl/loss.py:HybridContrastiveLoss`（`--use-proto`） |
| Alignment + Uniformity | `pcl/loss.py:HybridContrastiveLoss`（`--base-loss align_uniform`） |
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
| Augmentation | RandomCrop, ColorJitter, etc. | Jitter, Scaling, ContinuousMasking |
| サンプル単位 | 画像1枚 | スライディングウィンドウ |
| データモード | クラスラベルあり | individuals（個体ごとの系列） |
| 可視化の色 | クラスラベル | 時刻位置 (0=始め → 1=終わり) |
| オプティマイザ | SGD | Adam |
| LRスケジュール | MultiStepLR (epoch 120, 160 で×0.1) | CosineAnnealingLR |
| デフォルトLR | `0.03` | `1e-4` |

---

## デバイス対応

`cuda → mps (Apple Silicon) → cpu` の順に自動選択。`pin_memory` はCUDAのみ有効（MPSは非対応）。
