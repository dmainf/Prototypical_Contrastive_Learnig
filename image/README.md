# Prototypical Contrastive Learning (PCL)

論文 "Prototypical Contrastive Learning of Unsupervised Representations" (ICLR 2021) の実装。

---

## ファイル構成と役割

```
.
├── train.py            # メインの学習スクリプト
├── visualize_tsne.py   # チェックポイントからt-SNE画像を生成
├── requirements.txt    # 依存パッケージ
├── datasets/           # ベンチマークデータ (ETT, Weather, ECL, Traffic 等)
└── pcl/
    ├── model.py        # PCLモデル (エンコーダ + Momentumエンコーダ + キュー)
    ├── loss.py         # ProtoNCE損失関数
    ├── clustering.py   # k-meansクラスタリング + 濃度推定 φ
    └── dataset.py      # データセット・Augmentationユーティリティ
```

---

## 各ファイルの詳細

### `pcl/model.py` — PCLモデル

**クラス: `PCL`**

| 要素 | 説明 |
|---|---|
| `encoder_q` | 学習対象のQueryエンコーダ (ResNet) |
| `encoder_k` | 重みが固定されたMomentumエンコーダ (EMA更新) |
| `queue` | インスタンスワイズ対照学習用の負例キュー (shape: `dim × queue_size`) |

- `forward(im_q, im_k)`: 2つのaugmented viewを受け取り、queryとkey特徴量を返す。Momentumエンコーダの重みをEMAで更新し、keyをキューに追加する
- `get_features(loader, device)`: Momentumエンコーダで全訓練データの特徴量を抽出する（クラスタリング用）

```
encoder_q ──→ L2-normalize ──→ q
encoder_k ──→ L2-normalize ──→ k ──→ queue に追加
(EMAで更新)
```

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
| `_proto_nce_single` | 1粒度分のプロトタイプ対照損失。r ≥ K−1 のとき全プロトタイプで正確なNCE、r < K−1 のとき randperm でr個サンプリング（重複なし） |
| `forward` | warm_up=True のときはInfoNCEのみ、Falseのときは両方の和を返す |

**負例サンプリングの扱い**: K が小さい（CIFAR-10: K=50 で r=49）場合は全プロトタイプを分母に使う完全なクロスエントロピーに切り替わる。これにより positive prototype の二重カウントを防ぐ。

---

### `pcl/clustering.py` — クラスタリング (E-step)

EMフレームワークのE-stepに対応。

| 関数 | 役割 |
|---|---|
| `_run_kmeans` | faiss (Linux) または sklearn KMeans (macOS) でk-meansを実行 |
| `_concentration` | 各プロトタイプの濃度 φ を計算 (論文 Eq.12)。numpy ベクトル化で高速 |
| `cluster_features` | k_listの各kに対してクラスタリング＋φ計算をまとめて実行 |

**濃度推定 φ (Eq.12)**:

```
φ_c = Σ||v'_z - c||₂ / (Z * log(Z + α))
```

- クラスタが疎 → φ大 → 類似度をダウンスケール（崩壊防止）
- クラスタが密 → φ小 → 類似度をアップスケール
- 最後に `mean(φ) = τ` になるよう正規化
- **φ計算後、セントロイドをL2正規化して返す** (論文 Sec.3.2: "apply l₂-normalization to both v and c")

戻り値は `(centroids, assignments, phi)` のリスト（粒度M個分）。
- `centroids`: L2正規化済み (k, D)
- `assignments`: 全訓練サンプルのクラスタ割当 (N,)
- `phi`: 正規化済み濃度 (k,)

---

### `pcl/dataset.py` — データセットユーティリティ

| クラス/関数 | 役割 |
|---|---|
| `TwoViewTransform` | 同じ画像に確率的augmentationを2回適用し、2つのviewを生成 |
| `IndexedDataset` | 既存Datasetをラップし、`(data, label, index)` を返すようにする |
| `imagenet_train_transform` | ImageNet用augmentation (RandomCrop, ColorJitter, Grayscale 等) |
| `cifar10_train_transform` | CIFAR-10用augmentation |
| `*_eval_transform` | クラスタリング・評価用 (augmentationなし) |

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
    for (im_q, im_k), _, indices in train_loader:
        q, k = model(im_q, im_k)
        loss = ProtoNCELoss(q, k, queue, cluster_results, indices, warm_up)
        loss.backward()
        optimizer.step()
```

| フェーズ | エポック | 内容 |
|---|---|---|
| warm-up | 0 〜 warm_up_epochs-1 | InfoNCEのみ（クラスタリングなし） |
| PCL | warm_up_epochs 〜 end | E-step (k-means) + M-step (ProtoNCE) |

LRスケジュール: epoch 120, 160 で×0.1（論文§4.1に従う）

---

### `visualize_tsne.py` — t-SNE可視化（論文 Figure 4 相当）

チェックポイントを渡すと、Momentumエンコーダが抽出した特徴量をt-SNEで2次元に圧縮し、クラスごとに色分けした散布図を PNG で保存する。

- 学習前: クラスが混在した状態
- 学習後: クラスごとに分離したクラスタが現れる

`train.py` の `--tsne-freq` でも同じ処理をエポックごとに自動実行できる。

---

## ファイル間の依存関係

```
train.py
├── pcl/model.py       (PCLモデル)
│   └── torchvision.models (ResNetバックボーン)
├── pcl/loss.py        (ProtoNCE損失)
├── pcl/clustering.py  (E-step: k-means + φ)
│   └── faiss / sklearn
└── pcl/dataset.py     (TwoViewTransform, IndexedDataset)

visualize_tsne.py
├── pcl/dataset.py
├── sklearn.manifold.TSNE
└── matplotlib
```

---

## 実行方法

### インストール

```bash
pip3 install torch torchvision faiss-cpu numpy scikit-learn matplotlib
```

---

### コマンドの読み方

すべての設定は `--オプション名 値` の形式でコマンドラインから指定する。
**省略した場合はデフォルト値が使われる**（後述のオプション一覧参照）。

```
python3 train.py --dataset cifar10 --epochs 200 --batch-size 256
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 これらはすべてオプション。書かなければデフォルト値が適用される。
```

`python3 train.py` だけで実行した場合は、全オプションがデフォルト値になる。
→ CIFAR-10 を `./data` にダウンロードして ResNet18 で200エポック学習する。

---

### train.py のオプション一覧

重要度: ★★★ 必ず確認する　／　★★☆ 状況によって変える　／　★☆☆ 基本触らなくていい

| 重要度 | オプション | デフォルト | 説明 |
|:---:|---|---|---|
| ★★★ | `--dataset` | `cifar10` | 使うデータセット。`cifar10` または `imagenet` |
| ★★★ | `--data-path` | `./data` | データの場所。CIFAR-10は自動DL。ImageNetは手動で用意 |
| ★★★ | `--arch` | `resnet18` | エンコーダ。論文はresnet50。動作確認はresnet18で十分 |
| ★★★ | `--num-clusters` | `50 200 500` | クラスタ数。PCLの核心部分。データセットサイズより小さい値にする必要がある。ImageNetなら `25000 50000 100000` |
| ★★★ | `--epochs` | `200` | 総エポック数 |
| ★★★ | `--output-dir` | `./checkpoints` | チェックポイントとt-SNE画像の保存先 |
| ★★☆ | `--batch-size` | `256` | メモリが足りなければ減らす |
| ★★☆ | `--queue-size` | `65536` | データセットサイズより大きくしても意味がない |
| ★★☆ | `--r` | `16000` | `--num-clusters` の最小値より小さくする必要がある |
| ★★☆ | `--warm-up-epochs` | `20` | 最初のNエポックはクラスタリングなしで学習 |
| ★★☆ | `--tsne-freq` | `0` | Nエポックごとにt-SNE画像を保存。`0`で無効 |
| ★★☆ | `--resume` | （なし） | 途中から再開する場合にチェックポイントのパスを指定 |
| ★☆☆ | `--use-mlp` | `False` | PCL v2にする。値不要、`--use-mlp` と書くだけ |
| ★☆☆ | `--tau` | `0.1` | 温度パラメータ |
| ★☆☆ | `--momentum` | `0.999` | Momentumエンコーダの更新係数 |
| ★☆☆ | `--lr` | `0.03` | 学習率。epoch 120・160で自動的に×0.1される |
| ★☆☆ | `--dim` | `128` | 特徴量の次元数 |
| ★☆☆ | `--alpha` | `10.0` | 濃度推定の平滑化係数 |
| ★☆☆ | `--weight-decay` | `1e-4` | 重み減衰 |
| ★☆☆ | `--workers` | `4` | データロードの並列数 |
| ★☆☆ | `--save-freq` | `10` | Nエポックごとにチェックポイントを保存 |
| ★☆☆ | `--tsne-classes` | `10` | t-SNEで描画するクラス数 |
| ★☆☆ | `--tsne-samples` | `200` | t-SNEのクラスあたりサンプル数 |

---

### 実行例

#### 最小構成（デフォルト値のまま）

```bash
python3 train.py
```

CIFAR-10を自動ダウンロードして、ResNet18で200エポック学習する。

#### CIFAR-10（設定を変える場合）

```bash
python3 train.py \
  --dataset cifar10 \
  --data-path ./data \
  --arch resnet18 \
  --num-clusters 50 200 500 \
  --queue-size 4096 \
  --r 500 \
  --epochs 200 \
  --batch-size 256 \
  --output-dir ./checkpoints \
  --tsne-freq 20
```

`--num-clusters 50 200 500` はクラスタ数を3段階（50・200・500）に設定している。
CIFAR-10は50,000サンプルしかないので、デフォルトの25000〜100000ではなく小さい値にする必要がある。

#### ImageNet（論文と同じ設定）

```bash
python3 train.py \
  --dataset imagenet \
  --data-path /path/to/imagenet \
  --arch resnet50 \
  --num-clusters 25000 50000 100000 \  # ImageNetのときはここを変える
  --queue-size 65536 \
  --r 16000 \
  --epochs 200 \
  --batch-size 256 \
  --output-dir ./checkpoints
```

PCL v2にする場合は `--use-mlp` を追加する。

#### 途中から再開

```bash
python3 train.py --resume ./checkpoints/pcl_epoch0100.pth
```

チェックポイントに引数が保存されているが、他のオプションを追加指定することもできる。

---

### visualize_tsne.py のオプション一覧

学習済みチェックポイントからt-SNE画像を生成する単体スクリプト。

| オプション | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | （必須） | 読み込むチェックポイントのパス |
| `--dataset` | `cifar10` | `cifar10` または `imagenet` |
| `--data-path` | `./data` | データセットのパス |
| `--arch` | `resnet18` | 学習時と同じアーキテクチャを指定する |
| `--num-classes` | `10` | 描画するクラス数 |
| `--samples-per-class` | `200` | クラスあたりのサンプル数 |
| `--perplexity` | `30.0` | t-SNEのperplexityパラメータ |
| `--output` | `./tsne.png` | 出力画像のパス |

#### 実行例

```bash
python3 visualize_tsne.py \
  --checkpoint ./checkpoints/pcl_epoch0200.pth \
  --dataset cifar10 \
  --data-path ./data \
  --arch resnet18 \
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
| セントロイドL2正規化 (Sec.3.2) | `pcl/clustering.py:cluster_features` (φ計算後に実施) |
| Momentumエンコーダ EMA更新 | `pcl/model.py:_update_momentum_encoder` |
| 負例キュー | `pcl/model.py:queue`, `_enqueue` |
| 階層的プロトタイプ (M粒度) | `cluster_features` の `k_list` |
| ウォームアップ | `train.py` の `warm_up` フラグ |
| t-SNE可視化 (Figure 4) | `visualize_tsne.py` / `train.py --tsne-freq` |

---

## デバイス対応

`cuda → mps (Apple Silicon) → cpu` の順に自動選択。`pin_memory` はCUDAのみ有効（MPSは非対応）。
