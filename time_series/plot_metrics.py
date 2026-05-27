import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = "./align_uniform"

def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    epoch_cols = [c for c in rows[0] if c.startswith("epoch_")]
    epochs = [int(c.replace("epoch_", "")) for c in epoch_cols]
    data = {}
    for row in rows:
        data[row["mode"]] = [float(row[c]) for c in epoch_cols]
    return epochs, data

def plot(epochs, data, ylabel, title, out_path, lower_is_better=True):
    fig, ax = plt.subplots(figsize=(9, 5))
    styles = ["-o", "-s", "-^", "-D"]
    for (mode, vals), style in zip(data.items(), styles):
        ax.plot(epochs, vals, style, label=mode, markersize=4, linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

epochs_a, align = read_csv(f"{DIR}/alignment.csv")
epochs_u, unif  = read_csv(f"{DIR}/uniformity.csv")

plot(epochs_a, align,
     ylabel="Mean L2 distance (↓ better)",
     title="Alignment over Training",
     out_path=f"{DIR}/plot_alignment.png")

plot(epochs_u, unif,
     ylabel="Uniformity loss (↓ better)",
     title="Uniformity over Training",
     out_path=f"{DIR}/plot_uniformity.png")
