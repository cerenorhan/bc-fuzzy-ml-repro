from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_ORDER = ["stage", "diagnosis", "laterality", "ihc"]
TARGET_LABEL = {
    "stage": "Stage",
    "diagnosis": "Diagnosis",
    "laterality": "Laterality",
    "ihc": "IHC",
}

MODEL_ORDER = ["expert_fuzzy", "fuzzy", "gb", "logreg", "logreg_bal", "rf", "svm", "svm_bal"]
MODEL_LABEL = {
    "expert_fuzzy": "Expert-rule fuzzy",
    "fuzzy": "Data-driven fuzzy",
    "gb": "Gradient boosting",
    "logreg": "Logistic regression",
    "logreg_bal": "Logistic regression (balanced)",
    "rf": "Random forest",
    "svm": "SVM (RBF)",
    "svm_bal": "SVM (RBF, balanced)",
}

MODEL_ALIASES = {
    "expert_fuzzy": "expert_fuzzy",
    "fuzzy": "fuzzy",
    "gb": "gb",
    "gradient_boosting": "gb",
    "logreg": "logreg",
    "logistic_regression": "logreg",
    "logreg_bal": "logreg_bal",
    "logistic_regression_balanced": "logreg_bal",
    "rf": "rf",
    "random_forest": "rf",
    "svm": "svm",
    "svm_rbf": "svm",
    "svm_bal": "svm_bal",
    "svm_rbf_bal": "svm_bal",
    "svm_rbf_balanced": "svm_bal",
}

def metric_cols(df, base):
    if base in df.columns and f"{base}.1" in df.columns:
        return base, f"{base}.1"
    if f"{base}_mean" in df.columns and f"{base}_std" in df.columns:
        return f"{base}_mean", f"{base}_std"
    raise KeyError(f"Could not find summary columns for {base}. Found: {list(df.columns)}")

def load_summary():
    frames = []
    for p in ["outputs/Fuzzy_summary.csv", "outputs/ML_summary.csv"]:
        fp = Path(p)
        if fp.exists():
            frames.append(pd.read_csv(fp))
    if not frames:
        raise FileNotFoundError("No summary files found")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(axis=0, how="all").copy()

    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().map(lambda x: MODEL_ALIASES.get(x, x))

    df = df[df["target"].isin(TARGET_ORDER)].copy()
    return df

def load_detailed():
    frames = []
    for p in ["outputs/Fuzzy_detailed.csv", "outputs/ML_detailed.csv", "outputs/ExpertFuzzy_detailed.csv"]:
        fp = Path(p)
        if fp.exists():
            frames.append(pd.read_csv(fp))
    if not frames:
        raise FileNotFoundError("No detailed files found")

    df = pd.concat(frames, ignore_index=True)
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip().map(lambda x: MODEL_ALIASES.get(x, x))
    return df

def fig1_heatmap():
    df = load_summary()
    wf1_mean, _ = metric_cols(df, "weighted_f1")

    piv = (
        df.pivot_table(index="target", columns="model", values=wf1_mean, aggfunc="first")
          .reindex(index=TARGET_ORDER, columns=MODEL_ORDER)
    )

    fig, ax = plt.subplots(figsize=(15, 6))
    arr = piv.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="Greens", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODEL_ORDER], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(TARGET_ORDER)))
    ax.set_yticklabels([TARGET_LABEL[t] for t in TARGET_ORDER])

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=10)

    ax.set_title("Model performance comparison (Weighted F1-score; mean across 5 repeats)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Weighted F1-score (mean)")

    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig1_heatmap.png", dpi=300)
    fig.savefig(OUTDIR / "Fig1_heatmap.pdf")
    plt.close(fig)

def fig2_grouped_bars():
    df = load_summary()
    wf1_mean, wf1_std = metric_cols(df, "weighted_f1")

    ml_models = ["gb", "logreg", "logreg_bal", "rf", "svm", "svm_bal"]
    rows = []

    for tgt in TARGET_ORDER:
        sub = df[df["target"] == tgt].copy()
        if sub.empty:
            continue

        expert = sub[sub["model"] == "expert_fuzzy"]
        datafz = sub[sub["model"] == "fuzzy"]
        ml = sub[sub["model"].isin(ml_models)].copy()

        if expert.empty or datafz.empty or ml.empty:
            continue

        best_ml = ml.sort_values(wf1_mean, ascending=False).iloc[0]

        rows.append({
            "target": tgt,
            "expert_mean": float(expert.iloc[0][wf1_mean]),
            "expert_std": float(expert.iloc[0][wf1_std]),
            "data_mean": float(datafz.iloc[0][wf1_mean]),
            "data_std": float(datafz.iloc[0][wf1_std]),
            "ml_mean": float(best_ml[wf1_mean]),
            "ml_std": float(best_ml[wf1_std]),
            "ml_name": MODEL_LABEL[best_ml["model"]],
        })

    x = np.arange(len(rows))
    w = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, [r["expert_mean"] for r in rows], w,
           yerr=[r["expert_std"] for r in rows], capsize=4,
           label="Expert-rule fuzzy")
    ax.bar(x, [r["data_mean"] for r in rows], w,
           yerr=[r["data_std"] for r in rows], capsize=4,
           label="Data-driven fuzzy")
    ax.bar(x + w, [r["ml_mean"] for r in rows], w,
           yerr=[r["ml_std"] for r in rows], capsize=4,
           label="Best ML")

    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABEL[r["target"]] for r in rows])
    ax.set_ylabel("Weighted F1-score (mean ± SD)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per endpoint: expert-rule fuzzy, data-driven fuzzy, and best machine-learning model")
    ax.legend()

    for i, r in enumerate(rows):
        ax.text(i + w + 0.02, r["ml_mean"], r["ml_name"], va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig2_grouped_bars.png", dpi=300)
    fig.savefig(OUTDIR / "Fig2_grouped_bars.pdf")
    plt.close(fig)

def fig4_metric_distributions():
    df = load_detailed()

    if "weighted_f1" not in df.columns:
        raise KeyError(f"'weighted_f1' not found in detailed tables. Found: {list(df.columns)}")

    panel_labels = ["A)", "B)", "C)", "D)"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.ravel()

    for i, (ax, tgt) in enumerate(zip(axes, TARGET_ORDER)):
        sub = df[df["target"] == tgt].copy()
        present = [m for m in MODEL_ORDER if m in set(sub["model"])]
        if not present:
            ax.axis("off")
            continue

        vals_list = [sub.loc[sub["model"] == m, "weighted_f1"].astype(float).values for m in present]

        parts = ax.violinplot(
            vals_list,
            positions=np.arange(len(present)),
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.25)

        for j, m in enumerate(present):
            vals = sub.loc[sub["model"] == m, "weighted_f1"].astype(float).values
            x = np.repeat(j, len(vals)).astype(float)
            if len(vals) > 1:
                x += np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(x, vals, s=18, alpha=0.85)
            ax.hlines(np.mean(vals), j - 0.18, j + 0.18, linewidth=1.8)

        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([MODEL_LABEL[m] for m in present], rotation=30, ha="right")
        ax.set_title(TARGET_LABEL[tgt])
        ax.set_ylabel("Weighted F1-score")
        label_x = -0.10 if i % 2 == 0 else -0.03
        ax.text(label_x, 1.03, panel_labels[i],
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, fontweight="bold")

    fig.suptitle("Fold/seed-wise weighted F1-score distributions across repeated splits")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig4_metric_distributions.png", dpi=300)
    fig.savefig(OUTDIR / "Fig4_metric_distributions.pdf")
    plt.close(fig)

def fig15_workflow_schematic():
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    boxes = {
        "input":  (0.03, 0.36, 0.14, 0.18, "Input dataset\nclinical + tumor\nfeatures"),
        "prep":   (0.21, 0.36, 0.14, 0.18, "Preprocessing\nand target definition"),
        "split":  (0.39, 0.36, 0.14, 0.18, "Repeated stratified\n80/20 splits"),
        "expert": (0.62, 0.62, 0.17, 0.14, "Expert-rule fuzzy"),
        "datafz": (0.62, 0.40, 0.17, 0.14, "Data-driven fuzzy\nWang–Mendel + Mamdani"),
        "ml":     (0.62, 0.18, 0.17, 0.14, "Machine learning\nLR / SVM / RF / GB"),
        "eval":   (0.84, 0.36, 0.17, 0.18, "Evaluation\nAccuracy / F1 / BAcc\nROC / PR / confusion"),
        "unk":    (0.43, 0.02, 0.28, 0.12, "Unknown-Stage\nsecondary analyses\ninitial prediction / self-training / label propagation"),
    }

    for _, (x, y, w, h, label) in boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.8))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=10)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.6))

    arrow(0.17, 0.45, 0.21, 0.45)
    arrow(0.35, 0.45, 0.39, 0.45)

    arrow(0.53, 0.45, 0.62, 0.69)
    arrow(0.53, 0.45, 0.62, 0.47)
    arrow(0.53, 0.45, 0.62, 0.25)
    arrow(0.46, 0.36, 0.57, 0.08)

    arrow(0.79, 0.69, 0.84, 0.45)
    arrow(0.79, 0.47, 0.84, 0.45)
    arrow(0.79, 0.25, 0.84, 0.45)

    ax.set_title("Overview of the expert-rule fuzzy, data-driven fuzzy, and machine-learning analysis workflow", fontsize=16)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.png", dpi=300)
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig1_heatmap()
    fig2_grouped_bars()
    fig4_metric_distributions()
    fig15_workflow_schematic()
    print("Fixed figures regenerated: Fig1, Fig2, Fig4, Fig15")
