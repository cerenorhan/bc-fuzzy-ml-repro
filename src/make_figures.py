
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ml = pd.read_csv("outputs/ML_summary.csv")
fz = pd.read_csv("outputs/Fuzzy_summary.csv")

def get_mean_std(df, metric):
    cols = df.columns.tolist()
    m = [c for c in cols if c.startswith(metric)]
    if len(m) < 2:
        raise ValueError(f"Cannot find mean/std columns for {metric}")
    return m[0], m[1]

for df in (ml, fz):
    df["target"] = df["target"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

all_df = pd.concat([ml, fz], ignore_index=True)

wf1_mean, wf1_sd = get_mean_std(all_df, "weighted_f1")
mf1_mean, mf1_sd = get_mean_std(all_df, "macro_f1")
bacc_mean, bacc_sd = get_mean_std(all_df, "balanced_acc")

for c in [wf1_mean, wf1_sd, mf1_mean, mf1_sd, bacc_mean, bacc_sd]:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

outdir = Path("outputs/figures")
outdir.mkdir(parents=True, exist_ok=True)

# --- Professional display labels ---
target_order = ["stage","diagnosis","laterality","ihc"]
target_label = {
    "stage": "Stage",
    "diagnosis": "Diagnosis",
    "laterality": "Laterality",
    "ihc": "IHC",
}

# Model display names (short but academic)
model_order = ["fuzzy","gb","logreg","logreg_bal","rf","svm_rbf","svm_rbf_bal"]
model_label = {
    "fuzzy": "Fuzzy logic",
    "gb": "Gradient boosting",
    "logreg": "Logistic regression",
    "logreg_bal": "Logistic regression (balanced)",
    "rf": "Random forest",
    "svm_rbf": "SVM (RBF)",
    "svm_rbf_bal": "SVM (RBF, balanced)",
}

def pretty_targets(xs):
    return [target_label.get(x, x) for x in xs]

def pretty_models(xs):
    return [model_label.get(x, x) for x in xs]

# ---------- Fig1: heatmap with numbers ----------
pivot = (all_df.pivot_table(index="target", columns="model", values=wf1_mean, aggfunc="first")
           .reindex(index=target_order, columns=model_order))
vals = pivot.values

plt.figure(figsize=(12.5, 3.8))
im = plt.imshow(vals, aspect="auto")
plt.xticks(range(pivot.shape[1]), pretty_models(pivot.columns), rotation=30, ha="right")
plt.yticks(range(pivot.shape[0]), pretty_targets(pivot.index))
plt.colorbar(im, label="Weighted F1-score (mean)")

for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        if np.isfinite(v):
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

plt.title("Model performance comparison (Weighted F1-score; mean across 5 repeats)")
plt.tight_layout()
plt.savefig(outdir / "Fig1_heatmap.png", dpi=300)
plt.savefig(outdir / "Fig1_heatmap.pdf")
plt.close()

# ---------- Fig2: grouped bars per target ----------
rows = []
for t in target_order:
    sub = all_df[all_df["target"]==t].copy()
    fuzzy = sub[sub["model"]=="fuzzy"].copy()
    nonfz = sub[sub["model"]!="fuzzy"].copy()

    best_w = nonfz.sort_values(wf1_mean, ascending=False).head(1)
    best_m = nonfz.sort_values(mf1_mean, ascending=False).head(1)

    best_w["pick"] = "Best (weighted F1-score)"
    best_m["pick"] = "Best (macro F1-score)"
    if len(fuzzy):
        fuzzy["pick"] = "Fuzzy logic"

    rows.append(pd.concat([best_w, best_m, fuzzy], ignore_index=True))
sel = pd.concat(rows, ignore_index=True)
pick_order = ["Best (weighted F1-score)", "Best (macro F1-score)", "Fuzzy logic"]
sel["pick"] = pd.Categorical(sel["pick"], categories=pick_order, ordered=True)

fig, ax = plt.subplots(figsize=(10.5, 4.2))
x = np.arange(len(target_order))
width = 0.25

for k, pick in enumerate(pick_order):
    part = sel[sel["pick"]==pick].set_index("target").reindex(target_order)
    y = part[wf1_mean].to_numpy()
    e = part[wf1_sd].to_numpy()
    ax.bar(x + (k-1)*width, y, width, yerr=e, capsize=3, label=pick)

ax.set_xticks(x)
ax.set_xticklabels(pretty_targets(target_order))
ax.set_ylabel("Weighted F1-score (mean ± SD)")
ax.set_title("Per endpoint: best machine-learning model vs fuzzy-logic baseline")
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(outdir / "Fig2_grouped_bars.png", dpi=300)
plt.savefig(outdir / "Fig2_grouped_bars.pdf")
plt.close()

# ---------- Fig3: scatter (balanced accuracy vs weighted F1-score) ----------
marker_map = {"stage":"o","diagnosis":"s","laterality":"^","ihc":"D"}

plt.figure(figsize=(7.3, 5.2))
for model in model_order:
    subm = all_df[all_df["model"]==model]
    for t in target_order:
        ss = subm[subm["target"]==t]
        if len(ss)==0:
            continue
        plt.scatter(ss[wf1_mean], ss[bacc_mean], marker=marker_map[t], s=70)

# build two legends: targets and model family (short)
from matplotlib.lines import Line2D
target_handles = [
    Line2D([0],[0], marker=marker_map[t], linestyle="None", markersize=8, label=target_label[t])
    for t in target_order
]
plt.legend(handles=target_handles, title="Endpoint", loc="lower right", frameon=True)

plt.xlabel("Weighted F1-score (mean)")
plt.ylabel("Balanced accuracy (mean)")
plt.title("Balanced accuracy vs Weighted F1-score (marker indicates endpoint)")
plt.tight_layout()
plt.savefig(outdir / "Fig3_scatter.png", dpi=300)
plt.savefig(outdir / "Fig3_scatter.pdf")
plt.close()

print("Wrote final figures (Fig1/Fig2/Fig3) to outputs/figures/")
