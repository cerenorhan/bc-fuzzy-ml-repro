
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ml = pd.read_csv("outputs/metrics_test_ml_80_20_repeats_summary_v2metrics.csv")
fz = pd.read_csv("outputs/metrics_test_fuzzy_80_20_repeats_summary_v2metrics.csv")

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

targets = ["stage2","diagnosis","laterality","ihc"]
models_order = ["fuzzy","gb","logreg","logreg_bal","rf","svm_rbf","svm_rbf_bal"]

# ---------- Fig1b: heatmap with numbers ----------
pivot = (all_df.pivot_table(index="target", columns="model", values=wf1_mean, aggfunc="first")
           .reindex(index=targets, columns=models_order))
vals = pivot.values

plt.figure(figsize=(11, 3.6))
im = plt.imshow(vals, aspect="auto")
plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=35, ha="right")
plt.yticks(range(pivot.shape[0]), pivot.index)
plt.colorbar(im, label="weighted-F1 (mean)")

# annotate cells
for i in range(vals.shape[0]):
    for j in range(vals.shape[1]):
        v = vals[i, j]
        if np.isfinite(v):
            plt.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

plt.title("Model comparison (weighted-F1 mean ± across 5 repeats; stratified 80/20)")
plt.tight_layout()
plt.savefig(outdir / "Fig1b_heatmap_weightedF1_annot.png", dpi=300)
plt.savefig(outdir / "Fig1b_heatmap_weightedF1_annot.pdf")
plt.close()

# ---------- Fig2b: grouped bars per target ----------
# pick: best by weighted-F1, best by macro-F1, plus fuzzy
rows = []
for t in targets:
    sub = all_df[all_df["target"]==t].copy()
    fuzzy = sub[sub["model"]=="fuzzy"].copy()
    nonfz = sub[sub["model"]!="fuzzy"].copy()

    best_w = nonfz.sort_values(wf1_mean, ascending=False).head(1)
    best_m = nonfz.sort_values(mf1_mean, ascending=False).head(1)

    best_w["pick"] = "best_weightedF1"
    best_m["pick"] = "best_macroF1"
    if len(fuzzy):
        fuzzy["pick"] = "fuzzy"

    rows.append(pd.concat([best_w, best_m, fuzzy], ignore_index=True))
sel = pd.concat(rows, ignore_index=True)

# make a consistent order
pick_order = ["best_weightedF1","best_macroF1","fuzzy"]
sel["pick"] = pd.Categorical(sel["pick"], categories=pick_order, ordered=True)

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(targets))
width = 0.25

for k, pick in enumerate(pick_order):
    part = sel[sel["pick"]==pick].set_index("target").reindex(targets)
    y = part[wf1_mean].to_numpy()
    e = part[wf1_sd].to_numpy()
    ax.bar(x + (k-1)*width, y, width, yerr=e, capsize=3, label=pick)

ax.set_xticks(x)
ax.set_xticklabels(targets)
ax.set_ylabel("weighted-F1 (mean ± SD)")
ax.set_title("Per target: best weighted-F1 ML vs best macro-F1 ML vs fuzzy")
ax.legend(frameon=True)
plt.tight_layout()
plt.savefig(outdir / "Fig2b_grouped_bars.png", dpi=300)
plt.savefig(outdir / "Fig2b_grouped_bars.pdf")
plt.close()

# ---------- Fig3b: scatter with target markers ----------
marker_map = {"stage2":"o","diagnosis":"s","laterality":"^","ihc":"D"}

plt.figure(figsize=(7,5))
for model in models_order:
    sub = all_df[all_df["model"]==model]
    for t in targets:
        ss = sub[sub["target"]==t]
        if len(ss)==0:
            continue
        plt.scatter(ss[wf1_mean], ss[bacc_mean], marker=marker_map[t], s=70, label=f"{model} ({t})")

plt.xlabel("weighted-F1 (mean)")
plt.ylabel("balanced accuracy (mean)")
plt.title("Balanced accuracy vs weighted-F1 (marker=target)")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(outdir / "Fig3b_scatter_targets.png", dpi=300)
plt.savefig(outdir / "Fig3b_scatter_targets.pdf")
plt.close()

print("Wrote improved figures (Fig1b/Fig2b/Fig3b) to outputs/figures/")
