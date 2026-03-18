from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ml = pd.read_csv("outputs/metrics_test_ml_80_20_repeats_summary_v2metrics.csv")
fz = pd.read_csv("outputs/metrics_test_fuzzy_80_20_repeats_summary_v2metrics.csv")

# Clean columns: our CSV has duplicated names like accuracy,accuracy.1 etc
def get_mean_std(df, metric):
    cols = df.columns.tolist()
    mcols = [c for c in cols if c.startswith(metric)]
    if len(mcols) < 2:
        raise ValueError(f"Cannot find mean/std columns for {metric}")
    return mcols[0], mcols[1]

for df in (ml, fz):
    df["target"] = df["target"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()

# Merge ML + Fuzzy into one table
all_df = pd.concat([ml, fz], ignore_index=True)

wf1_mean, wf1_sd = get_mean_std(all_df, "weighted_f1")
mf1_mean, mf1_sd = get_mean_std(all_df, "macro_f1")
bacc_mean, bacc_sd = get_mean_std(all_df, "balanced_acc")

# Convert to numeric
for c in [wf1_mean, wf1_sd, mf1_mean, mf1_sd, bacc_mean, bacc_sd]:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

outdir = Path("outputs/figures")
outdir.mkdir(parents=True, exist_ok=True)

targets = ["stage2","diagnosis","laterality","ihc"]

# -------- Figure 1: Heatmap (weighted-F1 mean) --------
pivot = all_df.pivot_table(index="target", columns="model", values=wf1_mean, aggfunc="first").reindex(targets)
plt.figure(figsize=(12, 3.5))
plt.imshow(pivot.values, aspect="auto")
plt.xticks(range(pivot.shape[1]), pivot.columns, rotation=45, ha="right")
plt.yticks(range(pivot.shape[0]), pivot.index)
plt.colorbar(label="weighted-F1 (mean)")
plt.title("Model comparison (weighted-F1 mean) — repeated stratified 80/20")
plt.tight_layout()
plt.savefig(outdir / "Fig1_heatmap_weightedF1.png", dpi=300)
plt.savefig(outdir / "Fig1_heatmap_weightedF1.pdf")
plt.close()

# -------- Figure 2: Barplot with error bars (Top2 + Fuzzy per target) --------
rows = []
for t in targets:
    sub = all_df[all_df["target"]==t].copy()
    # ensure fuzzy included
    fuzzy_row = sub[sub["model"]=="fuzzy"].copy()
    # pick top2 by weighted-F1 mean among non-fuzzy
    nonfz = sub[sub["model"]!="fuzzy"].sort_values(wf1_mean, ascending=False).head(2)
    pick = pd.concat([nonfz, fuzzy_row], ignore_index=True)
    pick["target"] = t
    rows.append(pick)
sel = pd.concat(rows, ignore_index=True)

# order bars: for each target => best, second, fuzzy
fig, ax = plt.subplots(figsize=(10, 4))
x_labels = []
y = []
yerr = []
for t in targets:
    sub = sel[sel["target"]==t].sort_values(wf1_mean, ascending=False)
    # force fuzzy last if present
    if (sub["model"]=="fuzzy").any():
        sub = pd.concat([sub[sub["model"]!="fuzzy"], sub[sub["model"]=="fuzzy"]], ignore_index=True)
    for _, r in sub.iterrows():
        x_labels.append(f"{t}\n{r['model']}")
        y.append(r[wf1_mean])
        yerr.append(r[wf1_sd])
ax.bar(range(len(y)), y, yerr=yerr, capsize=3)
ax.set_xticks(range(len(y)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_ylabel("weighted-F1 (mean ± SD)")
ax.set_title("Top-2 ML vs Fuzzy per target (repeated stratified 80/20)")
plt.tight_layout()
plt.savefig(outdir / "Fig2_bar_top2_vs_fuzzy.png", dpi=300)
plt.savefig(outdir / "Fig2_bar_top2_vs_fuzzy.pdf")
plt.close()

# -------- Figure 3: Scatter (balanced acc vs weighted-F1) --------
plt.figure(figsize=(6,5))
for model, sub in all_df.groupby("model"):
    plt.scatter(sub[wf1_mean], sub[bacc_mean], label=model)
plt.xlabel("weighted-F1 (mean)")
plt.ylabel("balanced accuracy (mean)")
plt.title("Imbalance-aware view: balanced acc vs weighted-F1")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=7)
plt.tight_layout()
plt.savefig(outdir / "Fig3_scatter_bacc_vs_wf1.png", dpi=300)
plt.savefig(outdir / "Fig3_scatter_bacc_vs_wf1.pdf")
plt.close()

print("Wrote figures to outputs/figures/")
