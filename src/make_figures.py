
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

# Normalize target labels for robust matching (e.g., 'Stage' vs 'stage')
all_df["target_key"] = (
    all_df["target"].astype(str).str.strip().str.lower()
    .replace({"ihc": "ihc"})  # keep explicit for readability
)

wf1_mean, wf1_sd = get_mean_std(all_df, "weighted_f1")
mf1_mean, mf1_sd = get_mean_std(all_df, "macro_f1")
bacc_mean, bacc_sd = get_mean_std(all_df, "balanced_acc")

for c in [wf1_mean, wf1_sd, mf1_mean, mf1_sd, bacc_mean, bacc_sd]:
    all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

outdir = Path("outputs/figures")
outdir.mkdir(parents=True, exist_ok=True)

# --- Professional display labels ---
target_order = ["stage", "diagnosis", "laterality", "ihc"]
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
pivot = (all_df.pivot_table(index="target_key", columns="model", values=wf1_mean, aggfunc="first")
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
    sub = all_df[all_df["target_key"]==t].copy()
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
        ss = subm[subm["target_key"]==t]
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


# =========================
# ROBUST_FIG2_OVERRIDE
# (Ensures Stage is included; uses ML_summary/Fuzzy_summary explicitly)
# =========================
def _robust_fig2_grouped_bars(all_df, outdir, target_label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # column discovery: mean/std for metrics
    cols = all_df.columns.tolist()
    def _col(prefix, stat):
        matches = [c for c in cols if c.startswith(prefix)]
        # expected: [prefix, prefix.1] where first is mean, second is std
        if len(matches) < 2:
            raise RuntimeError(f"Cannot find mean/std columns for {prefix}: {matches}")
        return matches[0] if stat=="mean" else matches[1]

    wf1_mean = _col("weighted_f1", "mean")
    wf1_std  = _col("weighted_f1", "std")
    mf1_mean = _col("macro_f1", "mean")
    mf1_std  = _col("macro_f1", "std")

    # normalized target key
    df = all_df.dropna(subset=["target","model"]).copy()
    df["target_key"] = df["target"].astype(str).str.strip().str.lower()

    targets = ["stage","diagnosis","laterality","ihc"]

    # best ML per target by weighted-F1 and by macro-F1
    best_w = []
    best_m = []
    for t in targets:
        sub = df[(df["target_key"]==t) & (df["model"]!="fuzzy")].copy()
        if sub.empty:
            # no ML rows for this target -> keep NaN so we see it missing
            best_w.append((t, np.nan, np.nan))
            best_m.append((t, np.nan, np.nan))
            continue

        subw = sub.sort_values(wf1_mean, ascending=False).iloc[0]
        subm = sub.sort_values(mf1_mean, ascending=False).iloc[0]

        best_w.append((t, float(subw[wf1_mean]), float(subw[wf1_std])))
        best_m.append((t, float(subm[mf1_mean]), float(subm[mf1_std])))

    # fuzzy per target (single row)
    fuzz = df[df["model"]=="fuzzy"].copy()
    fuzz_map = {}
    for t in targets:
        r = fuzz[fuzz["target_key"]==t]
        if r.empty:
            fuzz_map[t] = (np.nan, np.nan)
        else:
            r = r.iloc[0]
            fuzz_map[t] = (float(r[wf1_mean]), float(r[wf1_std]))

    # plot (grouped by endpoint)
    x = np.arange(len(targets))
    width = 0.25

    y_w = [v[1] for v in best_w]
    e_w = [v[2] for v in best_w]
    y_m = [v[1] for v in best_m]
    e_m = [v[2] for v in best_m]
    y_f = [fuzz_map[t][0] for t in targets]
    e_f = [fuzz_map[t][1] for t in targets]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - width, y_w, width, yerr=e_w, capsize=4, label="Best (weighted F1-score)")
    ax.bar(x,         y_m, width, yerr=e_m, capsize=4, label="Best (macro F1-score)")
    ax.bar(x + width, y_f, width, yerr=e_f, capsize=4, label="Fuzzy logic")

    ax.set_xticks(x)
    ax.set_xticklabels([target_label[t] for t in targets])
    ax.set_ylabel("F1-score (mean ± SD)")
    ax.set_title("Per endpoint: best machine-learning model vs fuzzy-logic baseline")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", ncol=3, frameon=True)

    out_png = outdir / "Fig2_grouped_bars.png"
    out_pdf = outdir / "Fig2_grouped_bars.pdf"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("Overwrote Fig2_grouped_bars with robust version (Stage included).")

# Call override at end if variables exist
try:
    _robust_fig2_grouped_bars(all_df, outdir, target_label)
except Exception as _e:
    print("ROBUST_FIG2_OVERRIDE failed:", _e)


# =========================
# ROBUST_FIG3_OVERRIDE
# Adds model-color legend + endpoint-marker legend
# =========================
def _robust_fig3_scatter(all_df, outdir, target_label):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cols = all_df.columns.tolist()
    def _col(prefix, stat):
        matches = [c for c in cols if c.startswith(prefix)]
        if len(matches) < 2:
            raise RuntimeError(f"Cannot find mean/std columns for {prefix}: {matches}")
        return matches[0] if stat=="mean" else matches[1]

    wf1_mean = _col("weighted_f1", "mean")
    bacc_mean = _col("balanced_acc", "mean")

    df = all_df.dropna(subset=["target","model"]).copy()
    df["target_key"] = df["target"].astype(str).str.strip().str.lower()

    # Endpoint markers
    marker_map = {"stage":"o", "diagnosis":"s", "laterality":"^", "ihc":"D"}

    # Model labels (academic)
    model_label = {
        "fuzzy": "Fuzzy logic",
        "gb": "Gradient boosting",
        "logreg": "Logistic regression",
        "logreg_bal": "Logistic regression (balanced)",
        "rf": "Random forest",
        "svm_rbf": "SVM (RBF)",
        "svm_rbf_bal": "SVM (RBF, balanced)",
    }

    # Stable model order for legend
    model_order = ["fuzzy","gb","logreg","logreg_bal","rf","svm_rbf","svm_rbf_bal"]

    # Choose colors automatically by matplotlib cycle, but keep consistent order
    colors = {}
    for i, m in enumerate(model_order):
        colors[m] = f"C{i}"

    fig, ax = plt.subplots(figsize=(10.5, 6))

    # Scatter points
    for _, r in df.iterrows():
        m = str(r["model"])
        t = str(r["target_key"])
        if m not in colors or t not in marker_map:
            continue
        ax.scatter(
            float(r[wf1_mean]),
            float(r[bacc_mean]),
            marker=marker_map[t],
            color=colors[m],
            s=90,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95
        )

    ax.set_title("Balanced accuracy vs Weighted F1-score (marker = endpoint; color = model)")
    ax.set_xlabel("Weighted F1-score (mean)")
    ax.set_ylabel("Balanced accuracy (mean)")

    # Legend 1: Endpoints (markers)
    endpoint_handles = [
        Line2D([0],[0], marker=marker_map[k], color="black", linestyle="",
               markersize=9, label=target_label[k])
        for k in ["stage","diagnosis","laterality","ihc"]
    ]
    leg1 = ax.legend(handles=endpoint_handles, title="Endpoint", loc="lower right", frameon=True)

    # Legend 2: Models (colors)
    model_handles = [
        Line2D([0],[0], marker="o", color=colors[m], linestyle="",
               markersize=9, label=model_label[m])
        for m in model_order
    ]
    leg2 = ax.legend(handles=model_handles, title="Model", loc="upper left", frameon=True)

    ax.add_artist(leg1)  # keep both legends

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(outdir / "Fig3_scatter.png", dpi=200)
    fig.savefig(outdir / "Fig3_scatter.pdf")
    plt.close(fig)
    print("Overwrote Fig3_scatter with model-color + endpoint-marker legends.")

try:
    _robust_fig3_scatter(all_df, outdir, target_label)
except Exception as _e:
    print("ROBUST_FIG3_OVERRIDE failed:", _e)
