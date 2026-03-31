from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

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
COLOR_MAP = {m: f"C{i}" for i, m in enumerate(MODEL_ORDER)}


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
    "svm_rbf_balanced": "svm_bal",
}
MARKER_MAP = {"stage": "o", "diagnosis": "s", "laterality": "^", "ihc": "D"}


def read_summary(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["target"] = df["target"].astype(str).str.strip().str.lower()
    df["model"] = df["model"].astype(str).str.strip()
    return df


def flatten_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    seen = {}
    for c in df.columns:
        key = c.strip()
        seen[key] = seen.get(key, 0) + 1
        new_cols.append(key if seen[key] == 1 else f"{key}.{seen[key]-1}")
    df.columns = new_cols
    return df


def metric_cols(df: pd.DataFrame, prefix: str):
    cols = [c for c in df.columns if c == prefix or c.startswith(prefix + ".")]
    if len(cols) < 2:
        raise RuntimeError(f"Could not find mean/std columns for {prefix}. Found: {cols}")
    return cols[0], cols[1]


def pretty_targets(xs):
    return [TARGET_LABEL.get(x, x) for x in xs]


def pretty_models(xs):
    return [MODEL_LABEL.get(x, x) for x in xs]


def build_all_summary():
    ml = flatten_summary_columns(read_summary("outputs/ML_summary.csv"))
    fz = flatten_summary_columns(read_summary("outputs/Fuzzy_summary.csv"))
    all_df = pd.concat([ml, fz], ignore_index=True)
    for c in all_df.columns:
        if c not in {"target", "model"}:
            try:
                all_df[c] = pd.to_numeric(all_df[c])
            except Exception:
                pass
    return all_df


def fig1_heatmap(all_df):
    wf1_mean, _ = metric_cols(all_df, "weighted_f1")
    pivot = (
        all_df.pivot_table(index="target", columns="model", values=wf1_mean, aggfunc="first")
        .reindex(index=TARGET_ORDER, columns=MODEL_ORDER)
    )
    pivot = pivot.apply(pd.to_numeric, errors="coerce")
    vals = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    im = ax.imshow(vals, aspect="auto", cmap="BuGn", vmin=0, vmax=1)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pretty_models(pivot.columns), rotation=30, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pretty_targets(pivot.index))
    fig.colorbar(im, ax=ax, label="Weighted F1-score (mean)")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Model performance comparison (Weighted F1-score; mean across 5 repeats)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig1_heatmap.png", dpi=300)
    fig.savefig(OUTDIR / "Fig1_heatmap.pdf")
    plt.close(fig)


def fig2_grouped_bars(all_df):
    wf1_mean, wf1_sd = metric_cols(all_df, "weighted_f1")
    mf1_mean, mf1_sd = metric_cols(all_df, "macro_f1")

    best_w = []
    best_m = []
    fuzz = []
    for t in TARGET_ORDER:
        sub = all_df[all_df["target"] == t].copy()
        ml_sub = sub[~sub["model"].isin(["fuzzy", "expert_fuzzy"])].copy()
        fz_sub = sub[sub["model"] == "fuzzy"].copy()
        if not ml_sub.empty:
            rw = ml_sub.sort_values(wf1_mean, ascending=False).iloc[0]
            rm = ml_sub.sort_values(mf1_mean, ascending=False).iloc[0]
            best_w.append((t, float(rw[wf1_mean]), float(rw[wf1_sd])))
            best_m.append((t, float(rm[mf1_mean]), float(rm[mf1_sd])))
        else:
            best_w.append((t, np.nan, np.nan))
            best_m.append((t, np.nan, np.nan))
        if not fz_sub.empty:
            rf = fz_sub.iloc[0]
            fuzz.append((t, float(rf[wf1_mean]), float(rf[wf1_sd])))
        else:
            fuzz.append((t, np.nan, np.nan))

    x = np.arange(len(TARGET_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.bar(x - width, [v[1] for v in best_w], width, yerr=[v[2] for v in best_w], capsize=4, label="Best ML (weighted F1-score)")
    ax.bar(x, [v[1] for v in best_m], width, yerr=[v[2] for v in best_m], capsize=4, label="Best ML (macro F1-score)")
    ax.bar(x + width, [v[1] for v in fuzz], width, yerr=[v[2] for v in fuzz], capsize=4, label="Fuzzy logic")
    ax.set_xticks(x)
    ax.set_xticklabels(pretty_targets(TARGET_ORDER))
    ax.set_ylabel("F1-score (mean ± SD)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per endpoint: best machine-learning model versus fuzzy-logic baseline")
    ax.legend(frameon=True, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig2_grouped_bars.png", dpi=300)
    fig.savefig(OUTDIR / "Fig2_grouped_bars.pdf")
    plt.close(fig)


def fig3_scatter(all_df):
    wf1_mean, _ = metric_cols(all_df, "weighted_f1")
    bacc_mean, _ = metric_cols(all_df, "balanced_acc")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for _, r in all_df.iterrows():
        t = r["target"]
        m = r["model"]
        if t not in MARKER_MAP or m not in COLOR_MAP:
            continue
        ax.scatter(
            float(r[wf1_mean]),
            float(r[bacc_mean]),
            marker=MARKER_MAP[t],
            color=COLOR_MAP[m],
            s=90,
            edgecolors="black",
            linewidths=0.4,
            alpha=0.95,
        )

    endpoint_handles = [
        Line2D([0], [0], marker=MARKER_MAP[t], color="black", linestyle="", markersize=9, label=TARGET_LABEL[t])
        for t in TARGET_ORDER
    ]
    model_handles = [
        Line2D([0], [0], marker="o", color=COLOR_MAP[m], linestyle="", markersize=9, label=MODEL_LABEL[m])
        for m in MODEL_ORDER
    ]
    leg1 = ax.legend(handles=endpoint_handles, title="Endpoint", loc="lower right", frameon=True)
    ax.legend(handles=model_handles, title="Model", loc="upper left", frameon=True)
    ax.add_artist(leg1)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Weighted F1-score (mean)")
    ax.set_ylabel("Balanced accuracy (mean)")
    ax.set_title("Balanced accuracy versus Weighted F1-score (marker = endpoint; color = model)")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig3_scatter.png", dpi=300)
    fig.savefig(OUTDIR / "Fig3_scatter.pdf")
    plt.close(fig)


def fig4_metric_distributions():
    all_df = _full_summary_df().copy()

    # standardize model names if needed
    all_df["model"] = all_df["model"].astype(str)

    # helper for summary column names
    def _metric_cols_local(df, base):
        if f"{base}" in df.columns and f"{base}.1" in df.columns:
            return f"{base}", f"{base}.1"
        if f"{base}_mean" in df.columns and f"{base}_std" in df.columns:
            return f"{base}_mean", f"{base}_std"
        raise KeyError(f"Could not find summary columns for {base}")

    wf1_mean, _ = _metric_cols_local(all_df, "weighted_f1")

    panel_labels = ["A)", "B)", "C)", "D)"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    axes = axes.ravel()

    desired_order = [
        "expert_fuzzy",
        "fuzzy",
        "gb",
        "logreg",
        "logreg_bal",
        "rf",
        "svm",
        "svm_bal",
    ]

    for i, (ax, tgt) in enumerate(zip(axes, TARGET_ORDER)):
        sub = all_df[all_df["target"] == tgt].copy()
        if sub.empty:
            ax.axis("off")
            continue

        # keep only models that exist in this target
        present_order = [m for m in desired_order if m in set(sub["model"].tolist())]
        if not present_order:
            ax.axis("off")
            continue

        # build long dataframe from mean ± std for plotting points
        rows = []
        for model_name in present_order:
            row = sub[sub["model"] == model_name]
            if row.empty:
                continue
            mean_val = float(row.iloc[0][wf1_mean])

            # create pseudo-repeats only for visual consistency if only summary exists
            # keep deterministic narrow spread around mean
            vals = [mean_val]
            rows.append(pd.DataFrame({
                "target": tgt,
                "model": model_name,
                "weighted_f1": vals
            }))

        plot_df = pd.concat(rows, ignore_index=True)

        # if detailed per-seed values exist in Fuzzy_detailed / ML_detailed, prefer them
        try:
            fz_det = pd.read_csv("outputs/Fuzzy_detailed.csv")
            ml_det = pd.read_csv("outputs/ML_detailed.csv")
            det = pd.concat([fz_det, ml_det], ignore_index=True)
            det = det[(det["target"].astype(str) == str(tgt)) &
                      (det["model"].astype(str).isin(present_order))].copy()
            if not det.empty and "weighted_f1" in det.columns:
                plot_df = det[["target", "model", "weighted_f1"]].copy()
        except Exception:
            pass

        order_for_plot = [m for m in present_order if m in set(plot_df["model"].astype(str))]
        if not order_for_plot:
            ax.axis("off")
            continue

        positions = np.arange(len(order_for_plot))

        parts = ax.violinplot(
            [plot_df.loc[plot_df["model"] == m, "weighted_f1"].astype(float).values for m in order_for_plot],
            positions=positions,
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_alpha(0.25)

        for j, model_name in enumerate(order_for_plot):
            vals = plot_df.loc[plot_df["model"] == model_name, "weighted_f1"].astype(float).values
            if len(vals) == 0:
                continue
            xj = np.repeat(j, len(vals)).astype(float)
            if len(vals) > 1:
                jitter = np.linspace(-0.08, 0.08, len(vals))
                xj = xj + jitter
            ax.scatter(xj, vals, s=16, alpha=0.8)
            ax.hlines(np.mean(vals), j - 0.18, j + 0.18, linewidth=1.8)

        labels = [MODEL_LABEL.get(m, m.replace("_", " ").title()) for m in order_for_plot]
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(TARGET_LABEL.get(tgt, tgt).title())
        ax.set_ylabel("Weighted F1-score")

        label_x = -0.10 if i % 2 == 0 else -0.03
        ax.text(
            label_x, 1.03, panel_labels[i],
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold"
        )

    fig.suptitle("Fold/seed-wise weighted F1-score distributions across repeated splits")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig4_metric_distributions.png", dpi=300)
    fig.savefig(OUTDIR / "Fig4_metric_distributions.pdf")
    plt.close(fig)


def fig5_split_overview():
    df = pd.read_csv("outputs/Split_info.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["n_train_total"], width, label="Train")
    ax.bar(x + width / 2, df["n_test_total"], width, label="Test")
    ax.plot(x, df["n_test_known_stage"], marker="o", linestyle="--", label="Known Stage in test")
    ax.plot(x, df["n_test_unknown_stage"], marker="s", linestyle="--", label="Unknown Stage in test")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in df["seed"]])
    ax.set_ylabel("Number of samples")
    ax.set_title("Repeated 80/20 split overview across seeds")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig5_split_overview.png", dpi=300)
    fig.savefig(OUTDIR / "Fig5_split_overview.pdf")
    plt.close(fig)


def fig6_unknown_stage_methods():
    files = {
        "Initial prediction": "outputs/Stage_unknown_predictions.csv",
        "Self-training": "outputs/Stage_unknown_selftraining.csv",
        "Label propagation": "outputs/Stage_unknown_labelprop.csv",
    }

    frames = []
    for method, path in files.items():
        p = Path(path)
        if not p.exists():
            continue

        tmp = pd.read_csv(p)
        tmp.columns = [c.strip() for c in tmp.columns]

        pred_candidates = [
            "Stage_predicted",
            "Stage_pseudolabel",
            "Stage_labelprop",
            "predicted_stage",
            "Prediction",
            "prediction",
            "Pred",
            "pred",
        ]
        conf_candidates = [
            "confidence",
            "Confidence",
            "conf",
            "Conf",
        ]

        pred_col = next((c for c in pred_candidates if c in tmp.columns), None)
        conf_col = next((c for c in conf_candidates if c in tmp.columns), None)

        if pred_col is not None:
            tmp["Predicted_Stage_std"] = tmp[pred_col].astype(str).str.strip()
        else:
            tmp["Predicted_Stage_std"] = np.nan

        if conf_col is not None:
            tmp["Confidence_std"] = pd.to_numeric(tmp[conf_col], errors="coerce")
        else:
            tmp["Confidence_std"] = np.nan

        tmp["method"] = method
        frames.append(tmp)

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    methods = list(files.keys())

    pred_base = df.dropna(subset=["Predicted_Stage_std"]).copy()
    if "PATIENT ID" in pred_base.columns:
        pred_base = pred_base.drop_duplicates(subset=["method", "PATIENT ID"])

    pred_summary = (
        pred_base.groupby(["method", "Predicted_Stage_std"])
                 .size()
                 .reset_index(name="n")
    )

    labels = ["I-II", "III-IV"]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left panel: stacked percentage bar
    ax = axes[0]
    bottoms = np.zeros(len(methods), dtype=float)

    for lab in labels:
        counts = np.array([
            pred_summary[
                (pred_summary["method"] == m) &
                (pred_summary["Predicted_Stage_std"].astype(str) == lab)
            ]["n"].sum()
            for m in methods
        ], dtype=float)

        totals = np.array([
            pred_summary[pred_summary["method"] == m]["n"].sum()
            for m in methods
        ], dtype=float)

        props = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals > 0
        )

        ax.bar(x, props, bottom=bottoms, width=0.6, label=lab)
        for i, (b, p, c) in enumerate(zip(bottoms, props, counts)):
            if p > 0:
                ax.text(i, b + p / 2, f"{int(c)}", ha="center", va="center", fontsize=9)
        bottoms += props

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion of unlabeled Stage cases")
    ax.set_title("Predicted Stage composition in unlabeled cases")
    ax.text(
        -0.10, 1.03, "A)",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold"
    )
    ax.legend(title="Predicted Stage", frameon=True, loc="upper right")

    # Right panel: violin + jittered points + mean line
    ax2 = axes[1]
    conf_data = [df[df["method"] == m]["Confidence_std"].dropna().to_numpy() for m in methods]

    vp = ax2.violinplot(
        conf_data,
        positions=np.arange(1, len(methods) + 1),
        widths=0.48,
        showmeans=False,
        showmedians=False,
        showextrema=False
    )

    for body in vp["bodies"]:
        body.set_alpha(0.22)

    for i, arr in enumerate(conf_data, start=1):
        if len(arr):
            jitter = np.linspace(-0.07, 0.07, len(arr)) if len(arr) > 1 else np.array([0.0])

            ax2.scatter(
                np.full(len(arr), i) + jitter,
                arr,
                s=30,
                alpha=0.8
            )

            mean_val = float(np.mean(arr))
            ax2.hlines(mean_val, i - 0.18, i + 0.18, linewidth=2.0)

    ax2.set_xticks(np.arange(1, len(methods) + 1))
    ax2.set_xticklabels(methods, rotation=15, ha="right")
    ax2.set_ylabel("Confidence")
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Confidence scores across exploratory labeling methods")
    ax2.text(
        -0.03, 1.03, "B)",
        transform=ax2.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold"
    )

    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig6_unknown_stage_methods.png", dpi=300)
    fig.savefig(OUTDIR / "Fig6_unknown_stage_methods.pdf")
    plt.close(fig)

def fig7_confusion_matrices():
    from sklearn.metrics import confusion_matrix

    def _metric_cols_local(df, base):
        if f"{base}" in df.columns and f"{base}.1" in df.columns:
            return f"{base}", f"{base}.1"
        if f"{base}_mean" in df.columns and f"{base}_std" in df.columns:
            return f"{base}_mean", f"{base}_std"
        raise KeyError(f"Could not find summary columns for {base}")

    all_df = _full_summary_df()
    wf1_mean, _ = _metric_cols_local(all_df, "weighted_f1")

    pred_df = pd.read_csv("outputs/Test_predictions_detailed.csv")
    expert_df = pd.read_csv("outputs/ExpertFuzzy_detailed.csv")

    panel_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    target_rows = []

    for tgt in TARGET_ORDER:
        sub = all_df[all_df["target"] == tgt].copy()
        if sub.empty:
            continue

        ml_sub = sub[~sub["model"].isin(["fuzzy", "expert_fuzzy"])]
        if ml_sub.empty:
            continue

        best_ml = ml_sub.sort_values(wf1_mean, ascending=False).iloc[0]["model"]

        fuzzy_sub = pred_df[(pred_df["target"].astype(str) == str(tgt)) &
                            (pred_df["model"].astype(str) == "fuzzy")].copy()
        expert_sub = expert_df[(expert_df["target"].astype(str) == str(tgt)) &
                               (expert_df["model"].astype(str) == "expert_fuzzy")].copy()
        ml_pred_sub = pred_df[(pred_df["target"].astype(str) == str(tgt)) &
                              (pred_df["model"].astype(str) == str(best_ml))].copy()

        if len(fuzzy_sub) == 0 or len(expert_sub) == 0 or len(ml_pred_sub) == 0:
            continue

        target_rows.append((tgt, [
            ("fuzzy", fuzzy_sub),
            ("expert_fuzzy", expert_sub),
            (best_ml, ml_pred_sub),
        ]))

    if not target_rows:
        return

    n_rows = len(target_rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    panel_idx = 0

    for r, (tgt, entries) in enumerate(target_rows):
        for c, (model_name, det_sub) in enumerate(entries):
            ax = axes[r, c]

            y_true = det_sub["y_true_label"].astype(str)
            y_pred = det_sub["y_pred_label"].astype(str)
            labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            im = ax.imshow(cm, aspect="auto", cmap="BuGn")

            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels)

            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)

            model_label = MODEL_LABEL.get(model_name, str(model_name).replace("_", " ").title())
            ax.set_title(f"{TARGET_LABEL.get(tgt, tgt)} — {model_label}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

            label_x = -0.16 if c == 0 else -0.10
            ax.text(
                label_x, 1.03, f"{panel_labels[panel_idx]})",
                transform=ax.transAxes,
                ha="left", va="bottom",
                fontsize=12, fontweight="bold"
            )
            panel_idx += 1

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig7_confusion_combined_vertical.png", dpi=300)
    fig.savefig(OUTDIR / "Fig7_confusion_combined_vertical.pdf")
    plt.close(fig)


def fig8_stage_roc():
    path = Path("outputs/Test_predictions_detailed.csv")
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["target"] = df["target"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)
    stage = df[(df["target"] == "stage") & (df["model"] != "fuzzy")].copy()
    if stage.empty:
        return

    all_df = build_all_summary()
    wf1_mean, _ = metric_cols(all_df, "weighted_f1")
    best_models = (
        all_df[(all_df["target"] == "stage") & (all_df["model"] != "fuzzy")]
        .sort_values(wf1_mean, ascending=False)
        .head(3)["model"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = False
    for model in best_models:
        sub = stage[(stage["model"] == model) & stage["y_score"].notna()].copy()
        if sub.empty:
            continue
        y_true = sub["y_true_class"].astype(int).to_numpy()
        y_score = sub["y_score"].astype(float).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{MODEL_LABEL.get(model, model)} (AUC={roc_auc:.2f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Stage classification ROC curves for top ML models")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig8_stage_roc.png", dpi=300)
    fig.savefig(OUTDIR / "Fig8_stage_roc.pdf")
    plt.close(fig)



def _normalize_stage_raw(series):
    s = series.astype(str).str.strip()
    s = s.str.replace("–", "-", regex=False).str.replace("—", "-", regex=False)
    return s


def _load_raw_dataframe():
    from .io_utils import load_xy
    df, X, Y = load_xy()
    return df, X, Y


def _full_summary_df():
    return build_all_summary()


def fig9_stage_pr_curve():
    path = Path("outputs/Test_predictions_detailed.csv")
    if not path.exists():
        return

    df = pd.read_csv(path)
    df["target"] = df["target"].astype(str).str.lower()
    df["model"] = df["model"].astype(str)

    stage = df[(df["target"] == "stage") & (df["model"] != "fuzzy")].copy()
    if stage.empty:
        return

    all_df = _full_summary_df()
    wf1_mean, _ = metric_cols(all_df, "weighted_f1")
    best_models = (
        all_df[(all_df["target"] == "stage") & (all_df["model"] != "fuzzy")]
        .sort_values(wf1_mean, ascending=False)
        .head(3)["model"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plotted = False
    for model in best_models:
        sub = stage[(stage["model"] == model) & stage["y_score"].notna()].copy()
        if sub.empty:
            continue
        y_true = sub["y_true_class"].astype(int).to_numpy()
        y_score = sub["y_score"].astype(float).to_numpy()
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, label=f"{MODEL_LABEL.get(model, model)} (AP={ap:.2f})")
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    pos_rate = stage["y_true_class"].astype(int).mean()
    ax.hlines(pos_rate, 0, 1, linestyle="--", label=f"Baseline ({pos_rate:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Stage classification precision–recall curves")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig9_stage_pr_curve.png", dpi=300)
    fig.savefig(OUTDIR / "Fig9_stage_pr_curve.pdf")
    plt.close(fig)


def fig10_feature_importance():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from .config import X_COLS

    df, X, Y = _load_raw_dataframe()
    stage_raw = _normalize_stage_raw(df["Stage_2grp"])
    known_mask = stage_raw.isin(["I-II", "III-IV"])
    if known_mask.sum() == 0:
        return

    Xk = X[known_mask.to_numpy()]
    yk = stage_raw[known_mask].map({"I-II": 0, "III-IV": 1}).astype(int).to_numpy()
    feat_names = list(X_COLS)

    models = {
        "Logistic regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=500, random_state=42),
        "Gradient boosting": GradientBoostingClassifier(random_state=42),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    panel_labels = ["A)", "B)", "C)"]

    for idx, (ax, (title, model)) in enumerate(zip(axes, models.items())):
        model.fit(Xk, yk)

        if hasattr(model, "coef_"):
            imp = np.abs(np.ravel(model.coef_))
        elif hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float)
        else:
            ax.axis("off")
            continue

        order = np.argsort(imp)[::-1][:15]
        vals = imp[order][::-1]
        names = [feat_names[i] for i in order][::-1]

        ax.barh(np.arange(len(names)), vals)
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(title)

        label_x = -0.18 if idx == 0 else -0.10
        ax.text(
            label_x, 1.03, panel_labels[idx],
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold"
        )

        ax.set_xlabel("Importance" if title != "Logistic regression" else "|Coefficient|")

    fig.suptitle("Top input features for Stage classification models")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig10_feature_importance.png", dpi=300)
    fig.savefig(OUTDIR / "Fig10_feature_importance.pdf")
    plt.close(fig)


def fig11_correlation_heatmap():
    from .config import X_COLS

    df, X, Y = _load_raw_dataframe()
    feat_df = pd.DataFrame(X, columns=X_COLS)

    # Keep the top 20 most variable features for readability
    var = feat_df.var(axis=0, numeric_only=True).sort_values(ascending=False)
    keep = list(var.head(min(20, len(var))).index)
    corr = feat_df[keep].corr().apply(pd.to_numeric, errors="coerce")
    vals = corr.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(vals, aspect="auto", cmap="BuGn", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(keep)))
    ax.set_xticklabels(keep, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels(keep, fontsize=8)
    ax.set_title("Feature–feature correlation heatmap (top-variance inputs)")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig11_correlation_heatmap.png", dpi=300)
    fig.savefig(OUTDIR / "Fig11_correlation_heatmap.pdf")
    plt.close(fig)


def fig12_class_distribution():
    from .config import Y_STAGE2_STR_COL, Y_DIAG_COL, Y_LAT_COL, Y_IHC_COL

    df, X, Y = _load_raw_dataframe()

    stage_raw = _normalize_stage_raw(df[Y_STAGE2_STR_COL])
    stage_plot = stage_raw.replace({"Unknown": "Unknown", "Unkown": "Unknown"})

    diag = df[Y_DIAG_COL].copy()
    lat = df[Y_LAT_COL].copy()
    ihc = df[Y_IHC_COL].copy()

    diag = diag.where(~pd.isna(diag), "Missing").astype(str)
    lat = lat.where(~pd.isna(lat), "Missing").astype(str)
    ihc = ihc.where(~pd.isna(ihc), "Missing").astype(str)

    series_list = [
        ("Stage", stage_plot),
        ("Diagnosis", diag),
        ("Laterality", lat),
        ("IHC", ihc),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for panel_idx, (ax, (title, ser)) in enumerate(zip(axes, series_list)):
        vc = ser.value_counts(dropna=False)
        labels = vc.index.astype(str).tolist()
        vals = vc.values.astype(float)

        ax.bar(np.arange(len(vals)), vals)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("n")
        ax.set_title(title)
        label_x = -0.10 if panel_idx % 2 == 0 else -0.06
        ax.text(
            label_x, 1.03, ["A)", "B)", "C)", "D)"][panel_idx],
            transform=ax.transAxes,
            ha="left", va="bottom",
            fontsize=12, fontweight="bold"
        )

        ymax = max(vals) if len(vals) else 0
        upper = ymax * 1.12 if ymax > 0 else 1
        ax.set_ylim(0, upper)

        for i, v in enumerate(vals):
            y_text = min(v + max(ymax * 0.02, 0.6), upper * 0.97)
            ax.text(i, y_text, str(int(v)), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Class distribution across prediction targets")
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig12_class_distribution.png", dpi=300)
    fig.savefig(OUTDIR / "Fig12_class_distribution.pdf")
    plt.close(fig)

def fig13_method_agreement_heatmap():
    files = {
        "Initial prediction": "outputs/Stage_unknown_predictions.csv",
        "Self-training": "outputs/Stage_unknown_selftraining.csv",
        "Label propagation": "outputs/Stage_unknown_labelprop.csv",
    }

    rows = []
    for method, path in files.items():
        p = Path(path)
        if not p.exists():
            continue

        tmp = pd.read_csv(p)
        tmp.columns = [c.strip() for c in tmp.columns]

        pred_col = None
        for c in ["Stage_predicted", "Stage_pseudolabel", "Stage_labelprop"]:
            if c in tmp.columns:
                pred_col = c
                break

        if pred_col is None or "PATIENT ID" not in tmp.columns:
            continue

        part = tmp[["PATIENT ID", pred_col]].copy()
        part["PATIENT ID"] = part["PATIENT ID"].astype(str).str.strip()
        part[pred_col] = part[pred_col].astype(str).str.strip()
        part["method"] = method
        part = part.rename(columns={pred_col: "prediction"})
        part = part.drop_duplicates(subset=["PATIENT ID", "method"])
        rows.append(part)

    if len(rows) < 2:
        return

    long_df = pd.concat(rows, ignore_index=True)
    wide = long_df.pivot(index="PATIENT ID", columns="method", values="prediction")

    methods = [m for m in files.keys() if m in wide.columns]
    if len(methods) < 2:
        return

    agree = np.full((len(methods), len(methods)), np.nan, dtype=float)

    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if m1 == m2:
                agree[i, j] = 1.0
                continue

            sub = wide[[m1, m2]].dropna()
            if len(sub) == 0:
                continue

            s1 = sub.iloc[:, 0].astype(str)
            s2 = sub.iloc[:, 1].astype(str)
            agree[i, j] = float((s1 == s2).mean())

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(agree, cmap="BuGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)

    for i in range(len(methods)):
        for j in range(len(methods)):
            v = agree[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center")

    ax.set_title("Agreement across unknown-Stage methods", fontsize=15, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    cbar.set_label("Agreement proportion")

    fig.subplots_adjust(top=0.88, right=0.86)
    fig.savefig(OUTDIR / "Fig13_method_agreement_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTDIR / "Fig13_method_agreement_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

def fig14_agreement_lollipop():
    all_df = _full_summary_df()
    wf1_mean, _ = metric_cols(all_df, "weighted_f1")

    rows = []
    for t in TARGET_ORDER:
        sub = all_df[all_df["target"] == t].copy()
        fz = sub[sub["model"] == "fuzzy"]
        ml = sub[~sub["model"].isin(["fuzzy", "expert_fuzzy"])]
        if fz.empty or ml.empty:
            continue
        fz_val = float(fz.iloc[0][wf1_mean])
        best_ml_row = ml.sort_values(wf1_mean, ascending=False).iloc[0]
        ml_val = float(best_ml_row[wf1_mean])
        rows.append((t, fz_val, ml_val, best_ml_row["model"]))

    if not rows:
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(rows))
    for i, (t, fz_val, ml_val, best_model) in enumerate(rows):
        ax.hlines(i, min(fz_val, ml_val), max(fz_val, ml_val), linewidth=2)
        ax.scatter(fz_val, i, s=70, label="Fuzzy logic" if i == 0 else None)
        ax.scatter(ml_val, i, s=70, marker="s", label="Best ML" if i == 0 else None)
        ax.text(ml_val + 0.01, i, MODEL_LABEL.get(best_model, best_model), va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels([TARGET_LABEL.get(r[0], r[0]) for r in rows])
    ax.set_xlabel("Weighted F1-score (mean)")
    ax.set_xlim(0, 1)
    ax.set_title("Best machine-learning model versus fuzzy-logic baseline")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(OUTDIR / "Fig14_agreement_lollipop.png", dpi=300)
    fig.savefig(OUTDIR / "Fig14_agreement_lollipop.pdf")
    plt.close(fig)


def fig15_workflow_schematic():
    fig, ax = plt.subplots(figsize=(13.5, 4.7))
    ax.axis("off")

    ax.set_xlim(0.00, 0.96)
    ax.set_ylim(-0.06, 0.86)

    boxes = [
        (0.03, 0.28, 0.13, 0.18, "Input dataset\nclinical + tumor\nfeatures"),
        (0.20, 0.28, 0.13, 0.18, "Preprocessing\nand target definition"),
        (0.37, 0.28, 0.13, 0.18, "Repeated stratified\n80/20 splits"),
        (0.58, 0.50, 0.16, 0.15, "Fuzzy logic\nWang–Mendel + Mamdani"),
        (0.58, 0.12, 0.16, 0.15, "Machine learning\nLR / SVM / RF / GB"),
        (0.79, 0.28, 0.16, 0.18, "Evaluation\nAccuracy / F1 / BAcc\nROC / PR / confusion"),
        (0.41, -0.06, 0.26, 0.12, "Unknown-Stage\nsecondary analyses\ninitial prediction / self-training / label propagation"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=10)

    arrows = [
        ((0.16, 0.37), (0.20, 0.37)),
        ((0.33, 0.37), (0.37, 0.37)),
        ((0.50, 0.37), (0.58, 0.575)),
        ((0.50, 0.37), (0.58, 0.195)),
        ((0.74, 0.575), (0.79, 0.37)),
        ((0.74, 0.195), (0.79, 0.37)),
        ((0.435, 0.28), (0.54, 0.06)),
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=1.6)
        )

    # Başlığı ayrı yerleştiriyoruz: şemayı bozmadan konum kontrolü
    fig.text(
        0.47, 0.935,
        "Overview of the fuzzy-logic versus machine-learning analysis workflow",
        ha="center", va="center", fontsize=14
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03)
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)








def main():
    all_df = build_all_summary()
    fig1_heatmap(all_df)
    fig2_grouped_bars(all_df)
    fig3_scatter(all_df)

    if Path("outputs/ML_detailed.csv").exists() and Path("outputs/Fuzzy_detailed.csv").exists():
        fig4_metric_distributions()
    if Path("outputs/Split_info.csv").exists():
        fig5_split_overview()
    fig6_unknown_stage_methods()
    fig7_confusion_matrices()
    fig8_stage_roc()
    fig9_stage_pr_curve()
    fig10_feature_importance()
    fig11_correlation_heatmap()
    fig12_class_distribution()
    fig13_method_agreement_heatmap()
    fig14_agreement_lollipop()
    fig15_workflow_schematic()
    print(f"Figures written to: {OUTDIR}")


if __name__ == "__main__":
    main()
