import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .config import MAX_RULES, STAGE2_CENTERS
from .io_utils import load_xy
from .fuzzy_wm import wang_mendel_train, mamdani_predict
from .metrics_utils import eval_target, weighted_f1_from_confusion
from .ml_models import get_models, to_class

TARGETS = ["stage2", "diagnosis", "laterality", "ihc"]

def centers_from_train(y):
    c = np.unique(y.astype(float))
    return np.sort(c).tolist()

def run_one_split(X, Y, seed, test_size=0.20):
    # stratify by stage2 class
    y_stage2_class = to_class(Y[:, 0], STAGE2_CENTERS)
    idx = np.arange(len(Y))
    tr_idx, te_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y_stage2_class
    )
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xte, Yte = X[te_idx], Y[te_idx]

    centers_map = {
        "stage2": STAGE2_CENTERS,
        "diagnosis": centers_from_train(Ytr[:, 1]),
        "laterality": centers_from_train(Ytr[:, 2]),
        "ihc": centers_from_train(Ytr[:, 3]),
    }

    # fuzzy (single-output)
    fuzzy_rows = []
    for target_idx, target_name in enumerate(TARGETS):
        ytr = Ytr[:, target_idx:target_idx+1]
        yte = Yte[:, target_idx]
        in_mfs, out_mfs, rules = wang_mendel_train(Xtr, ytr, max_rules=MAX_RULES)
        yhat = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]
        acc, wf1, _ = eval_target(yte, yhat, centers_map[target_name])
        fuzzy_rows.append({"seed": seed, "target": target_name, "model": "fuzzy", "accuracy": acc, "weighted_f1": wf1})

    # ML
    ml_models = get_models(seed=seed)
    ml_rows = []
    for target_idx, target_name in enumerate(TARGETS):
        centers = centers_map[target_name]
        ytr_cls = to_class(Ytr[:, target_idx], centers)
        yte_cls = to_class(Yte[:, target_idx], centers)

        for model_name, model in ml_models.items():
            model.fit(Xtr, ytr_cls)
            pred = model.predict(Xte)
            acc = float((pred == yte_cls).mean())
            C = confusion_matrix(yte_cls, pred)
            wf1 = weighted_f1_from_confusion(C)
            ml_rows.append({"seed": seed, "target": target_name, "model": model_name, "accuracy": acc, "weighted_f1": wf1})

    return pd.DataFrame(fuzzy_rows), pd.DataFrame(ml_rows)

def main():
    df, X, Y = load_xy()

    seeds = [7, 13, 21, 42, 99]  # 5 repeats
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    all_fuzzy = []
    all_ml = []

    for s in seeds:
        fz, ml = run_one_split(X, Y, seed=s, test_size=0.20)
        all_fuzzy.append(fz)
        all_ml.append(ml)

    fz_df = pd.concat(all_fuzzy, ignore_index=True)
    ml_df = pd.concat(all_ml, ignore_index=True)

    fz_df.to_csv(outdir / "metrics_test_fuzzy_80_20_repeats.csv", index=False)
    ml_df.to_csv(outdir / "metrics_test_ml_80_20_repeats.csv", index=False)

    # Summary mean±sd
    fz_sum = (fz_df.groupby(["target","model"])[["accuracy","weighted_f1"]]
                .agg(["mean","std"])
                .reset_index())
    ml_sum = (ml_df.groupby(["target","model"])[["accuracy","weighted_f1"]]
                .agg(["mean","std"])
                .reset_index())

    fz_sum.to_csv(outdir / "metrics_test_fuzzy_80_20_repeats_summary.csv", index=False)
    ml_sum.to_csv(outdir / "metrics_test_ml_80_20_repeats_summary.csv", index=False)

    print("Done. Wrote repeat 80/20 outputs to outputs/")
    print("- metrics_test_fuzzy_80_20_repeats.csv (+ _summary)")
    print("- metrics_test_ml_80_20_repeats.csv (+ _summary)")

if __name__ == "__main__":
    main()
