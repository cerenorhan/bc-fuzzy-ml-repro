import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .config import RNG_SEED, MAX_RULES, STAGE2_CENTERS
from .io_utils import load_xy
from .fuzzy_wm import wang_mendel_train, mamdani_predict
from .metrics_utils import eval_target, weighted_f1_from_confusion
from .ml_models import get_models, to_class

TARGETS = ["stage2", "diagnosis", "laterality", "ihc"]

def centers_from_train(y):
    c = np.unique(y.astype(float))
    return np.sort(c).tolist()

def main():
    df, X, Y = load_xy()

    # Stratify by stage2 class for stable split
    y_stage2_class = to_class(Y[:, 0], STAGE2_CENTERS)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=RNG_SEED,
        shuffle=True,
        stratify=y_stage2_class
    )

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    # centers derived from TRAIN for non-stage2 outputs
    centers_map = {
        "stage2": STAGE2_CENTERS,
        "diagnosis": centers_from_train(Ytr[:, 1]),
        "laterality": centers_from_train(Ytr[:, 2]),
        "ihc": centers_from_train(Ytr[:, 3]),
    }

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    # Save split stats
    split_info = {
        "n_total": len(df),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "stage2_train_counts": dict(pd.Series(y_stage2_class[train_idx]).value_counts().sort_index()),
        "stage2_test_counts": dict(pd.Series(y_stage2_class[test_idx]).value_counts().sort_index()),
    }
    pd.Series(split_info).to_json(outdir / "split_80_20_info.json", indent=2)

    # =========================
    # FUZZY (single-output per target)
    # =========================
    fuzzy_rows = []
    for target_idx, target_name in enumerate(TARGETS):
        ytr = Ytr[:, target_idx:target_idx+1]  # (n,1)
        yte = Yte[:, target_idx]              # (n,)
        in_mfs, out_mfs, rules = wang_mendel_train(Xtr, ytr, max_rules=MAX_RULES)
        yhat = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]

        acc, wf1, C = eval_target(yte, yhat, centers_map[target_name])
        fuzzy_rows.append({"target": target_name, "model": "fuzzy_80_20", "accuracy": acc, "weighted_f1": wf1})
        pd.DataFrame(C).to_csv(outdir / f"confusion_{target_name}_fuzzy_80_20.csv", index=False)

    pd.DataFrame(fuzzy_rows).to_csv(outdir / "metrics_test_fuzzy_80_20.csv", index=False)

    # =========================
    # ML baselines
    # =========================
    ml_models = get_models(seed=RNG_SEED)
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

            ml_rows.append({"target": target_name, "model": model_name, "accuracy": acc, "weighted_f1": wf1})
            pd.DataFrame(C).to_csv(outdir / f"confusion_{target_name}_{model_name}_80_20.csv", index=False)

    pd.DataFrame(ml_rows).to_csv(outdir / "metrics_test_ml_80_20.csv", index=False)

    print("Done. Wrote 80/20 results to outputs/:")
    print("- split_80_20_info.json")
    print("- metrics_test_fuzzy_80_20.csv, metrics_test_ml_80_20.csv")
    print("- confusion_*_80_20.csv")

if __name__ == "__main__":
    main()
