import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold

from .config import KFOLDS, RNG_SEED, MAX_RULES, STAGE2_CENTERS
from .io_utils import load_xy
from .split_utils import train_test_split_indices
from .fuzzy_wm import mamdani_predict
from .fuzzy_wm_cov import wang_mendel_train_coverage
from .metrics_utils import eval_target

TARGETS = ["stage2", "diagnosis", "laterality", "ihc"]

def centers_from_train(y):
    c = np.unique(y.astype(float))
    return np.sort(c).tolist()

def main():
    df, X, Y = load_xy()
    train_idx, test_idx = train_test_split_indices(len(df))

    Xtr_all, Ytr_all = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    centers_map = {
        "stage2": STAGE2_CENTERS,
        "diagnosis": centers_from_train(Ytr_all[:, 1]),
        "laterality": centers_from_train(Ytr_all[:, 2]),
        "ihc": centers_from_train(Ytr_all[:, 3]),
    }

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=RNG_SEED)

    rows = []
    # CV
    for fold, (tr, va) in enumerate(kf.split(Xtr_all), 1):
        Xtr, Xva = Xtr_all[tr], Xtr_all[va]
        for target_idx, target_name in enumerate(TARGETS):
            ytr = Ytr_all[tr, target_idx]
            yva = Ytr_all[va, target_idx]

            # coverage-aware rules
            in_mfs, out_mfs, rules = wang_mendel_train_coverage(
                Xtr, ytr, max_rules=MAX_RULES,
                per_class_min=20, cap_majority_frac=0.6
            )
            yhat = mamdani_predict(Xva, in_mfs, out_mfs, rules)[:, 0]

            acc, wf1, _ = eval_target(yva, yhat, centers_map[target_name])
            rows.append({"fold": fold, "target": target_name, "model": "fuzzy_v4_cov", "accuracy": acc, "weighted_f1": wf1})

    pd.DataFrame(rows).to_csv(outdir / "metrics_cv_fuzzy_v4.csv", index=False)

    # Independent test (train per target on all training)
    test_rows = []
    for target_idx, target_name in enumerate(TARGETS):
        ytr = Ytr_all[:, target_idx]
        yte = Yte[:, target_idx]

        in_mfs, out_mfs, rules = wang_mendel_train_coverage(
            Xtr_all, ytr, max_rules=MAX_RULES,
            per_class_min=20, cap_majority_frac=0.6
        )
        yhat_te = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]
        acc, wf1, C = eval_target(yte, yhat_te, centers_map[target_name])

        test_rows.append({"target": target_name, "model": "fuzzy_v4_cov", "accuracy": acc, "weighted_f1": wf1})
        pd.DataFrame(C).to_csv(outdir / f"confusion_{target_name}_fuzzy_v4.csv", index=False)

    pd.DataFrame(test_rows).to_csv(outdir / "metrics_test_fuzzy_v4.csv", index=False)
    print("Done. Wrote fuzzy v4 outputs to outputs/")

if __name__ == "__main__":
    main()
