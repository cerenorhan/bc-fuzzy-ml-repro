import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

from .config import KFOLDS, RNG_SEED, MAX_RULES, STAGE2_CENTERS
from .io_utils import load_xy
from .split_utils import train_test_split_indices
from .fuzzy_wm import wang_mendel_train, mamdani_predict
from .metrics_utils import eval_target, weighted_f1_from_confusion
from .ml_models import get_models, to_class

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

    # =========================================================
    # FUZZY v2: train separate single-output fuzzy models
    # =========================================================
    rows = []

    # CV
    for fold, (tr, va) in enumerate(kf.split(Xtr_all), 1):
        Xtr, Xva = Xtr_all[tr], Xtr_all[va]

        for target_idx, target_name in enumerate(TARGETS):
            ytr = Ytr_all[tr, target_idx:target_idx+1]   # (n,1)
            yva = Ytr_all[va, target_idx:target_idx+1]   # (n,1)

            in_mfs, out_mfs, rules = wang_mendel_train(Xtr, ytr, max_rules=MAX_RULES)
            yhat = mamdani_predict(Xva, in_mfs, out_mfs, rules)[:, 0]

            acc, wf1, _ = eval_target(yva[:, 0], yhat, centers_map[target_name])
            rows.append({"fold": fold, "target": target_name, "model": "fuzzy_v2_single", "accuracy": acc, "weighted_f1": wf1})

    pd.DataFrame(rows).to_csv(outdir / "metrics_cv_fuzzy_v2.csv", index=False)

    # Independent test (train on full train set per target)
    test_rows = []
    for target_idx, target_name in enumerate(TARGETS):
        ytr = Ytr_all[:, target_idx:target_idx+1]
        yte = Yte[:, target_idx]

        in_mfs, out_mfs, rules = wang_mendel_train(Xtr_all, ytr, max_rules=MAX_RULES)
        yhat_te = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]

        acc, wf1, C = eval_target(yte, yhat_te, centers_map[target_name])
        test_rows.append({"target": target_name, "model": "fuzzy_v2_single", "accuracy": acc, "weighted_f1": wf1})
        pd.DataFrame(C).to_csv(outdir / f"confusion_{target_name}_fuzzy_v2.csv", index=False)

    pd.DataFrame(test_rows).to_csv(outdir / "metrics_test_fuzzy_v2.csv", index=False)

    # =========================================================
    # ML baselines (same as before)
    # =========================================================
    ml_models = get_models(seed=RNG_SEED)

    rows_ml = []
    test_rows_ml = []

    for target_idx, target_name in enumerate(TARGETS):
        centers = centers_map[target_name]
        y_all = to_class(Ytr_all[:, target_idx], centers)
        y_test = to_class(Yte[:, target_idx], centers)

        for model_name, model in ml_models.items():
            # CV
            for fold, (tr, va) in enumerate(kf.split(Xtr_all), 1):
                Xtr, Xva = Xtr_all[tr], Xtr_all[va]
                ytr, yva = y_all[tr], y_all[va]

                model.fit(Xtr, ytr)
                pred = model.predict(Xva)

                acc = float((pred == yva).mean())
                C = confusion_matrix(yva, pred)
                wf1 = weighted_f1_from_confusion(C)

                rows_ml.append({"fold": fold, "target": target_name, "model": model_name, "accuracy": acc, "weighted_f1": wf1})

            # Independent test
            model.fit(Xtr_all, y_all)
            pred_te = model.predict(Xte)

            acc_te = float((pred_te == y_test).mean())
            Cte = confusion_matrix(y_test, pred_te)
            wf1_te = weighted_f1_from_confusion(Cte)

            test_rows_ml.append({"target": target_name, "model": model_name, "accuracy": acc_te, "weighted_f1": wf1_te})
            pd.DataFrame(Cte).to_csv(outdir / f"confusion_{target_name}_{model_name}.csv", index=False)

    pd.DataFrame(rows_ml).to_csv(outdir / "metrics_cv_ml.csv", index=False)
    pd.DataFrame(test_rows_ml).to_csv(outdir / "metrics_test_ml.csv", index=False)

    print("Done. Wrote fuzzy v2 + ML outputs to outputs/")

if __name__ == "__main__":
    main()
