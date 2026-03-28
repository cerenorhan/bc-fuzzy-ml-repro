
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .config import MAX_RULES
from .io_utils import load_xy
from .fuzzy_wm import wang_mendel_train, mamdani_predict
from .metrics_utils import (
    eval_target,
    weighted_f1_from_confusion,
    macro_f1_from_confusion,
    balanced_accuracy_from_confusion,
)
from .ml_models import get_models, to_class

TARGETS = ["stage", "diagnosis", "laterality", "ihc"]

def centers_from_train(y):
    c = np.unique(y.astype(float))
    return np.sort(c).tolist()

def normalize_stage(s: str) -> str:
    s = str(s).strip()
    # normalize unicode dashes
    s = s.replace("–", "-").replace("—", "-")
    return s

def stage_binary_from_str(series: pd.Series):
    s = series.astype(str).map(normalize_stage)
    is_unknown = s.str.lower().eq("unknown") | s.str.lower().eq("nan") | s.eq("")
    # binary labels: I-II -> 0, III-IV -> 1
    y = pd.Series(np.nan, index=s.index, dtype=float)
    y.loc[s.eq("I-II")] = 0.0
    y.loc[s.eq("III-IV")] = 1.0
    # Anything not recognized is treated as unknown/missing
    is_unknown = is_unknown | y.isna()
    return y, is_unknown

def run_one_split(df, X, Y, seed, test_size=0.20, conf_thr=0.80):
    # -----------------------------
    # Split strategy:
    # 1) Split ONLY known stage samples stratified (binary)
    # 2) Assign unknown stage samples randomly to train/test to preserve ratio
    # -----------------------------
    y_stage_bin, is_unknown = stage_binary_from_str(df["Stage_2grp"])

    known_idx = np.where(~is_unknown.to_numpy())[0]
    unk_idx   = np.where(is_unknown.to_numpy())[0]

    y_known = y_stage_bin.iloc[known_idx].astype(int).to_numpy()

    tr_known, te_known = train_test_split(
        known_idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y_known
    )

    # Randomly allocate unknown stage samples to train/test
    rng = np.random.default_rng(seed)
    unk_perm = rng.permutation(unk_idx)
    n_te_unk = int(round(test_size * len(unk_idx)))
    te_unk = unk_perm[:n_te_unk]
    tr_unk = unk_perm[n_te_unk:]

    train_idx = np.sort(np.concatenate([tr_known, tr_unk]))
    test_idx  = np.sort(np.concatenate([te_known, te_unk]))

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    # Build centers for non-stage outputs from TRAIN (as before)
    centers_map = {
        # stage handled separately (binary)
        "diagnosis": centers_from_train(Ytr[:, 1]),
        "laterality": centers_from_train(Ytr[:, 2]),
        "ihc": centers_from_train(Ytr[:, 3]),
    }

    # -----------------------------
    # FUZZY + ML for STAGE2 (binary, unknown excluded in train+eval)
    # -----------------------------
    stage_rows_fuzzy = []
    stage_rows_ml = []
    unknown_pred_rows = []

    # stage: prepare train/eval masks inside this split
    is_unknown_split = is_unknown.to_numpy()
    tr_stage = np.intersect1d(train_idx, known_idx)
    te_stage = np.intersect1d(test_idx, known_idx)

    Xtr_s = X[tr_stage]
    Xte_s = X[te_stage]

    ytr_s = y_stage_bin.iloc[tr_stage].astype(float).to_numpy()
    yte_s = y_stage_bin.iloc[te_stage].astype(float).to_numpy()

    # Define binary centers [0,1]
    STAGE2_BIN_CENTERS = [0.0, 1.0]

    # FUZZY stage (train only known)
    # For fuzzy we keep numeric y in {0,1} and treat it as classification via centers.
    in_mfs, out_mfs, rules = wang_mendel_train(Xtr_s, ytr_s.reshape(-1,1), max_rules=MAX_RULES)
    yhat_s = mamdani_predict(Xte_s, in_mfs, out_mfs, rules)[:, 0]
    acc, wf1, mf1, bacc, _ = eval_target(yte_s, yhat_s, STAGE2_BIN_CENTERS)
    stage_rows_fuzzy.append({
        "seed": seed, "target": "stage", "model": "fuzzy",
        "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "balanced_acc": bacc
    })

    # ML stage (binary)
    ml_models = get_models(seed=seed)
    # We will use logreg/svm/rf/gb variants already included; for stage use binary y
    ytr_cls = ytr_s.astype(int)
    yte_cls = yte_s.astype(int)

    for model_name, model in ml_models.items():
        model.fit(Xtr_s, ytr_cls)
        pred = model.predict(Xte_s)
        C = confusion_matrix(yte_cls, pred)
        wf1_m = weighted_f1_from_confusion(C)
        mf1_m = macro_f1_from_confusion(C)
        bacc_m = balanced_accuracy_from_confusion(C)
        acc_m = float((pred == yte_cls).mean())
        stage_rows_ml.append({
            "seed": seed, "target": "stage", "model": model_name,
            "accuracy": acc_m, "weighted_f1": wf1_m, "macro_f1": mf1_m, "balanced_acc": bacc_m
        })

    # -----------------------------
    # Predict stage for Unknown (no evaluation)
    # -----------------------------
    # Use best-weightedF1 primary model choice: logreg (stable) as default
    # Train on known-train only, predict unknowns in this split
    model_for_unknown = ml_models["logreg"]
    model_for_unknown.fit(Xtr_s, ytr_cls)

    unk_in_test = np.intersect1d(test_idx, unk_idx)
    if len(unk_in_test) > 0:
        proba = model_for_unknown.predict_proba(X[unk_in_test])
        pred_u = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        # map 0->I-II, 1->III-IV
        lab = np.where(pred_u==0, "I-II", "III-IV")
        tmp = pd.DataFrame({
            "seed": seed,
            "PATIENT ID": df.loc[unk_in_test, "PATIENT ID"].values,
            "Stage_predicted": lab,
            "confidence": np.round(conf, 4),
            "high_confidence": conf >= conf_thr
        })
        unknown_pred_rows.append(tmp)

    # -----------------------------
    # Other targets (diagnosis/laterality/ihc) — unchanged, use all samples in split
    # -----------------------------
    fuzzy_rows = []
    ml_rows = []

    # FUZZY for other targets (single-output)
    for target_idx, target_name in [(1,"diagnosis"), (2,"laterality"), (3,"ihc")]:
        ytr = Ytr[:, target_idx:target_idx+1]
        yte = Yte[:, target_idx]

        in_mfs, out_mfs, rules = wang_mendel_train(Xtr, ytr, max_rules=MAX_RULES)
        yhat = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]

        acc, wf1, mf1, bacc, _ = eval_target(yte, yhat, centers_map[target_name])
        fuzzy_rows.append({
            "seed": seed, "target": target_name, "model": "fuzzy",
            "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "balanced_acc": bacc
        })

    # ML for other targets
    for target_idx, target_name in [(1,"diagnosis"), (2,"laterality"), (3,"ihc")]:
        centers = centers_map[target_name]
        ytr_cls2 = to_class(Ytr[:, target_idx], centers)
        yte_cls2 = to_class(Yte[:, target_idx], centers)

        for model_name, model in ml_models.items():
            model.fit(Xtr, ytr_cls2)
            pred = model.predict(Xte)

            C = confusion_matrix(yte_cls2, pred)
            wf1_m = weighted_f1_from_confusion(C)
            mf1_m = macro_f1_from_confusion(C)
            bacc_m = balanced_accuracy_from_confusion(C)
            acc_m = float((pred == yte_cls2).mean())

            ml_rows.append({
                "seed": seed, "target": target_name, "model": model_name,
                "accuracy": acc_m, "weighted_f1": wf1_m, "macro_f1": mf1_m, "balanced_acc": bacc_m
            })

    # Combine outputs
    fz_df = pd.DataFrame(stage_rows_fuzzy + fuzzy_rows)
    ml_df = pd.DataFrame(stage_rows_ml + ml_rows)
    unk_df = pd.concat(unknown_pred_rows, ignore_index=True) if unknown_pred_rows else pd.DataFrame()

    # split bookkeeping
    split_info = {
        "seed": seed,
        "n_total": len(df),
        "n_known_stage": int(len(known_idx)),
        "n_unknown_stage": int(len(unk_idx)),
        "n_train_total": int(len(train_idx)),
        "n_test_total": int(len(test_idx)),
        "n_train_known_stage": int(len(tr_stage)),
        "n_test_known_stage": int(len(te_stage)),
        "n_test_unknown_stage": int(len(unk_in_test)),
    }

    return fz_df, ml_df, unk_df, split_info

def summarize(df):
    return (df.groupby(["target","model"])[["accuracy","weighted_f1","macro_f1","balanced_acc"]]
              .agg(["mean","std"])
              .reset_index())

def main():
    df, X, Y = load_xy()

    seeds = [7, 13, 21, 42, 99]  # 5 repeats
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    all_fuzzy = []
    all_ml = []
    all_unk = []
    split_infos = []

    for s in seeds:
        fz, ml, unk, info = run_one_split(df, X, Y, seed=s, test_size=0.20, conf_thr=0.80)
        all_fuzzy.append(fz)
        all_ml.append(ml)
        if len(unk):
            all_unk.append(unk)
        split_infos.append(info)

    fz_df = pd.concat(all_fuzzy, ignore_index=True)
    ml_df = pd.concat(all_ml, ignore_index=True)
    unk_df = pd.concat(all_unk, ignore_index=True) if all_unk else pd.DataFrame()
    split_df = pd.DataFrame(split_infos)

    # Overwrite same filenames to keep pipeline stable
    fz_df.to_csv(outdir / "Fuzzy_detailed.csv", index=False)
    ml_df.to_csv(outdir / "ML_detailed.csv", index=False)

    fz_sum = summarize(fz_df)
    ml_sum = summarize(ml_df)

    fz_sum.to_csv(outdir / "Fuzzy_summary.csv", index=False)
    ml_sum.to_csv(outdir / "ML_summary.csv", index=False)

    split_df.to_csv(outdir / "Split_info.csv", index=False)

    # Unknown stage predictions (exploratory; no metrics)
    if len(unk_df):
        unk_df.to_csv(outdir / "predicted_stage_for_unknown_repeats.csv", index=False)

    print("Done. Updated run_benchmark with binary Stage (Unknown treated as missing).")
    print("- outputs/metrics_test_*_repeats_v2metrics.csv (+ summary) updated")
    print("- outputs/Split_info.csv written")
    if len(unk_df):
        print("- outputs/predicted_stage_for_unknown_repeats.csv written (secondary; not used as ground truth)")

if __name__ == "__main__":
    main()
