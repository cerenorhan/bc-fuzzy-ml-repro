import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from .config import MAX_RULES
from .io_utils import load_xy
from .fuzzy_wm import wang_mendel_train, mamdani_predict
from .expert_fuzzy import ExpertFuzzyMultiOutput
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
    s = s.replace("–", "-").replace("—", "-")
    return s


def stage_binary_from_str(series: pd.Series):
    s = series.astype(str).map(normalize_stage)
    is_unknown = s.str.lower().eq("unknown") | s.str.lower().eq("nan") | s.eq("")
    y = pd.Series(np.nan, index=s.index, dtype=float)
    y.loc[s.eq("I-II")] = 0.0
    y.loc[s.eq("III-IV")] = 1.0
    is_unknown = is_unknown | y.isna()
    return y, is_unknown


def evaluate_expert_fuzzy_on_split(df, train_idx, test_idx, known_idx, seed):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # Normalize target column names for expert fuzzy model
    rename_map = {
        "Stage_2grp": "Stage",
        "Diagnosis_val": "Diagnosis",
        "Laterality_val": "Laterality",
        "IHC_val": "IHC",
    }
    train_df = train_df.rename(columns=rename_map)
    test_df = test_df.rename(columns=rename_map)

    model = ExpertFuzzyMultiOutput.fit(train_df)
    pred_df = model.predict_dataframe(test_df).reset_index(drop=True)

    summary_rows = []
    detailed_rows = []

    target_map = {
        "stage": ("Stage", "Stage"),
        "diagnosis": ("Diagnosis", "Diagnosis"),
        "laterality": ("Laterality", "Laterality"),
        "ihc": ("IHC", "IHC"),
    }

    for target_out, (pred_prefix, truth_col) in target_map.items():
        if target_out == "stage":
            test_global_idx = test_df.index.to_numpy()
            keep_mask = np.isin(test_global_idx, known_idx)
            y_true = test_df.loc[keep_mask, truth_col].astype(str).reset_index(drop=True)
            y_pred = pred_df.loc[keep_mask, f"{pred_prefix}_pred"].astype(str).reset_index(drop=True)
            prob_max = pred_df.loc[keep_mask, f"{pred_prefix}_prob_max"].reset_index(drop=True)
            fired_rule = pred_df.loc[keep_mask, f"{pred_prefix}_fired_rule"].reset_index(drop=True)
            sample_index = test_df.index[keep_mask]
            sample_id = test_df.loc[keep_mask, "PATIENT ID"].values if "PATIENT ID" in test_df.columns else sample_index
        else:
            y_true = test_df[truth_col].astype(str).reset_index(drop=True)
            y_pred = pred_df[f"{pred_prefix}_pred"].astype(str).reset_index(drop=True)
            prob_max = pred_df[f"{pred_prefix}_prob_max"].reset_index(drop=True)
            fired_rule = pred_df[f"{pred_prefix}_fired_rule"].reset_index(drop=True)
            sample_index = test_df.index
            sample_id = test_df["PATIENT ID"].values if "PATIENT ID" in test_df.columns else sample_index

        if len(y_true) == 0:
            continue

        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        y_true_cls = pd.Categorical(y_true, categories=labels).codes
        y_pred_cls = pd.Categorical(y_pred, categories=labels).codes

        C = confusion_matrix(y_true_cls, y_pred_cls, labels=np.arange(len(labels)))
        wf1 = weighted_f1_from_confusion(C)
        mf1 = macro_f1_from_confusion(C)
        bacc = balanced_accuracy_from_confusion(C)
        acc = float((y_true.to_numpy() == y_pred.to_numpy()).mean())

        summary_rows.append({
            "seed": seed,
            "target": target_out,
            "model": "expert_fuzzy",
            "accuracy": acc,
            "weighted_f1": wf1,
            "macro_f1": mf1,
            "balanced_acc": bacc,
        })

        for i in range(len(y_true)):
            detailed_rows.append({
                "seed": seed,
                "target": target_out,
                "model": "expert_fuzzy",
                "sample_index": sample_index[i],
                "sample_id": sample_id[i],
                "y_true_label": str(y_true.iloc[i]),
                "y_pred_label": str(y_pred.iloc[i]),
                "prob_max": float(prob_max.iloc[i]),
                "fired_rule": bool(fired_rule.iloc[i]),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(detailed_rows)


def run_one_split(df, X, Y, seed, test_size=0.20, conf_thr=0.80):
    y_stage_bin, is_unknown = stage_binary_from_str(df["Stage_2grp"])

    known_idx = np.where(~is_unknown.to_numpy())[0]
    unk_idx = np.where(is_unknown.to_numpy())[0]

    y_known = y_stage_bin.iloc[known_idx].astype(int).to_numpy()

    tr_known, te_known = train_test_split(
        known_idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=y_known
    )

    rng = np.random.default_rng(seed)
    unk_perm = rng.permutation(unk_idx)
    n_te_unk = int(round(test_size * len(unk_idx)))
    te_unk = unk_perm[:n_te_unk]
    tr_unk = unk_perm[n_te_unk:]

    train_idx = np.sort(np.concatenate([tr_known, tr_unk]))
    test_idx = np.sort(np.concatenate([te_known, te_unk]))

    Xtr, Ytr = X[train_idx], Y[train_idx]
    Xte, Yte = X[test_idx], Y[test_idx]

    centers_map = {
        "diagnosis": centers_from_train(Ytr[:, 1]),
        "laterality": centers_from_train(Ytr[:, 2]),
        "ihc": centers_from_train(Ytr[:, 3]),
    }

    stage_rows_fuzzy = []
    stage_rows_ml = []
    unknown_pred_rows = []

    tr_stage = np.intersect1d(train_idx, known_idx)
    te_stage = np.intersect1d(test_idx, known_idx)

    Xtr_s = X[tr_stage]
    Xte_s = X[te_stage]

    ytr_s = y_stage_bin.iloc[tr_stage].astype(float).to_numpy()
    yte_s = y_stage_bin.iloc[te_stage].astype(float).to_numpy()

    STAGE2_BIN_CENTERS = [0.0, 1.0]

    # data-driven fuzzy: stage
    in_mfs, out_mfs, rules = wang_mendel_train(Xtr_s, ytr_s.reshape(-1, 1), max_rules=MAX_RULES)
    yhat_s = mamdani_predict(Xte_s, in_mfs, out_mfs, rules)[:, 0]
    acc, wf1, mf1, bacc, _ = eval_target(yte_s, yhat_s, STAGE2_BIN_CENTERS)
    stage_rows_fuzzy.append({
        "seed": seed, "target": "stage", "model": "fuzzy",
        "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "balanced_acc": bacc
    })

    # ml: stage
    ml_models = get_models(seed=seed)
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

    # unknown stage predictions (exploratory ML)
    model_for_unknown = ml_models["logreg"]
    model_for_unknown.fit(Xtr_s, ytr_cls)

    unk_in_test = np.intersect1d(test_idx, unk_idx)
    if len(unk_in_test) > 0:
        proba = model_for_unknown.predict_proba(X[unk_in_test])
        pred_u = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        lab = np.where(pred_u == 0, "I-II", "III-IV")
        tmp = pd.DataFrame({
            "seed": seed,
            "PATIENT ID": df.loc[unk_in_test, "PATIENT ID"].values,
            "Stage_predicted": lab,
            "confidence": np.round(conf, 4),
            "high_confidence": conf >= conf_thr
        })
        unknown_pred_rows.append(tmp)

    fuzzy_rows = []
    ml_rows = []

    # data-driven fuzzy for other targets
    for target_idx, target_name in [(1, "diagnosis"), (2, "laterality"), (3, "ihc")]:
        ytr = Ytr[:, target_idx:target_idx + 1]
        yte = Yte[:, target_idx]

        in_mfs, out_mfs, rules = wang_mendel_train(Xtr, ytr, max_rules=MAX_RULES)
        yhat = mamdani_predict(Xte, in_mfs, out_mfs, rules)[:, 0]

        acc, wf1, mf1, bacc, _ = eval_target(yte, yhat, centers_map[target_name])
        fuzzy_rows.append({
            "seed": seed, "target": target_name, "model": "fuzzy",
            "accuracy": acc, "weighted_f1": wf1, "macro_f1": mf1, "balanced_acc": bacc
        })

    # ml for other targets
    for target_idx, target_name in [(1, "diagnosis"), (2, "laterality"), (3, "ihc")]:
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

    # expert fuzzy
    expert_summary_df, expert_detailed_df = evaluate_expert_fuzzy_on_split(
        df=df,
        train_idx=train_idx,
        test_idx=test_idx,
        known_idx=known_idx,
        seed=seed
    )

    fz_df = pd.concat(
        [pd.DataFrame(stage_rows_fuzzy + fuzzy_rows), expert_summary_df],
        ignore_index=True
    )
    ml_df = pd.DataFrame(stage_rows_ml + ml_rows)
    unk_df = pd.concat(unknown_pred_rows, ignore_index=True) if unknown_pred_rows else pd.DataFrame()

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

    return fz_df, ml_df, unk_df, split_info, expert_detailed_df


def summarize(df):
    out = (
        df.groupby(["target", "model"])[["accuracy", "weighted_f1", "macro_f1", "balanced_acc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    out.columns = [
        "target", "model",
        "accuracy", "accuracy.1",
        "weighted_f1", "weighted_f1.1",
        "macro_f1", "macro_f1.1",
        "balanced_acc", "balanced_acc.1",
    ]
    return out


def main():
    df, X, Y = load_xy()

    seeds = [7, 13, 21, 42, 99]
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    all_fuzzy = []
    all_ml = []
    all_unk = []
    all_expert_detailed = []
    split_infos = []

    for s in seeds:
        fz, ml, unk, info, expert_detailed = run_one_split(
            df, X, Y, seed=s, test_size=0.20, conf_thr=0.80
        )
        all_fuzzy.append(fz)
        all_ml.append(ml)
        if len(unk):
            all_unk.append(unk)
        if len(expert_detailed):
            all_expert_detailed.append(expert_detailed)
        split_infos.append(info)

    fz_df = pd.concat(all_fuzzy, ignore_index=True)
    ml_df = pd.concat(all_ml, ignore_index=True)
    unk_df = pd.concat(all_unk, ignore_index=True) if all_unk else pd.DataFrame()
    split_df = pd.DataFrame(split_infos)
    expert_det_df = pd.concat(all_expert_detailed, ignore_index=True) if all_expert_detailed else pd.DataFrame()

    fz_df.to_csv(outdir / "Fuzzy_detailed.csv", index=False)
    ml_df.to_csv(outdir / "ML_detailed.csv", index=False)

    if len(expert_det_df):
        expert_det_df.to_csv(outdir / "ExpertFuzzy_detailed.csv", index=False)

    fz_sum = summarize(fz_df)
    ml_sum = summarize(ml_df)

    fz_sum.to_csv(outdir / "Fuzzy_summary.csv", index=False)
    ml_sum.to_csv(outdir / "ML_summary.csv", index=False)
    split_df.to_csv(outdir / "Split_info.csv", index=False)

    if len(unk_df):
        unk_df.to_csv(outdir / "predicted_stage_for_unknown_repeats.csv", index=False)

    print("Done. Updated benchmark with:")
    print("- data-driven fuzzy")
    print("- expert-rule multi-output fuzzy")
    print("- machine-learning baselines")
    print("- outputs/Fuzzy_summary.csv now includes model = expert_fuzzy")
    if len(expert_det_df):
        print("- outputs/ExpertFuzzy_detailed.csv written")
    if len(unk_df):
        print("- outputs/predicted_stage_for_unknown_repeats.csv written")


if __name__ == "__main__":
    main()
