from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from src.io_utils import load_xy

def normalize_stage2(s: str) -> str:
    s = str(s).strip().replace("–","-").replace("—","-")
    return s

def stage2_binary(series: pd.Series):
    s = series.astype(str).map(normalize_stage2)
    y = pd.Series(np.nan, index=s.index, dtype=float)
    y.loc[s.eq("I-II")] = 0.0
    y.loc[s.eq("III-IV")] = 1.0
    is_unknown = y.isna() | s.str.lower().eq("unknown")
    return y, is_unknown

def main():
    df, X, _ = load_xy()
    y, is_unknown = stage2_binary(df["Stage_2grp"])

    known = ~is_unknown.to_numpy()
    unk = is_unknown.to_numpy()

    Xk, yk = X[known], y[known].astype(int).to_numpy()
    Xu = X[unk]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ])
    model.fit(Xk, yk)

    proba = model.predict_proba(Xu)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    thr = 0.80
    out = pd.DataFrame({
        "PATIENT ID": df.loc[unk, "PATIENT ID"].values,
        "Stage2_pseudolabel": np.where(pred==0, "I-II", "III-IV"),
        "confidence": np.round(conf, 4),
        "high_confidence": conf >= thr
    }).sort_values(["high_confidence","confidence"], ascending=[False,False])

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/secondary_selftrain_stage2_pseudolabels.csv", index=False)
    print("Wrote outputs/secondary_selftrain_stage2_pseudolabels.csv")

if __name__ == "__main__":
    main()

