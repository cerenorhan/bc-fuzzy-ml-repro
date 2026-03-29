from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler

from src.io_utils import load_xy

def normalize_stage(s: str) -> str:
    s = str(s).strip().replace("–","-").replace("—","-")
    return s

def stage_binary(series: pd.Series):
    s = series.astype(str).map(normalize_stage)
    y = pd.Series(np.nan, index=s.index, dtype=float)
    y.loc[s.eq("I-II")] = 0.0
    y.loc[s.eq("III-IV")] = 1.0
    is_unknown = y.isna() | s.str.lower().eq("unknown")
    return y, is_unknown

def main():
    df, X, _ = load_xy()
    y, is_unknown = stage_binary(df["Stage_2grp"])

    y_lp = np.full(len(df), -1, dtype=int)
    known_idx = np.where(~is_unknown.to_numpy())[0]
    y_lp[known_idx] = y.iloc[known_idx].astype(int).to_numpy()

    Xs = StandardScaler().fit_transform(X)

    lp = LabelSpreading(kernel="rbf", gamma=20, max_iter=200)
    lp.fit(Xs, y_lp)

    proba = lp.label_distributions_
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)

    unk_idx = np.where(is_unknown.to_numpy())[0]
    out = pd.DataFrame({
        "PATIENT ID": df.loc[unk_idx, "PATIENT ID"].values,
        "Stage_labelprop": np.where(pred[unk_idx]==0, "I-II", "III-IV"),
        "confidence": np.round(conf[unk_idx], 4),
        "high_confidence": conf[unk_idx] >= 0.80
    }).sort_values(["high_confidence","confidence"], ascending=[False,False])

    Path("outputs").mkdir(exist_ok=True)
    out.to_csv("outputs/Stage_unknown_labelprop.csv", index=False)
    print("Wrote outputs/Stage_unknown_labelprop.csv")

if __name__ == "__main__":
    main()

