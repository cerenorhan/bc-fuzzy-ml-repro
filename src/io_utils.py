import numpy as np
import pandas as pd

from .config import (
    DATA_FILE,
    SHEET,
    X_COLS,
    Y_STAGE2_STR_COL,
    Y_DIAG_COL,
    Y_LAT_COL,
    Y_IHC_COL,
)


def load_xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET)

    required = X_COLS + [Y_STAGE2_STR_COL, Y_DIAG_COL, Y_LAT_COL, Y_IHC_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{SHEET}': {missing}")

    X = df[X_COLS].astype(float).to_numpy()

    stage_str = (
        df[Y_STAGE2_STR_COL]
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
    )
    y_stage = stage_str.map({"I-II": 0.0, "III-IV": 1.0}).astype(float).to_numpy()

    y_diag = df[Y_DIAG_COL].astype(float).to_numpy()
    y_lat = df[Y_LAT_COL].astype(float).to_numpy()
    y_ihc = df[Y_IHC_COL].astype(float).to_numpy()

    Y = np.column_stack([y_stage, y_diag, y_lat, y_ihc])
    return df, X, Y
