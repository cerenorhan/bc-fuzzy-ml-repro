import pandas as pd
import numpy as np
from .config import (
    DATA_FILE, SHEET, X_COLS,
    Y_STAGE2_STR_COL, Y_DIAG_COL, Y_LAT_COL, Y_IHC_COL,
    STAGE2_MAP
)

def load_xy():
    df = pd.read_excel(DATA_FILE, sheet_name=SHEET)

    required = X_COLS + [Y_STAGE2_STR_COL, Y_DIAG_COL, Y_LAT_COL, Y_IHC_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{SHEET}': {missing}")

    X = df[X_COLS].astype(float).to_numpy()

    # Stage2 numeric
    y_stage2 = df[Y_STAGE2_STR_COL].astype(str).map(STAGE2_MAP).fillna(0.0).astype(float).to_numpy()

    # Other outputs already numeric (0..1)
    y_diag = df[Y_DIAG_COL].astype(float).to_numpy()
    y_lat  = df[Y_LAT_COL].astype(float).to_numpy()
    y_ihc  = df[Y_IHC_COL].astype(float).to_numpy()

    Y = np.column_stack([y_stage2, y_diag, y_lat, y_ihc])
    return df, X, Y
