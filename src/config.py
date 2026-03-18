from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "AI_B_CANCER_STUDY_TNZ_Niyazi_CLEANED_STAGE2.xlsx"
SHEET = "ModelReady"

RNG_SEED = 42
TEST_N = 20
KFOLDS = 10
MAX_RULES = 250

# Stage 2-group mapping (I–II vs III–IV)
STAGE2_MAP = {"Unknown": 0.0, "I-II": 0.5, "III-IV": 1.0}
STAGE2_CENTERS = [0.0, 0.5, 1.0]

# Inputs (exact columns from your file)
X_COLS = [
    "Zone_val",
    "Age_val",
    "Family_val",
    "Menarche_val",
    "Menopause_val",
    "FirstPreg_val",
    "Breastfeed_val",
    "Contraceptives_val",
    "ER_val",
    "PR_val",
    "HER2_val",
]

# Outputs (exact columns from your file)
Y_STAGE2_STR_COL = "Stage_2grp"
Y_DIAG_COL = "Diagnosis_val"
Y_LAT_COL = "Laterality_val"
Y_IHC_COL = "IHC_val"
