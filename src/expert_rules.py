from __future__ import annotations

EXPERT_RULES = [
    {
        "name": "rule_1",
        "if": [
            ("FirstPreg_val", "EARLY"),
            ("ER_val", "POSITIVE"),
        ],
        "then": ("IHC", "LUMINAL_B"),
    },
    {
        "name": "rule_2",
        "if": [
            ("Breastfeed_val", "SHORT"),
            ("PR_val", "NEGATIVE"),
        ],
        "then": ("Stage", "ADVANCED"),
    },
    {
        "name": "rule_3",
        "if": [
            ("Menarche_val", "LATE"),
            ("HER2_val", "POSITIVE"),
        ],
        "then": ("IHC", "HER2_ENRICHED"),
    },
    {
        "name": "rule_4",
        "if": [
            ("Family_val", "POSITIVE"),
            ("ER_val", "NEGATIVE"),
        ],
        "then": ("Diagnosis", "INVASIVE_DUCTAL_CARCINOMA"),
    },
    {
        "name": "rule_5",
        "if": [
            ("Contraceptives_val", "YES"),
            ("Menopause_val", "EARLY"),
        ],
        "then": ("Stage", "EARLY"),
    },
    {
        "name": "rule_6",
        "if": [
            ("Zone_val", "LAKE_REGION"),
            ("HER2_val", "NEGATIVE"),
        ],
        "then": ("Laterality", "LEFT"),
    },
]

# IMPORTANT:
# Stage mapping is final and compatible with current benchmark labels.
# The remaining mappings may need to be adjusted to your actual dataset labels.
OUTPUT_LABEL_MAP = {
    "Stage": {
        "EARLY": "I-II",
        "ADVANCED": "III-IV",
    },
    "IHC": {
        "LUMINAL_B": "0.5",
        "HER2_ENRICHED": "1.0",
    },
    "Diagnosis": {
        "INVASIVE_DUCTAL_CARCINOMA": "0.5",
    },
    "Laterality": {
        "LEFT": "0.5",
    },
}
