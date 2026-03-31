from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from .expert_rules import EXPERT_RULES, OUTPUT_LABEL_MAP


def trapmf(x: float, a: float, b: float, c: float, d: float) -> float:
    x = float(x)
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    if c < x < d:
        return (d - x) / (d - c) if d != c else 0.0
    return 0.0


def trimf(x: float, a: float, b: float, c: float) -> float:
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if a < x < b:
        return (x - a) / (b - a) if b != a else 0.0
    if b < x < c:
        return (c - x) / (c - b) if c != b else 0.0
    return 0.0


def _to_float_or_nan(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return np.nan


def membership_degree(var_name: str, term: str, value: Any) -> float:
    if pd.isna(value):
        return 0.0

    if var_name == "FirstPreg_val":
        x = _to_float_or_nan(value)
        if np.isnan(x):
            return 0.0
        if term == "EARLY":
            return trapmf(x, 0, 0, 18, 24)
        if term == "MID":
            return trimf(x, 20, 26, 32)
        if term == "LATE":
            return trapmf(x, 28, 34, 60, 60)

    if var_name == "Breastfeed_val":
        x = _to_float_or_nan(value)
        if np.isnan(x):
            return 0.0
        if term == "SHORT":
            return trapmf(x, 0, 0, 3, 8)
        if term == "MEDIUM":
            return trimf(x, 6, 12, 18)
        if term == "LONG":
            return trapmf(x, 15, 20, 60, 60)

    if var_name == "Menarche_val":
        x = _to_float_or_nan(value)
        if np.isnan(x):
            return 0.0
        if term == "EARLY":
            return trapmf(x, 0, 0, 11, 12.5)
        if term == "MID":
            return trimf(x, 11.5, 13, 14.5)
        if term == "LATE":
            return trapmf(x, 13.5, 15, 25, 25)

    if var_name == "Menopause_val":
        x = _to_float_or_nan(value)
        if np.isnan(x):
            return 0.0
        if term == "EARLY":
            return trapmf(x, 0, 0, 42, 47)
        if term == "MID":
            return trimf(x, 45, 50, 55)
        if term == "LATE":
            return trapmf(x, 53, 57, 80, 80)

    s = str(value).strip().upper()

    if var_name in {"ER_val", "PR_val", "HER2_val", "Family_val", "Contraceptives_val"}:
        pos_tokens = {"1", "1.0", "POS", "POSITIVE", "YES", "TRUE"}
        neg_tokens = {"0", "0.0", "NEG", "NEGATIVE", "NO", "FALSE"}

        if term in {"POSITIVE", "YES"}:
            return 1.0 if s in pos_tokens else 0.0
        if term in {"NEGATIVE", "NO"}:
            return 1.0 if s in neg_tokens else 0.0

    if var_name == "Zone_val":
        lake_tokens = {"LAKE_REGION", "LAKE", "1", "1.0"}
        if term == "LAKE_REGION":
            return 1.0 if s in lake_tokens else 0.0
        if term == "OTHER":
            return 0.0 if s in lake_tokens else 1.0

    return 0.0


@dataclass
class ExpertFuzzyMultiOutput:
    rules: List[Dict[str, Any]]
    output_label_map: Dict[str, Dict[str, str]]
    majority_label_by_target: Dict[str, str]

    @classmethod
    def fit(cls, train_df: pd.DataFrame) -> "ExpertFuzzyMultiOutput":
        majority = {}
        for target in ["Stage", "Diagnosis", "Laterality", "IHC"]:
            vc = train_df[target].astype(str).value_counts(dropna=True)
            if vc.empty:
                raise ValueError(f"No labels found for target '{target}'")
            majority[target] = vc.index[0]

        return cls(
            rules=EXPERT_RULES,
            output_label_map=OUTPUT_LABEL_MAP,
            majority_label_by_target=majority,
        )

    def _fire_rule(self, row: pd.Series, rule: Dict[str, Any]) -> float:
        degs = []
        for var_name, term in rule["if"]:
            deg = membership_degree(var_name, term, row.get(var_name, np.nan))
            degs.append(deg)
        return min(degs) if degs else 0.0

    def score_row(self, row: pd.Series) -> Dict[str, Dict[str, float]]:
        scores = {
            "Stage": {},
            "Diagnosis": {},
            "Laterality": {},
            "IHC": {},
        }

        for rule in self.rules:
            strength = self._fire_rule(row, rule)
            if strength <= 0:
                continue

            target_name, out_term = rule["then"]
            mapped = self.output_label_map[target_name].get(out_term)
            if mapped is None:
                continue

            prev = scores[target_name].get(mapped, 0.0)
            scores[target_name][mapped] = max(prev, strength)

        return scores

    def predict_row(self, row: pd.Series) -> Dict[str, Dict[str, Any]]:
        scores = self.score_row(row)
        out = {}

        for target_name in ["Stage", "Diagnosis", "Laterality", "IHC"]:
            if len(scores[target_name]) == 0:
                out[target_name] = {
                    "pred_label": self.majority_label_by_target[target_name],
                    "prob_max": 0.0,
                    "fired_rule": False,
                }
            else:
                best_label, best_score = max(scores[target_name].items(), key=lambda kv: kv[1])
                out[target_name] = {
                    "pred_label": best_label,
                    "prob_max": float(best_score),
                    "fired_rule": True,
                }

        return out

    def predict_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for idx, row in df.iterrows():
            pred = self.predict_row(row)
            rec = {"row_index": idx}
            for target_name in ["Stage", "Diagnosis", "Laterality", "IHC"]:
                rec[f"{target_name}_pred"] = pred[target_name]["pred_label"]
                rec[f"{target_name}_prob_max"] = pred[target_name]["prob_max"]
                rec[f"{target_name}_fired_rule"] = pred[target_name]["fired_rule"]
            rows.append(rec)
        return pd.DataFrame(rows)
