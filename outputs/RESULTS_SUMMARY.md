# Results summary (repeated stratified 80/20 splits)

Evaluation used 5 repeated 80/20 splits (seeds: 7, 13, 21, 42, 99).
Stage was evaluated as a binary endpoint (I–II vs III–IV). 'Unknown' Stage values were treated as missing and excluded from Stage training and evaluation.
Predicted Stage labels for unknown cases are provided separately (not treated as ground truth).
Reported metrics are mean ± SD across repeats.

## Best ML model per target (by weighted-F1)
- **Diagnosis**: rf — weighted-F1 0.814 ± 0.071
- **IHC**: logreg — weighted-F1 0.963 ± 0.013
- **Laterality**: logreg — weighted-F1 0.493 ± 0.038
- **Stage**: logreg — weighted-F1 0.767 ± 0.000

## Best ML model per target (by macro-F1)
- **Diagnosis**: rf — macro-F1 0.353 ± 0.063
- **IHC**: logreg — macro-F1 0.948 ± 0.021
- **Laterality**: logreg — macro-F1 0.402 ± 0.108
- **Stage**: svm_rbf_bal — macro-F1 0.476 ± 0.033

## Fuzzy (single-output Wang–Mendel Mamdani) per target (by weighted-F1)
- **Diagnosis**: fuzzy — weighted-F1 0.167 ± 0.068
- **IHC**: fuzzy — weighted-F1 0.290 ± 0.037
- **Laterality**: fuzzy — weighted-F1 0.326 ± 0.073
- **Stage**: fuzzy — weighted-F1 0.129 ± 0.039

## Notes
- Macro-F1 and balanced accuracy are recommended for imbalanced targets, since accuracy/weighted-F1 can be dominated by the majority class.
- Fuzzy rules were derived from data using Wang–Mendel extraction; the original hand-crafted rule base was not available.

