# Results summary (repeated stratified 80/20 splits)

Evaluation used 5 repeated 80/20 train/test splits stratified by Stage2 (seeds: 7, 13, 21, 42, 99).
Reported metrics are mean ± SD across repeats.

## Best ML model per target (by weighted-F1)
- **diagnosis**: logreg — weighted-F1 0.809 ± 0.063
- **ihc**: logreg — weighted-F1 0.977 ± 0.021
- **laterality**: rf — weighted-F1 0.506 ± 0.040
- **stage2**: logreg — weighted-F1 0.695 ± 0.000

## Best ML model per target (by macro-F1)
- **diagnosis**: svm_rbf_bal — macro-F1 0.378 ± 0.080
- **ihc**: logreg — macro-F1 0.969 ± 0.025
- **laterality**: svm_rbf_bal — macro-F1 0.378 ± 0.043
- **stage2**: gb — macro-F1 0.303 ± 0.042

## Fuzzy (single-output Wang–Mendel Mamdani) per target (by weighted-F1)
- **diagnosis**: fuzzy — weighted-F1 0.150 ± 0.045
- **ihc**: fuzzy — weighted-F1 0.327 ± 0.068
- **laterality**: fuzzy — weighted-F1 0.381 ± 0.099
- **stage2**: fuzzy — weighted-F1 0.084 ± 0.013

## Notes
- Macro-F1 and balanced accuracy are recommended for imbalanced targets (e.g., Stage2), since accuracy/weighted-F1 may be dominated by the majority class.
- The fuzzy implementation here is based on data-driven Wang–Mendel rule extraction (the original hand-crafted rule base from the manuscript was not available).

