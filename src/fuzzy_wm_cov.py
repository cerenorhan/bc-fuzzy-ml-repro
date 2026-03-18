import numpy as np
from .fuzzy_wm import build_mfs_from_centers, eval_mf_scalar, trapmf, trimf

def wang_mendel_train_coverage(X, y, max_rules=250, per_class_min=20, cap_majority_frac=0.6):
    """
    Single-output Wang–Mendel with coverage-aware rule selection.

    - Ensures at least `per_class_min` rules per class (based on output MF index).
    - Limits the fraction of rules from the majority class to `cap_majority_frac`.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    n, p = X.shape

    # Build input MFs
    in_mfs = []
    for i in range(p):
        centers, mfs = build_mfs_from_centers(np.unique(X[:, i]))
        in_mfs.append((centers, mfs))

    # Build output MFs (from unique output values)
    out_centers, out_mfs_list = build_mfs_from_centers(np.unique(y[:, 0]))
    out_mfs = [(out_centers, out_mfs_list)]  # keep same structure as mamdani_predict expects (list per output)

    # Extract best rule per antecedent
    rule_dict = {}  # ant tuple -> (cons_idx, weight)
    for r in range(n):
        ant = []
        ant_deg = []
        for i in range(p):
            _, mfs = in_mfs[i]
            degs = [eval_mf_scalar(X[r, i], mf) for mf in mfs]
            ix = int(np.argmax(degs))
            ant.append(ix)
            ant_deg.append(degs[ix])

        # consequent (single output)
        degs_out = [eval_mf_scalar(y[r, 0], mf) for mf in out_mfs_list]
        cons_idx = int(np.argmax(degs_out))
        cons_deg = float(degs_out[cons_idx])

        w = float(np.prod(ant_deg) * cons_deg)
        key = tuple(ant)
        if key not in rule_dict or w > rule_dict[key][1]:
            rule_dict[key] = (cons_idx, w)

    # Convert to list grouped by class
    rules_by_class = {}
    for ant, (cidx, w) in rule_dict.items():
        rules_by_class.setdefault(cidx, []).append((ant, cidx, w))

    # Sort within each class by weight
    for cidx in rules_by_class:
        rules_by_class[cidx].sort(key=lambda x: x[2], reverse=True)

    # Identify majority class by available rules
    class_sizes = {c: len(rules_by_class.get(c, [])) for c in rules_by_class}
    majority_class = max(class_sizes, key=class_sizes.get)

    selected = []

    # 1) Guarantee per-class minimum (except cap majority later)
    for cidx, lst in rules_by_class.items():
        take = min(per_class_min, len(lst))
        selected.extend(lst[:take])

    # 2) Fill remaining by global weight, but cap majority contribution
    remaining = max_rules - len(selected)
    if remaining > 0:
        # Build global candidate list excluding already selected antecedents
        selected_set = set([r[0] for r in selected])

        candidates = []
        for cidx, lst in rules_by_class.items():
            for ant, c, w in lst:
                if ant in selected_set:
                    continue
                candidates.append((ant, c, w))
        candidates.sort(key=lambda x: x[2], reverse=True)

        # majority cap
        max_majority = int(max_rules * cap_majority_frac)
        current_majority = sum(1 for r in selected if r[1] == majority_class)

        for ant, c, w in candidates:
            if len(selected) >= max_rules:
                break
            if c == majority_class and current_majority >= max_majority:
                continue
            selected.append((ant, c, w))
            if c == majority_class:
                current_majority += 1

    # Format rules for mamdani_predict: (ant_idx_tuple, cons_idx_tuple, weight)
    rules = [(ant, (cidx,), w) for (ant, cidx, w) in selected]
    return in_mfs, out_mfs, rules
