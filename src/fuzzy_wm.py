import numpy as np

def trapmf(x, a, b, c, d):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    y = np.where((x >= b) & (x <= c), 1.0, y)
    left = (x - a) / max(b - a, 1e-12)
    right = (d - x) / max(d - c, 1e-12)
    y = np.where((x > a) & (x < b), left, y)
    y = np.where((x > c) & (x < d), right, y)
    return np.clip(y, 0.0, 1.0)

def trimf(x, a, b, c):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    left = (x - a) / max(b - a, 1e-12)
    right = (c - x) / max(c - b, 1e-12)
    y = np.where((x > a) & (x < b), left, y)
    y = np.where((x > b) & (x < c), right, y)
    y = np.where(x == b, 1.0, y)
    return np.clip(y, 0.0, 1.0)

def build_mfs_from_centers(centers):
    centers = np.array(sorted(np.unique(np.asarray(centers, float))))
    if centers.size < 2:
        centers = np.array([0.0, 1.0])

    mfs = []
    for k, c in enumerate(centers):
        if k == 0:
            r = (centers[min(1, len(centers)-1)] + centers[0]) / 2.0
            mfs.append(("trap", (0.0, 0.0, r, min(1.0, r + 1e-6))))
        elif k == len(centers) - 1:
            l = (centers[-2] + centers[-1]) / 2.0
            mfs.append(("trap", (max(0.0, l - 1e-6), l, 1.0, 1.0)))
        else:
            l = (centers[k-1] + centers[k]) / 2.0
            r = (centers[k] + centers[k+1]) / 2.0
            mfs.append(("tri", (l, c, r)))
    return centers, mfs

def eval_mf_scalar(x, mf):
    t, p = mf
    if t == "trap":
        return float(trapmf(x, *p))
    return float(trimf(x, *p))

def wang_mendel_train(X, Y, max_rules=250):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    n, p = X.shape
    m = Y.shape[1]

    in_mfs = []
    for i in range(p):
        centers, mfs = build_mfs_from_centers(np.unique(X[:, i]))
        in_mfs.append((centers, mfs))

    out_mfs = []
    for j in range(m):
        centers, mfs = build_mfs_from_centers(np.unique(Y[:, j]))
        out_mfs.append((centers, mfs))

    rule_dict = {}  # antecedent tuple -> (consequent tuple, weight)

    for r in range(n):
        ant = []
        ant_deg = []
        for i in range(p):
            _, mfs = in_mfs[i]
            degs = [eval_mf_scalar(X[r, i], mf) for mf in mfs]
            ix = int(np.argmax(degs))
            ant.append(ix)
            ant_deg.append(degs[ix])

        cons = []
        cons_deg = []
        for j in range(m):
            _, mfs = out_mfs[j]
            degs = [eval_mf_scalar(Y[r, j], mf) for mf in mfs]
            ix = int(np.argmax(degs))
            cons.append(ix)
            cons_deg.append(degs[ix])

        w = float(np.prod(ant_deg) * np.prod(cons_deg))
        key = tuple(ant)
        if key not in rule_dict or w > rule_dict[key][1]:
            rule_dict[key] = (tuple(cons), w)

    rules = [(k, v[0], v[1]) for k, v in rule_dict.items()]
    rules.sort(key=lambda x: x[2], reverse=True)
    rules = rules[:max_rules]
    return in_mfs, out_mfs, rules

def mamdani_predict(X, in_mfs, out_mfs, rules, grid_n=101):
    X = np.asarray(X, float)
    n, p = X.shape
    m = len(out_mfs)

    grid = np.linspace(0.0, 1.0, grid_n)

    # Precompute output MF curves on grid
    out_curves = []
    for j in range(m):
        _, mfs = out_mfs[j]
        curves = []
        for mf in mfs:
            t, params = mf
            if t == "trap":
                curves.append(trapmf(grid, *params))
            else:
                curves.append(trimf(grid, *params))
        out_curves.append(np.vstack(curves))  # (num_mf, grid_n)

    Yhat = np.zeros((n, m), float)

    for r in range(n):
        agg = [np.zeros(grid_n, float) for _ in range(m)]

        for ant_idx, cons_idx, w in rules:
            fire = 1.0
            for i in range(p):
                _, mfs = in_mfs[i]
                fire = min(fire, eval_mf_scalar(X[r, i], mfs[ant_idx[i]]))
                if fire == 0.0:
                    break
            if fire == 0.0:
                continue

            # NOTE: rule weight is used only for rule selection (top-K); do not cap firing here.

            for j in range(m):
                mf_curve = out_curves[j][cons_idx[j], :]
                implied = np.minimum(mf_curve, fire)
                agg[j] = np.maximum(agg[j], implied)

        for j in range(m):
            mu = agg[j]
            denom = mu.sum()
            Yhat[r, j] = 0.0 if denom <= 1e-12 else float((grid * mu).sum() / denom)

    return Yhat
