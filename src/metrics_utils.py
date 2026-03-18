import numpy as np
from sklearn.metrics import confusion_matrix

def nearest_class(y, centers):
    centers = np.asarray(centers, float)
    y = np.asarray(y, float)
    return np.argmin(np.abs(y[:, None] - centers[None, :]), axis=1)

def weighted_f1_from_confusion(C):
    C = np.asarray(C, float)
    support = C.sum(axis=1)
    N = support.sum()
    f1 = np.zeros(C.shape[0], float)
    for k in range(C.shape[0]):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k, :].sum() - tp
        prec = tp / max(tp + fp, 1e-12)
        rec  = tp / max(tp + fn, 1e-12)
        f1[k] = (2 * prec * rec) / max(prec + rec, 1e-12)
    w = support / max(N, 1e-12)
    return float((w * f1).sum())

def macro_f1_from_confusion(C):
    C = np.asarray(C, float)
    f1 = np.zeros(C.shape[0], float)
    for k in range(C.shape[0]):
        tp = C[k, k]
        fp = C[:, k].sum() - tp
        fn = C[k, :].sum() - tp
        prec = tp / max(tp + fp, 1e-12)
        rec  = tp / max(tp + fn, 1e-12)
        f1[k] = (2 * prec * rec) / max(prec + rec, 1e-12)
    return float(np.mean(f1))

def balanced_accuracy_from_confusion(C):
    C = np.asarray(C, float)
    recalls = []
    for k in range(C.shape[0]):
        tp = C[k, k]
        fn = C[k, :].sum() - tp
        rec = tp / max(tp + fn, 1e-12)
        recalls.append(rec)
    return float(np.mean(recalls))

def eval_target(y_true, y_pred, centers):
    yt = nearest_class(y_true, centers)
    yp = nearest_class(y_pred, centers)
    acc = float((yt == yp).mean())
    C = confusion_matrix(yt, yp)
    wf1 = weighted_f1_from_confusion(C)
    mf1 = macro_f1_from_confusion(C)
    bacc = balanced_accuracy_from_confusion(C)
    return acc, wf1, mf1, bacc, C
