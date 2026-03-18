import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def to_class(y_numeric, centers):
    centers = np.asarray(centers, float)
    y = np.asarray(y_numeric, float)
    return np.argmin(np.abs(y[:, None] - centers[None, :]), axis=1)

def get_models(seed=42):
    return {
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "logreg_bal": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, class_weight="balanced"))
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf"))
        ]),
        "svm_rbf_bal": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced"))
        ]),
        "rf": RandomForestClassifier(n_estimators=500, random_state=seed),
        "gb": GradientBoostingClassifier(random_state=seed),
    }
