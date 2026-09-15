from __future__ import annotations

from dataclasses import dataclass
import importlib
import warnings

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class Candidate:
    name: str
    estimator: object
    status: str = "available"
    note: str = ""


def _pipeline(estimator, numeric, categorical, scale=True):
    num_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), list(numeric)),
        ("cat", cat, list(categorical)),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def candidates(numeric, categorical, seed: int = 20260904) -> list[Candidate]:
    common = dict(max_iter=3000, random_state=seed)
    raw = [
        ("logistic_l1", LogisticRegression(penalty="l1", solver="saga", C=1.0, **common), True),
        ("logistic_l2", LogisticRegression(penalty="l2", solver="lbfgs", C=1.0, **common), True),
        ("logistic_elasticnet", LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0, **common), True),
        ("lda", LinearDiscriminantAnalysis(), True),
        ("qda", QuadraticDiscriminantAnalysis(reg_param=0.1), True),
        ("gaussian_nb", GaussianNB(), True),
        ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance"), True),
        ("svm_linear", SVC(kernel="linear", probability=True, C=1.0, random_state=seed), True),
        ("svm_rbf", SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=seed), True),
        ("decision_tree", DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, class_weight="balanced", random_state=seed), False),
        ("random_forest", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed), False),
        ("extra_trees", ExtraTreesClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed), False),
        ("adaboost", AdaBoostClassifier(n_estimators=150, learning_rate=0.03, random_state=seed), False),
        ("gradient_boosting", GradientBoostingClassifier(n_estimators=100, learning_rate=0.03, max_depth=2, min_samples_leaf=10, random_state=seed), False),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=7, l2_regularization=1.0, random_state=seed), False),
        ("mlp", MLPClassifier(hidden_layer_sizes=(16,), alpha=0.01, max_iter=500, early_stopping=True, random_state=seed), True),
    ]
    result = [Candidate(name, _pipeline(estimator, numeric, categorical, scale)) for name, estimator, scale in raw]
    for package, cls_name in (("xgboost", "XGBClassifier"), ("lightgbm", "LGBMClassifier"), ("catboost", "CatBoostClassifier")):
        try:
            cls = getattr(importlib.import_module(package), cls_name)
            if package == "catboost":
                estimator = cls(iterations=150, depth=3, learning_rate=0.03, random_state=seed, verbose=False)
            elif package == "lightgbm":
                estimator = cls(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=seed, verbosity=-1)
            else:
                estimator = cls(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=seed, eval_metric="logloss", verbosity=0)
            result.append(Candidate(package, _pipeline(estimator, numeric, categorical, False)))
        except (ImportError, AttributeError, TypeError) as exc:
            result.append(Candidate(package, None, "unavailable", str(exc)))
    return result


def available_candidates(numeric, categorical, seed=20260904):
    return [c for c in candidates(numeric, categorical, seed) if c.status == "available"]
