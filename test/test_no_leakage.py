from sklearn.model_selection import train_test_split

from mechine_learning_lsr.data import load_workbook
from mechine_learning_lsr.features import get_features
from mechine_learning_lsr.models import candidates


def test_train_validation_indices_are_disjoint_and_transformer_fits_train_only():
    df = load_workbook("datas/580-analysis.xlsx")
    X, y, feature_set = get_features(df, "m1")
    train, valid = train_test_split(range(len(y)), test_size=0.2, random_state=3, stratify=y)
    assert set(train).isdisjoint(valid)
    candidate = next(c for c in candidates(feature_set.numeric, feature_set.categorical) if c.name == "logistic_l2")
    candidate.estimator.fit(X.iloc[train], y.iloc[train])
    transformed = candidate.estimator.named_steps["preprocess"].transform(X.iloc[valid])
    assert transformed.shape[0] == len(valid)
