from mechine_learning_lsr.features import get_features
from mechine_learning_lsr.data import load_workbook
from mechine_learning_lsr.models import candidates


def test_all_core_candidates_build():
    df = load_workbook("datas/580-analysis.xlsx")
    X, _, feature_set = get_features(df, "m1")
    names = {c.name for c in candidates(feature_set.numeric, feature_set.categorical)}
    assert {"logistic_l1", "logistic_l2", "logistic_elasticnet", "random_forest", "mlp"} <= names
