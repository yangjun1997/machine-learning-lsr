from mechine_learning_lsr.data import load_workbook
from mechine_learning_lsr.features import get_features


def test_m1_excludes_postoperative_fields():
    df = load_workbook("datas/580-analysis.xlsx")
    X, y, feature_set = get_features(df, "m1")
    assert "day7_spasm" not in X.columns
    assert "length_of_stay" not in X.columns
    assert "zyg_lsr" in X.columns
    assert len(X) == len(y) == 580
