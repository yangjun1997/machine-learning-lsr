from mechine_learning_lsr.metrics import calibration_metrics, classification_metrics


def test_metrics_are_finite():
    result = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["brier"] >= 0
    assert result["auroc"] == 1.0
    calibration = calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert calibration["calibration_slope"] is not None
