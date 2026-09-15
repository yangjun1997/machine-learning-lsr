import numpy as np
import pandas as pd

from mechine_learning_lsr.models import candidates


def test_synthetic_model_smoke():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"age": rng.normal(55, 8, 40), "duration": rng.uniform(1, 8, 40), "sex": rng.integers(0, 2, 40), "zyg_lsr": rng.integers(1, 4, 40)})
    y = np.array([0, 1] * 20)
    pool = candidates(("age", "duration"), ("sex", "zyg_lsr"))
    candidate = next(c for c in pool if c.name == "logistic_l2")
    candidate.estimator.fit(frame, y)
    assert candidate.estimator.predict_proba(frame).shape == (40, 2)
