import xml.etree.ElementTree as ET

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from mechine_learning_lsr.interaction_report import _plot_all580_forest, _plot_combined, _plot_strata


@pytest.mark.parametrize("age_p", [0.0165, 0.05, float("nan")])
def test_strata_use_forest_wald_p_values(tmp_path, age_p):
    interactions = pd.DataFrame({
        "modifier": ["age", "duration", "pica"],
        "label": ["LSR x age", "LSR x duration", "LSR x pica"],
        "wald_p": [age_p, 0.05, float("nan")],
        "interaction_or": [2.4, 1.3, 0.2],
        "ci_low": [1.2, 0.6, 0.01],
        "ci_high": [5.0, 5.0, 0.9],
    })
    strata = pd.DataFrame([
        {"modifier": modifier, "modifier_group": group, "lsr_residual": residual, "n": 10, "events": residual + 1}
        for modifier, groups in [("age_ge_60", ["<60", ">=60"]), ("duration_ge_3y", ["<3", ">=3"]), ("pica", ["absent", "present"])]
        for group in groups
        for residual in [0, 1]
    ])
    with matplotlib.rc_context({"svg.fonttype": "none"}):
        _plot_strata(strata, tmp_path, interactions)
        _plot_all580_forest(interactions, tmp_path)
        _plot_combined(strata, interactions, tmp_path)
    strata_text = " ".join(ET.parse(tmp_path / "17_interaction_stratified_rates.svg").getroot().itertext())
    forest_text = " ".join(ET.parse(tmp_path / "19_all580_lsr_interaction_forest.svg").getroot().itertext())
    combined_text = " ".join(ET.parse(tmp_path / "19_17_combined_lsr_interaction.svg").getroot().itertext())
    assert ("age_ge_60: <60" in strata_text) == (age_p < 0.05)
    assert ("age_ge_60: >=60" in strata_text) == (age_p < 0.05)
    assert "duration_ge_3y" not in strata_text
    assert "pica:" not in strata_text
    assert ("No interaction has Wald p < 0.05" in strata_text) == (not age_p < 0.05)
    assert "LSR x duration (Wald p=0.0500)" in forest_text
    assert "LSR x pica (Wald p=NA)" in forest_text
    expected_p = f"{age_p:.4f}" if pd.notna(age_p) else "NA"
    assert f"LSR x age (Wald p={expected_p})" in forest_text
    assert "A" in combined_text and "B" in combined_text
    assert "Interaction test" in combined_text
    assert "Age-stratified observed event rates" in combined_text
