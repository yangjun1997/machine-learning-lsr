import json
import re
from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np

from mechine_learning_lsr.data import load_workbook, locked_split
from mechine_learning_lsr.privacy import check_aggregate_text


RESULTS = Path("results")
FIGURES = RESULTS / "图"
TABLES = RESULTS / "表"
FIGURE_STEMS = [
    "Figure_1_study_flow",
    "Figure_2_model_comparison",
    "Figure_3_shap_interpretation",
    "Figure_4_temporal_validation",
    "Figure_5_exploratory_interaction",
]
TABLE_FILES = [
    "Table_1_baseline_characteristics.md",
    "Table_2_model_comparison.md",
    "Table_3_temporal_validation.md",
]


def _svg_text(stem: str) -> str:
    return " ".join(ElementTree.parse(FIGURES / f"{stem}.svg").getroot().itertext())


def test_publication_package_is_complete_and_source_consistent():
    canonical = [RESULTS / "SCI_Methods_Results.md", RESULTS / "publication_manifest.md"]
    canonical += [FIGURES / f"{stem}.{suffix}" for stem in FIGURE_STEMS for suffix in ("png", "svg")]
    canonical += [TABLES / name for name in TABLE_FILES]
    assert all(path.is_file() for path in canonical)

    link_sources = [RESULTS / "SCI_Methods_Results.md", TABLES / "README.md", FIGURES / "表.md"]
    for source in link_sources:
        for target in re.findall(r"!?(?:\[[^]]*\])\(([^)]+)\)", source.read_text(encoding="utf-8")):
            assert (source.parent / target).resolve().is_file(), f"Broken link in {source}: {target}"

    manuscript = (RESULTS / "SCI_Methods_Results.md").read_text(encoding="utf-8")
    assert "M1" not in manuscript
    assert "lsr_combo" not in manuscript
    assert "Logistic regression with L1" not in manuscript
    assert "Logistic L2" not in manuscript
    dev, washout, val = locked_split(load_workbook("datas/580-analysis.xlsx"))
    assert f"{len(dev)} Group 1 patients" in manuscript
    assert f"{len(val)} Group 2 patients" in manuscript
    assert f"{int(dev.outcome.sum())}/{len(dev)}" in manuscript
    assert f"{int(val.outcome.sum())}/{len(val)}" in manuscript
    assert len(washout) == 0

    metrics = json.loads(Path("reports/group2_supplementary_metrics.json").read_text(encoding="utf-8"))
    for key in ("auroc", "auprc", "brier", "log_loss", "calibration_intercept", "calibration_slope"):
        estimate = metrics["point_estimates"][key]
        lower, upper = metrics["point_estimate_ci"][key]
        assert f"{estimate:.3f}" in manuscript
        assert f"{lower:.3f}-{upper:.3f}" in manuscript

    figure_3_text = _svg_text("Figure_3_shap_interpretation")
    assert "Mandibular-branch LSR" in figure_3_text
    assert "Zygomatic-branch LSR" in figure_3_text
    assert "Prior acupuncture" in figure_3_text
    assert "Man lsr" not in figure_3_text
    figure_2_text = _svg_text("Figure_2_model_comparison")
    assert "Logistic regression (elastic net)" in figure_2_text
    assert "Logistic regression (L1)" not in figure_2_text
    assert "Logistic regression (L2)" not in figure_2_text
    assert "Gaussian naive Bayes" in figure_2_text
    assert "LightGBM" in figure_2_text
    figure_1_text = _svg_text("Figure_1_study_flow")
    assert "M1" not in figure_1_text
    assert "14 fixed candidates" in figure_1_text
    figure_4_text = _svg_text("Figure_4_temporal_validation")
    assert "Exploratory threshold performance" in figure_4_text
    assert "Confusion matrix (threshold=0.10)" in figure_4_text
    figure_5_text = _svg_text("Figure_5_exploratory_interaction")
    assert "Other vessel (not estimable)" in figure_5_text
    assert "NE" in figure_5_text
    assert "<60 years" in figure_5_text and ">=60 years" in figure_5_text
    figure_5_svg = (FIGURES / "Figure_5_exploratory_interaction.svg").read_text(encoding="utf-8")
    assert "fill: url(" not in figure_5_svg
    assert "<pattern" not in figure_5_svg
    assert figure_5_svg.count("stroke: #333333") >= 10
    assert figure_5_svg.count("Residual LSR") >= 1

    table_1_text = (TABLES / "Table_1_baseline_characteristics.md").read_text(encoding="utf-8")
    assert "| Statistical method |" in table_1_text
    assert "Mann-Whitney U test (two-sided)" in table_1_text
    assert "Fisher exact test (two-sided, 2 x 2)" in table_1_text
    assert "Pearson chi-square test (3 x 2)" in table_1_text
    table_2_text = (TABLES / "Table_2_model_comparison.md").read_text(encoding="utf-8")
    assert "Logistic regression (elastic net)" in table_2_text
    assert "Logistic regression (L1)" not in table_2_text
    assert "Logistic regression (L2)" not in table_2_text

    for stem in FIGURE_STEMS:
        assert _svg_text(stem).strip(), f"Empty SVG: {stem}"
        pixels = plt.imread(FIGURES / f"{stem}.png")
        assert pixels.size and np.nanstd(pixels[..., :3]) > 0.01, f"Blank figure: {stem}"

    text_files = [path for path in canonical if path.suffix == ".md"] + [TABLES / "README.md", FIGURES / "表.md"]
    assert not check_aggregate_text(text_files)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_files).lower()
    prohibited = [
        r"\bexternal validation\b",
        r"\bprospective validation\b",
        r"gradient boosting (?:was|is) superior",
        r"(?:establishes|established|demonstrates|demonstrated) clinical (?:utility|benefit)",
    ]
    assert not any(re.search(pattern, combined) for pattern in prohibited)
    assert "draft with unresolved author inputs" in manuscript.lower()
    assert "[[author input required:" in manuscript.lower()
