from pathlib import Path

from mechine_learning_lsr.privacy import check_aggregate_text


def test_reports_do_not_expose_six_digit_identifiers():
    paths = [p for p in Path("reports").glob("*.md") if p.is_file()]
    assert not check_aggregate_text(paths)
