from pathlib import Path

from mechine_learning_lsr.audit import audit


def test_real_workbook_passes_contract():
    _, result = audit(Path("datas/580-analysis.xlsx"))
    assert result.passed
    assert result.rows == 580
    assert result.columns == 24
    assert result.group_counts == {"组1": 476, "组2": 104}
    assert result.event_counts == {"组1": 58, "组2": 9}
    assert result.washout_rows == 0
