from __future__ import annotations

from pathlib import Path
import re


def check_aggregate_text(paths: list[str | Path]) -> list[str]:
    findings = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        if re.search(r"(?<!\d)\d{6}(?!\d)", text):
            findings.append(str(path))
    return findings
