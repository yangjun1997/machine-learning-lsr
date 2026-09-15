from pathlib import Path

from .audit import audit, render_report


def main() -> None:
    source = Path("datas/580-analysis.xlsx")
    _, result = audit(source)
    render_report(result, "reports/data_audit.md")
    print(f"PASS: {result.rows} rows, {result.columns} columns; report written to reports/data_audit.md")


if __name__ == "__main__":
    main()
