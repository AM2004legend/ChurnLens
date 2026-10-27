"""Fill docs/README_template.md with the tables in reports/results.md -> README.md.

Run after `python -m churnlens.run --all` so README numbers always come from code output.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def table_starting_with(lines: list[str], header: str) -> str:
    """The markdown table whose first line starts with `header`."""
    start = next(i for i, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        rows.append(line)
    return "\n".join(rows)


def main() -> None:
    lines = (ROOT / "reports" / "results.md").read_text(encoding="utf-8").splitlines()
    readme = (ROOT / "docs" / "README_template.md").read_text(encoding="utf-8")
    readme = readme.replace("{{RESULTS_TABLE}}", table_starting_with(lines, "| dataset | mode"))
    readme = readme.replace("{{SENSITIVITY_TABLE}}",
                            table_starting_with(lines, "| inactivity_days"))
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("README.md rebuilt from reports/results.md")


if __name__ == "__main__":
    main()
