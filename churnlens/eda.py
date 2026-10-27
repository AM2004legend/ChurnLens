"""Exploratory analysis: who churns, and how churners differ from stayers.

Run on the training split only. Findings are generated from the numbers by fixed templates
(no LLM), so every sentence can be traced back to a value in `reports/<name>/eda.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # no GUI: figures are only saved to disk
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from churnlens import config  # noqa: E402
from churnlens.io import TARGET  # noqa: E402
from churnlens.schema import CATEGORICAL, NUMERIC  # noqa: E402
from churnlens.utils import md_table  # noqa: E402

LOW_CARDINALITY = 10  # numeric columns with ≤ this many values are also analysed as groups


@dataclass
class EdaResult:
    dataset: str
    rows: int
    churn_rate: float
    numeric: pd.DataFrame
    groups: pd.DataFrame
    findings: list[str] = field(default_factory=list)
    figures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "churn_rate": self.churn_rate,
            "numeric_differences": self.numeric.to_dict(orient="records"),
            "group_differences": self.groups.to_dict(orient="records"),
            "findings": self.findings,
            "figures": self.figures,
        }


def numeric_differences(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Compare churners vs stayers per numeric column.

    `std_diff` (difference in means divided by pooled std, a.k.a. Cohen's d) makes columns on
    different scales comparable: 0.2 small, 0.5 medium, 0.8 large.
    """
    churn = df[TARGET] == 1
    rows = []
    for col in cols:
        x = pd.to_numeric(df[col], errors="coerce")
        if x.nunique() <= 2:
            continue  # binary flags read better as group churn rates
        a, b = x[churn].dropna(), x[~churn].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        pooled = np.sqrt((a.var() + b.var()) / 2)
        rows.append(
            {
                "column": col,
                "mean_churned": round(float(a.mean()), 3),
                "mean_stayed": round(float(b.mean()), 3),
                "median_churned": round(float(a.median()), 3),
                "median_stayed": round(float(b.median()), 3),
                "std_diff": round(float((a.mean() - b.mean()) / pooled), 3) if pooled else 0.0,
            }
        )
    out = pd.DataFrame(rows, columns=["column", "mean_churned", "mean_stayed", "median_churned",
                                      "median_stayed", "std_diff"])
    return out.reindex(out["std_diff"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def group_differences(df: pd.DataFrame, cols: list[str], min_share: float = 0.01) -> pd.DataFrame:
    """Churn rate per category (and per value of low-cardinality numeric columns).

    Tiny groups are skipped (< max(30 rows, 1%)) because their rates are mostly noise.
    """
    overall = df[TARGET].mean()
    min_n = max(30, int(min_share * len(df)))
    rows = []
    for col in cols:
        values = df[col].astype("string").fillna("Missing")
        stats = df.groupby(values.values)[TARGET].agg(["size", "mean"])
        for value, (n, rate) in stats.iterrows():
            if n >= min_n:
                rows.append(
                    {
                        "column": col,
                        "value": str(value),
                        "n": int(n),
                        "churn_rate": round(float(rate), 4),
                        "lift": round(float(rate / overall), 3) if overall else 0.0,
                    }
                )
    out = pd.DataFrame(rows, columns=["column", "value", "n", "churn_rate", "lift"])
    gap = (out["churn_rate"] - overall).abs()
    return out.reindex(gap.sort_values(ascending=False).index).reset_index(drop=True)


def _num(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def write_findings(
    churn_rate: float, numeric: pd.DataFrame, groups: pd.DataFrame, n: int = 5
) -> list[str]:
    """Five plain-English findings: overall rate, then the strongest differences (one per
    column, alternating group and numeric so both kinds of evidence appear)."""
    findings = [f"Overall, {_pct(churn_rate)} of customers in the training data churned."]
    group_lines, numeric_lines, seen = [], [], set()
    for row in groups.itertuples():
        if row.column in seen:
            continue
        seen.add(row.column)
        direction = "more" if row.churn_rate > churn_rate else "less"
        times = row.churn_rate / churn_rate if row.churn_rate > churn_rate else (
            churn_rate / row.churn_rate if row.churn_rate else float("inf"))
        group_lines.append(
            f"Customers with {row.column} = {row.value} churn at {_pct(row.churn_rate)} "
            f"vs {_pct(churn_rate)} overall ({times:.1f}× {direction} likely; n={row.n:,})."
        )
    for row in numeric.itertuples():
        if row.column in seen:
            continue
        seen.add(row.column)
        word = "higher" if row.std_diff > 0 else "lower"
        size = "large" if abs(row.std_diff) >= 0.8 else "medium" if abs(row.std_diff) >= 0.5 \
            else "small"
        numeric_lines.append(
            f"Churners have {word} {row.column} (median {_num(row.median_churned)} vs "
            f"{_num(row.median_stayed)} for stayers; {size} difference, d={row.std_diff:+.2f})."
        )
    while len(findings) < n and (group_lines or numeric_lines):
        for pool in (group_lines, numeric_lines):
            if pool and len(findings) < n:
                findings.append(pool.pop(0))
    return findings


def _plot_groups(groups: pd.DataFrame, churn_rate: float, dataset: str, top: int = 4) -> str:
    cols = list(dict.fromkeys(groups["column"]))[:top]
    fig, axes = plt.subplots(1, max(len(cols), 1), figsize=(4 * max(len(cols), 1), 3.6),
                             squeeze=False)
    for ax, col in zip(axes[0], cols, strict=False):
        g = groups[groups["column"] == col].sort_values("churn_rate")
        ax.barh(g["value"].str.slice(0, 18), g["churn_rate"] * 100, color="#4C78A8")
        ax.axvline(churn_rate * 100, color="#E45756", ls="--", lw=1, label="overall")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("churn rate %")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: churn rate by group (training data)", fontsize=11)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{dataset}_eda_groups.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(config.ROOT).as_posix())


def _plot_numeric(df: pd.DataFrame, numeric: pd.DataFrame, dataset: str, top: int = 4) -> str:
    cols = numeric["column"].head(top).tolist()
    fig, axes = plt.subplots(1, max(len(cols), 1), figsize=(4 * max(len(cols), 1), 3.6),
                             squeeze=False)
    for ax, col in zip(axes[0], cols, strict=False):
        x = pd.to_numeric(df[col], errors="coerce")
        data = [x[df[TARGET] == 0].dropna(), x[df[TARGET] == 1].dropna()]
        ax.boxplot(data, tick_labels=["stayed", "churned"], showfliers=False)
        ax.set_title(col, fontsize=10)
    fig.suptitle(f"{dataset}: strongest numeric differences (training data)", fontsize=11)
    fig.tight_layout()
    path = config.FIGURES_DIR / f"{dataset}_eda_numeric.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path.relative_to(config.ROOT).as_posix())


def run_eda(df: pd.DataFrame, types: dict[str, str], dataset: str,
            save_figures: bool = True) -> EdaResult:
    """Full EDA on a (training) frame with standard `churn` column."""
    numeric_cols = [c for c, k in types.items() if k == NUMERIC]
    group_cols = [c for c, k in types.items() if k == CATEGORICAL] + [
        c for c in numeric_cols if df[c].nunique() <= LOW_CARDINALITY
    ]
    churn_rate = float(df[TARGET].mean())
    numeric = numeric_differences(df, numeric_cols)
    groups = group_differences(df, group_cols)
    result = EdaResult(
        dataset=dataset,
        rows=len(df),
        churn_rate=round(churn_rate, 4),
        numeric=numeric,
        groups=groups,
        findings=write_findings(churn_rate, numeric, groups),
    )
    if save_figures:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        if not groups.empty:
            result.figures.append(_plot_groups(groups, churn_rate, dataset))
        if not numeric.empty:
            result.figures.append(_plot_numeric(df, numeric, dataset))
    return result


def save_eda(result: EdaResult) -> None:
    out_dir = config.REPORTS_DIR / result.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eda.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    md = [f"# EDA — {result.dataset}", "",
          f"Training rows: {result.rows:,} · churn rate: {_pct(result.churn_rate)}", "",
          "## Findings", ""]
    md += [f"{i}. {f}" for i, f in enumerate(result.findings, 1)]
    md += ["", "## Strongest numeric differences", "", md_table(result.numeric, 8),
           "", "## Groups furthest from the overall churn rate", "", md_table(result.groups, 10)]
    (out_dir / "eda.md").write_text("\n".join(md) + "\n", encoding="utf-8")
