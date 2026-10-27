"""Project-wide constants.

Kept in one place so every split, model and sample uses the same seed and paths,
which makes every number in `reports/` reproducible.
"""

import math
import os
from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "datasets"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

CGROUP_ROOT = Path("/sys/fs/cgroup")


def _cgroup_cpu_limit(root: Path = CGROUP_ROOT) -> float | None:
    """CPU quota of a Linux container (cgroup v2 `cpu.max`, or v1 cfs quota/period).

    Containers often report the HOST's core count through os.cpu_count() while only being
    allowed 1–2 cores. Starting one thread per reported core then makes the threads fight over
    the few real ones — LightGBM can slow down 10–100× (seen on Streamlit Cloud).
    """
    try:
        v2 = root / "cpu.max"
        if v2.exists():
            quota, period = v2.read_text().split()[:2]
            return None if quota == "max" else int(quota) / int(period)
        q, p = root / "cpu" / "cpu.cfs_quota_us", root / "cpu" / "cpu.cfs_period_us"
        if q.exists() and p.exists():
            quota_us = int(q.read_text())
            return None if quota_us <= 0 else quota_us / int(p.read_text())
    except (OSError, ValueError):
        return None
    return None


def n_jobs(root: Path = CGROUP_ROOT) -> int:
    """Threads to use: CHURNLENS_N_JOBS if set, else min(visible cores, container quota)."""
    override = os.environ.get("CHURNLENS_N_JOBS")
    if override:
        return max(1, int(override))
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (
        os.cpu_count() or 1)
    limit = _cgroup_cpu_limit(root)
    if limit is not None:
        cores = min(cores, max(1, math.floor(limit)))
    return max(1, cores)
