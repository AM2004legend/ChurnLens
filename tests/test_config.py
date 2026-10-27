import pytest

from churnlens import config


def test_random_state_is_fixed() -> None:
    assert config.RANDOM_STATE == 42


def test_paths_are_inside_project() -> None:
    assert config.DATA_RAW.is_relative_to(config.ROOT)
    assert config.REPORTS_DIR.is_relative_to(config.ROOT)


def test_cgroup_v2_quota_limits_threads(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CHURNLENS_N_JOBS", raising=False)
    (tmp_path / "cpu.max").write_text("200000 100000\n")  # 2 CPUs
    assert config._cgroup_cpu_limit(tmp_path) == 2.0
    assert 1 <= config.n_jobs(tmp_path) <= 2


def test_cgroup_v2_unlimited(tmp_path) -> None:
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert config._cgroup_cpu_limit(tmp_path) is None


def test_cgroup_v1_fractional_quota_rounds_down_to_one(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("CHURNLENS_N_JOBS", raising=False)
    (tmp_path / "cpu").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("50000")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000")
    assert config.n_jobs(tmp_path) == 1


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CHURNLENS_N_JOBS", "3")
    assert config.n_jobs() == 3


@pytest.mark.parametrize("content", ["garbage", ""])
def test_unreadable_cgroup_is_ignored(tmp_path, content: str) -> None:
    (tmp_path / "cpu.max").write_text(content)
    assert config._cgroup_cpu_limit(tmp_path) is None
