from churnlens import config


def test_random_state_is_fixed() -> None:
    assert config.RANDOM_STATE == 42


def test_paths_are_inside_project() -> None:
    assert config.DATA_RAW.is_relative_to(config.ROOT)
    assert config.REPORTS_DIR.is_relative_to(config.ROOT)
