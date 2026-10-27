# Progress

| Phase | Status | Notes |
|---|---|---|
| 0. Setup | ✅ done | venv (Python 3.12.6), pinned requirements, ruff + pytest config, folders, .gitignore, docs |
| 1. Dataset registry | ✅ done | 4 YAML configs; `io.load_config / read_raw / apply_mapping`; clear errors with close-match hints; xlsx parsed once and cached as pickle; all 4 datasets load |
| 2. Schema + cleaning | ⏳ next | |
| 3. Leakage guard + EDA | | |
| 4. Transactions mode | | |
| 5. Modelling | | |
| 6. Cost-based threshold | | |
| 7. Explain + segments | | |
| 8. Cross-industry run | | |
| 9. Streamlit app | | |
| 10. Ship | | |

## Known issues
- None yet.

## Later (ideas, not in scope yet)
- Survival analysis (lifelines) — PRD stretch, only after phase 10.
- Calibration plots — PRD stretch.
