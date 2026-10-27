# Paste these into Claude Code, one at a time

## Before you start
1. Create an empty `churnlens` folder and open it in VS Code.
2. Put `CLAUDE.md` in the root, `PRD.md` in `docs/`, and `settings.json` in `.claude/`
   (so the path is `churnlens/.claude/settings.json`).
3. Make sure Git knows it's you (once per laptop):
   ```powershell
   git config --global user.name "Aritra Maity"
   git config --global user.email "your-github-email@example.com"
   ```
4. Download the 4 datasets listed in docs/PRD.md into `data/raw/` (create the folder). Kaggle needs
   you to be logged in. Rename them to `bank.csv`, `ecommerce.csv` (or .xlsx), `subscription.csv`,
   `retail.xlsx` (or .csv).

## Prompt 1 — orientation (no code)

Read CLAUDE.md and docs/PRD.md fully. Don't write any code yet.

Then:
1. Explain the project back to me in one paragraph, in simple words.
2. Explain the two input modes and why the transactions mode needs a time-based split.
3. Confirm the workflow you'll follow each phase and the Git rules (especially: no co-author or
   "Generated with" lines). Check that `.claude/settings.json` exists and tell me what it contains.
4. Give me the Phase 0 plan. Tell me what I need installed and the PowerShell commands to check.

Wait for my "go".

## Prompt 2 — Phase 0

Go ahead with Phase 0. After the first commit, run `git log -1 --format=full` and show me the
output so I can confirm the commit has only me as author and no co-author line. Then stop.

## Prompt for every later phase

Start Phase N. Follow the workflow in CLAUDE.md exactly: plan → wait for my go → implement → test →
verify on real data → explain → update PROGRESS.md and DECISIONS.md → commit → stop.

## Prompt at the start of each new session

Follow the "At the start of every session" steps in CLAUDE.md and tell me where we are.

## Handy prompts

- "Show me the leakage report for <dataset> and explain each flag. What would you drop and why?"
- "Explain this metric table like I'm in an interview. Why PR-AUC and not just accuracy?"
- "Walk me through the threshold maths with one hand-calculated example."
- "Pick one customer and explain their churn prediction in plain English using SHAP."
- "Quiz me with 5 interview questions on this phase. Don't answer until I reply."
- "Does anything here look too good to be true? Check for leakage."
- "Run `git log -3 --format=full` — confirm there are no co-author lines."
