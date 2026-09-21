# AGENTS

Independent research package reconstructing the public System One *interface* (Choice / Score / Noul). Not TypeSafe Jev and not RLCD.

Python ≥3.11, `pip install -e ".[dev]"`, CLI `jev`. CPU tests: `pytest -m "not cuda and not hf"`. Hosted GPU training is Colab T4 (`docs/11-colab.md`); do not run `jev serve` there. Do not overwrite `reports/v0` 1650 numbers with T4 rows.

## Sprint deliverables

This project keeps a Scrum pack in `docs/agile/`. Keep it current whenever work changes product behavior.

- `docs/agile/product-backlog.md` — prioritized work not in the sprint
- `docs/agile/current-sprint.md` — goal, board (To Do / In Progress / Done), committed stories
- `docs/agile/definition-of-done.md` — a story is not Done until every box is checked
- `docs/agile/retro.md` — append only at sprint end

When implementing a story: move it to In Progress, check off AC, satisfy DoD, then move it to Done before ending the turn.
When adding a feature: add or refine backlog stories (stable ids like `US-012`), pull Ready ones into the current sprint if they fit the goal, otherwise start the next sprint.
Skip typo, config, and questions-only work. Do not recreate the pack from scratch.

Format and quality bar: `.cursor/rules/agile-docs.mdc` (bootstrapped from the user-level agile-sprint rule).
