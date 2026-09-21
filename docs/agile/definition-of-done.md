# Definition of done

A story is not Done until every box is checked.

- [ ] Acceptance criteria for the story are checked off
- [ ] Behavior verified (pytest for CLI/metrics; Colab notebook string tests; T4 eval JSON for US-006)
- [ ] No leftover debug / TODOs in the changed files
- [ ] `current-sprint.md` board updated
- [ ] User-facing copy and empty/error states handled if the story touches UI
- [ ] CPU tests `pytest -m "not cuda and not hf"` pass
- [ ] No claim that Colab T4 numbers equal the GTX 1650 `reports/v0` table
- [ ] No TypeSafe affiliation, RLCD clone, or production-serving claim
