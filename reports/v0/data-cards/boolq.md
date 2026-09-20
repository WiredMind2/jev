# Data card — BoolQ

- **Task:** Noul (yes/no reading comprehension).
- **Source:** Reddy, Chen, Manning 2019. HF `google/boolq`.
- **License:** CC BY-SA 3.0.
- **Criteria:** `assets/criteria/boolq.json`.
- **Public test is unlabeled.** Frozen test = official validation.
- **`group_id`:** Wikipedia title when present. The `google/boolq` mirror **omits title**; conversion then hashes the passage (`passage:{sha256[:16]}`) so the same article text does not cross train and test.
- **This run (HF):** train 6410 / val 790 / calib 789 / test 3270 groups 2937. Manifest `reports/v0/manifests/boolq-hf.json`. Fixture rows still use explicit titles (`tests/fixtures/data/boolq.json`).
