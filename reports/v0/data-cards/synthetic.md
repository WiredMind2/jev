# Data card — synthetic menus

- **Task:** Choice. Each state names exactly one gold color bucket.
- **License:** generated in-repo.
- **Criteria:** `assets/criteria/synthetic.json` (synthetic-v0), hashed into `criteria_version`.
- **Split rule:** `group_id = synthetic:{i//4}`; hash groups into train/validation/calibration/test. Calibration unused for early stopping.
- **This run:** n train 172 / val 24 / calib 48 / test 12. Manifest `reports/v0/manifests/synthetic.json`.
- **Purpose:** prove the option-attention trainer. ~98% expected. Shuffled-context should collapse.
