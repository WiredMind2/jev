# Data card — CLINC150

- **Task:** Choice over 150 in-scope intents plus `out_of_scope`.
- **Source:** Larson et al., EMNLP 2019. HF `clinc_oos` config `plus`.
- **License:** CC BY 3.0.
- **Criteria:** `assets/criteria/clinc150.json` (151 keys, official intent names + OOS).
- **Split rule:** official train and test; official validation split even/odd into validation vs calibration (calibration unused for early stopping). OOS test kept.
- **`group_id`:** example id, not intent (so OOS and in-scope are not grouped by label).
- **This run (HF):** train 15250 / val 1550 / calib 1550 / test 5500. Manifest `reports/v0/manifests/clinc150-hf.json`.
- **Fixture:** `tests/fixtures/data/clinc150.json` includes genuine `out_of_scope` gold rows and a 151-option menu. Converts to train 52 / val 13 / calib 13 / test 26. Manifest `reports/v0/manifests/clinc150-fixture.json`.
