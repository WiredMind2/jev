# Data card — BANKING77

- **Task:** Choice over 77 banking intents. Full menu on every row.
- **Source:** Casanueva et al., EMNLP 2020. HF `mteb/banking77` this run (`PolyAI/banking77` failed unauthenticated).
- **License:** CC BY 4.0.
- **Criteria:** `assets/criteria/banking77.json` (77 keys, banking77-v0). Frozen descriptions.
- **`group_id`:** row id / utterance id. **Never the intent label.**
- **Split rule:** official test frozen; official train carved 80/10/10 stratified by intent into train/validation/calibration.
- **This run (HF `mteb/banking77`):** train 8067 / val 965 / calib 961 / test 3076. Manifest `reports/v0/manifests/banking77-hf.json`.
- **Fixture:** 40 train + 8 test rows in `tests/fixtures/data/banking77.json` for converters without network.
- **Pitfall:** 77-way softmax over names without descriptions is a lexical prior test. Criteria stay attached.
