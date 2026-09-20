# Data card — SST-5

- **Task:** Score, K=5 ordered sentiment levels.
- **Source:** Stanford Sentiment Treebank, sentence-level HF `SetFit/sst5`.
- **License:** original SST / Penn Treebank terms (research; not MIT-like redistribution).
- **Criteria:** `assets/criteria/sst5.json` situational level texts, not "0,1,2,3,4".
- **Split rule:** official validation kept as validation; official test frozen; 10% of train carved as calibration (not used for early stopping). Sentence-level only.
- **This run (HF):** train 7687 / val 1101 / calib 857 / test 2210. Manifest `reports/v0/manifests/sst5-hf.json`.
- **Fixture:** `tests/fixtures/data/sst5.json` converts to train 30 / val 15 / calib 5 / test 15. Manifest `reports/v0/manifests/sst5-fixture.json`.
- **Levels are ordered; do not shuffle Score criteria.**
