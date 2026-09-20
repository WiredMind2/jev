# Model card — hashing option-attention head

Frozen hashing encoder (MD5 token buckets + fixed random embeddings) plus a trained option-query head.

- **Use:** prove listwise CE + shuffled-context collapse on synthetic menus. CPU trainer.
- **Not:** a substitute for Qwen zero-shot vs frozen-Qwen head.
- **Train:** synthetic train JSONL, validation for early stopping only. Calibration unused.
- **This run:** train acc 1.00, shuffled train 0.24, frozen test acc 1.00, shuffled test 0.42, T=0.25. Checkpoint path `runs/synthetic-hashing-head.pt` (gitignored `*.pt`).
- **Evidence:** `reports/v0/metrics/train-hashing.json`, `eval-hashing-test.json`.
