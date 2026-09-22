# Model card — Qwen2.5-0.5B zero-shot logprob (Colab T4)

**Status:** executed on Tesla T4 (14912 MiB, PyTorch 2.11.0+cu128).

Cached prefix continuation scoring (`jev evaluate --backend hf-logprob`).
Temperature fit on the calibration split (`jev calibrate --backend hf-logprob`).
Same frozen official JSONL and `--limit 300 --seed 0` slice as the frozen head.

Zero-shot beats the 400-step frozen head on every public task in this report.
Do not mix these rows into `reports/v0`. This is not TypeSafe Jev and not RLCD.
