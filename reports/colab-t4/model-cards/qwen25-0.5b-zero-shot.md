# Model card — Qwen2.5-0.5B zero-shot logprob (Colab T4)

**Status:** not executed. Fill after `jev evaluate --backend hf-logprob --limit 300 --seed 0`.

Cached prefix continuation scoring. Temperature fit on the calibration split
(`jev calibrate --backend hf-logprob`). This is not TypeSafe Jev and not RLCD.
