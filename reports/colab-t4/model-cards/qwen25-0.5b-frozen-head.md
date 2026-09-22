# Model card — Qwen2.5-0.5B frozen option head (Colab T4)

**Status:** executed on Tesla T4 (14912 MiB, PyTorch 2.11.0+cu128).

Encoder frozen. Option-attention head trained with `jev train-head --encoder hf`
`--max-steps` 400 (CLINC150 800), `--save-every` 50, `--resume` across Colab
sessions. Temperature fit on the calibration split. Comparison is stratified
`--limit 300 --seed 0` of the frozen official test.

At this step budget the head stays at majority on BANKING77 / SST-5 / BoolQ
and below TF-IDF on CLINC150. That is the measured T4 result, not a 1650
number. Evidence: `reports/colab-t4/metrics/eval-*-hf-head.json` plus
`hardware.md`. This is not TypeSafe Jev and not RLCD.
