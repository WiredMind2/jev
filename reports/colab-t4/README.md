# Public-split GPU report

This directory is **not** `reports/v0` (GTX 1650 synthetic table).

Do not paste rows into [`../v0/metrics.md`](../v0/metrics.md).

## Status (2026-09-21)

Phase 1 code is on `main`: `train-head --resume`, `calibrate --backend`,
`evaluate --temperature-json`, risk-coverage curve, batched HF option
scoring, and notebook cells for BANKING77 → SST-5 → BoolQ → Wikispeedia →
CLINC150 plus a 3B VRAM probe.

**Remaining Qwen compute is Google Colab T4**, not the local laptop GPU.
Open [`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb)
(Runtime → T4 GPU), keep Drive `jev-runs/` as the persist path, `--resume`
if the VM dies, then file eval JSON here with
`python -m jev ingest-colab /path/to/jev-runs --dest reports/colab-t4`.
That command refuses to write [`../v0/metrics.md`](../v0/metrics.md).

CPU majority / TF-IDF on the official `--limit 300 --seed 0` slices are
filled. Frozen-head and most zero-shot Qwen rows stay empty until that T4
session. BoolQ / SST-5 zero-shot JSON already in `metrics/` came from an
interrupted RTX 4050 smoke and are labeled as such.

This is independent research. It is not TypeSafe Jev and not RLCD.
