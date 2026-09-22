# Public-split GPU report

This directory is **not** `reports/v0` (GTX 1650 synthetic table).

Do not paste rows into [`../v0/metrics.md`](../v0/metrics.md).

## Status (2026-09-22)

Live Colab T4 session filed. Frozen-head vs zero-shot JSON exists for
BANKING77, SST-5, BoolQ, Wikispeedia, and CLINC150 on the official
`--limit 300 --seed 0` slices. Hardware: Tesla T4, 14912 MiB,
PyTorch 2.11.0+cu128 (`colab-live-hardware.json`). Qwen2.5-3B
`--max-steps` 2 batch 1 **fit**.

`python -m jev ingest-colab` copied Drive `jev-runs` here and left
[`../v0/metrics.md`](../v0/metrics.md) unchanged.

This is independent research. It is not TypeSafe Jev and not RLCD.
