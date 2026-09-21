# Colab T4 report (pending live run)

This directory is the **T4 pin**. It is not `reports/v0` (GTX 1650).

Do not paste rows into [`../v0/metrics.md`](../v0/metrics.md).

## Status (2026-09-21)

Phase 1 code is on `main`: `train-head --resume`, `calibrate --backend`,
`evaluate --temperature-json`, risk-coverage curve, and notebook cells for
BANKING77 → SST-5 → BoolQ → Wikispeedia → CLINC150 plus a 3B VRAM probe.

A **live Colab T4 session has not been executed from this agent**. Fill this
tree after you run
[`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb)
with Runtime → T4 GPU, then copy Drive `jev-runs/reports/` here.

Expected after a real run:

- `hardware.md` — live `nvidia-smi` / `torch.cuda.get_device_properties`
- `metrics.md` — one table per dataset, GPU labeled Colab T4
- `metrics/eval-{dataset}-{hf-head,hf-logprob,majority,tfidf}.json`
- `model-cards/` for 0.5B frozen-head, 0.5B zero-shot, 3B probe outcome

This is independent research. It is not TypeSafe Jev and not RLCD.
