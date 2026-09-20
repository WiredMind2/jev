# Colab / hosted-GPU path (research v0)

This is the evidence file for Google Colab as a **required compute
lane**, not a replacement for the 4 GiB GTX 1650 pin.

Authoritative recipe: [`docs/11-colab.md`](../../docs/11-colab.md).
Runnable notebook: [`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb).
CPU CLI proof: [`scripts/colab_cpu_smoke.sh`](../../scripts/colab_cpu_smoke.sh).

This is independent research. It is **not** TypeSafe Jev and **not** RLCD.

## Why Colab is a v0 gate

`reports/v0/metrics.md` was measured on a GTX 1650 (4 GiB). Both
zero-shot and the frozen head therefore use `Qwen/Qwen2.5-0.5B` fp16.
Qwen2.5-3B fp16 does **not** fit that box.

[`docs/11-colab.md`](../../docs/11-colab.md) and
[`docs/04-training.md`](../../docs/04-training.md) assign later jobs to
a free Colab T4 (~16 GiB):

| Job | Where |
|---|---|
| Schema / converters / hashing-head unit tests | CPU / CI |
| v0 0.5B synthetic frozen-head (64 rows) | 1650. Do not replace those cells. |
| Frozen-head BANKING77, SST-5, BoolQ | **Colab T4** |
| Wikispeedia next-click (variable `N`) | **Colab T4** (week 3 main hosted-GPU job) |
| Frozen 3B encoder | **Colab T4**. Does not fit 4 GiB. |
| `jev serve` | Local process. Not Colab. |

v0 cannot be declared done without this path existing (docs + notebook
+ CLI sequence). v0 also cannot pretend a T4 run happened if it did
not. Do not mix a 1650 row and a T4 row in `metrics.md`.

## How to reproduce on Colab

1. Runtime → Change runtime type → T4 GPU. Python ≥3.11.
2. Open [`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb)
   ([Colab](https://colab.research.google.com/github/WiredMind2/jev/blob/main/notebooks/jev_colab_hosted_gpu.ipynb)).
   Pin a commit when reproducing a named report.
3. Optional: Colab Secrets → `HF_TOKEN`. Never paste a token into a cell.
4. Run all cells. The notebook mounts Drive, sets `HF_HOME` there,
   records the **live** GPU (`nvidia-smi` / `torch.cuda.get_device_properties`),
   and invokes `python -m jev`. `jev hardware` still prints the 1650 pin;
   that is expected.
5. Copy `jev-runs/reports/` and `jev-runs/runs/` off the VM before the
   session dies. File a **new** hardware note beside this 1650 pin.
   Do not edit [`hardware.md`](hardware.md) or `configs/hardware.yaml`
   to match the Colab box.

Install rule (from the recipe): Colab already ships CUDA PyTorch. Use
`pip install -e ".[dev]" --no-deps`, then install the other deps
**without replacing torch**. Confirm `torch.cuda.is_available()` before
any HF train.

## Success criteria (Colab lane)

All of the following must be true before research v0 can stop:

1. [`docs/11-colab.md`](../../docs/11-colab.md) is in git and linked from
   README, training, evaluation, implementation, datasets, contributing,
   and this report.
2. [`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb)
   installs the package, converts / trains / calibrates / evaluates at
   least the synthetic path via `python -m jev`, and can run the
   BANKING77 hosted-GPU job from the same cells.
3. Synthetic hashing CPU smoke passes (`scripts/colab_cpu_smoke.sh` or
   the notebook's smoke cells). Report JSON uses the same metric keys as
   [`metrics.md`](metrics.md) (`accuracy`, `nll`, `brier`, `ece`,
   `shuffled_accuracy`, …).
4. No hardcoded Hugging Face tokens. Optional auth is Colab Secrets
   `HF_TOKEN` only.
5. A **live Colab T4 run** for BANKING77 (then Wikispeedia, then a 3B
   VRAM probe) is executed, with eval JSON + a new hardware note + a
   model card under `reports/` for that named run. Until that happens,
   those rows stay empty. This file is not that run.
6. The main package CLI (`data-convert`, `train-head`, `calibrate`,
   `evaluate`) stays the thing the notebook calls. Notebook cells must
   not become a second trainer.

## What this machine executed

| Item | Status |
|---|---|
| Docs + notebook + CPU smoke script authored | Yes |
| Local CPU synthetic hashing CLI sequence | See `colab-cpu-smoke.json` if present after smoke |
| Live Colab T4 GPU session from this agent | **Not executed** (no Colab runtime on this host) |
| BANKING77 / Wikispeedia / 3B hosted-GPU numbers | **Not executed** |

A CPU smoke proves the CLI path exists. It is not evidence about T4
VRAM, 3B fit, or BANKING77 accuracy.

## Report layout for a future T4 run

Write under Drive `MyDrive/jev-runs/` (or a named `reports/<run-id>/`):

```text
reports/<run-id>/
  hardware.md          # live GPU name, VRAM, driver, PyTorch, dtype, batch
  metrics.md           # labeled Colab T4; do not merge into v0/metrics.md
  eval-*.json          # same keys as reports/v0/metrics/eval-*.json
  model-cards/
```

Never pass the calibration split to `--val-jsonl`.
