# Hosted GPUs (Google Colab)

Free notebook GPUs (Google Colab, and similar Kaggle kernels) are an
optional **training and measurement** lane. They are not a substitute
for the v0 local pin in [`configs/hardware.yaml`](../configs/hardware.yaml)
and they are not a place to run `jev serve`.

The v0 report was measured on a 4 GiB GTX 1650, which is why both
zero-shot scoring and the frozen head use `Qwen/Qwen2.5-0.5B` fp16.
A typical Colab free runtime is a Tesla T4 (~16 GiB VRAM). That is
enough to keep the 0.5B path and to try the originally planned
**Qwen 2.5 3B** frozen-encoder class. GPU type, idle timeout, and
session length are **not guaranteed**; Colab does not publish a fixed
free GPU-hour quota.

Use this file when a later run needs more VRAM or wall-clock than the
1650. Keep CPU tests, hashing-trainer proofs, and published v0 numbers
on the machines they were pinned to.

## Where it belongs in the pipeline

Map onto [Training](04-training.md) and the
[implementation plan](06-implementation.md):

| Step | Colab? | Why |
|---|---|---|
| Schema / converters / hashing-head unit tests | No | CPU. CI already runs `pytest -m "not cuda and not hf"`. |
| Tiny CUDA gates (`pytest -m cuda`) | Optional | Useful if you have no local NVIDIA GPU. |
| v0 0.5B synthetic frozen-head (64 rows) | No | Already measured on the 1650. Do not replace those cells. |
| Frozen-head train on BANKING77, SST-5, BoolQ | **Yes** | Larger JSONL, longer state, bigger batches than 4 GiB allows. |
| Wikispeedia next-click (variable `N`) | **Yes** | Main hosted-GPU job in week 3. |
| 3B frozen encoder (plan starting class) | **Yes** | 3B fp16 weights do not fit the 1650; they do fit a T4. |
| 7B/8B fp16 | No (free T4) | Weights alone are ~14–16 GiB. Needs 8-bit/4-bit or a paid GPU. |
| Temperature / isotonic fit | After GPU logits | Can finish on CPU if you write logits first. |
| `jev serve` / public HTTP | **No** | Ephemeral VM, idle disconnect, Colab is for interactive notebooks. |
| Mixing Colab rows into `reports/v0/metrics.md` | **No** | Different pin. New run → new hardware note + model card. |

Call the existing CLI. Do not reimplement `train_option_head` as ad-hoc
notebook cells; cells should only provision the runtime and invoke
`jev`.

## Constraints to plan around

**Fact** (Colab FAQ, retrieved 20 September 2026): free notebooks can
run for at most 12 hours; idle timeouts, max VM lifetime, and GPU
models fluctuate; accelerators are not guaranteed.
https://research.google.com/colaboratory/faq.html

Practical consequences for this repo:

- Disk under `/content` disappears when the runtime dies. Put
  checkpoints, converted JSONL, and eval JSON on Google Drive (or
  Hugging Face) **before** the job finishes. The trainer currently
  writes the checkpoint at the end of `jev train-head`, so keep the
  tab active and copy `runs/` as soon as the command returns. Use
  `--max-steps` for smoke tests if you expect a disconnect. After the
  session, file Drive `jev-runs` with `python -m jev ingest-colab`
  into `reports/colab-t4/` — never into `reports/v0/`. `train-head`,
  `calibrate`, and `evaluate` print a stderr progress bar with ETA
  (`JEV_PROGRESS=0` disables it). After a disconnect, Restart runtime
  and Run all: the notebook skips a dataset when both
  `eval-{name}-hf-head.json` and `eval-{name}-hf-logprob.json` already
  exist on Drive (`JEV_FORCE_JOB=1` reruns). Individual stages
  (convert, train-head, calibrate, evaluate) also skip when their
  output file is already on Drive, so a restart does not reload Qwen
  just to finish zero-shot eval. Each dataset prints a task bar with
  ETA (`banking77 tasks  3/8 ...`).
- Colab already ships a CUDA PyTorch. A naive
  `pip install -e ".[dev]"` from PyPI can replace it with a CPU wheel.
  Install the package, then confirm `torch.cuda.is_available()` before
  training.
- `jev hardware` prints the **repo pin** (GTX 1650 / 0.5B), not the
  live Colab GPU. Record `nvidia-smi` and
  `torch.cuda.get_device_properties(0)` into a run-specific note.
- System RAM on free Colab is modest (~12 GiB). Prefer fp16, freeze
  the encoder, and keep `batch_size` small until a VRAM re-measure
  succeeds.
- Hugging Face downloads repeat every session unless you cache
  `HF_HOME` on Drive.
- Optional Hugging Face auth: store `HF_TOKEN` in Colab Secrets (the
  key icon). Read it via `google.colab.userdata` into `HF_TOKEN` /
  `HUGGING_FACE_HUB_TOKEN`. Never paste a token into a cell or commit
  one. Qwen2.5-0.5B is typically ungated; BANKING77 may still fail
  unauthenticated on `PolyAI/banking77` (the converter already tries
  `mteb/banking77`).
- Do not use Colab as a file host, a 24/7 scorer, or a way around
  quota (multiple accounts). That conflicts with Colab's terms.

Kaggle notebooks are the same class of tool (free T4/P100, weekly
hour cap, ephemeral disk): useful for the same training steps, still
not a serving runtime.

## Runtime recipe

Committed notebook (same recipe, with a synthetic hashing smoke and
optional 3B VRAM probe):
[`notebooks/jev_colab_hosted_gpu.ipynb`](../notebooks/jev_colab_hosted_gpu.ipynb).
Open in Colab:
https://colab.research.google.com/github/WiredMind2/jev/blob/main/notebooks/jev_colab_hosted_gpu.ipynb
Pin a commit/branch when reproducing a named report. How this fits the
v0 gate: [`reports/v0/colab.md`](../reports/v0/colab.md).

Runtime → Change runtime type → T4 GPU (or whatever Colab offers).
Python must be ≥3.11 (`python --version`).

```python
# Cell 1 — GPU check. Stop if this is false.
import torch
print(torch.__version__, torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(p.name, p.total_memory // (1024**2), "MiB")
```

```python
# Cell 2 — clone and install without clobbering Colab's CUDA torch.
import os, subprocess, sys
from pathlib import Path
import torch

REPO = "https://github.com/WiredMind2/jev.git"
if Path("pyproject.toml").exists() and Path("src/jev").exists():
    pass  # already in the clone
elif (Path("jev") / "pyproject.toml").exists():
    os.chdir("jev")
else:
    # Pin a commit/branch when reproducing a report.
    subprocess.check_call(["git", "clone", "--depth", "1", REPO, "jev"])
    os.chdir("jev")
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-e", ".[dev]", "--no-deps"]
)
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "pydantic>=2.7",
        "jsonschema>=4.22",
        "fastapi>=0.111",
        "uvicorn[standard]>=0.30",
        "typer>=0.12",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "transformers>=4.44",
        "datasets>=2.20",
        "safetensors>=0.4",
        "httpx>=0.27",
        "pyyaml>=6.0",
        "pytest>=8.2",
    ]
)
assert torch.cuda.is_available(), "CUDA torch missing after pip; do not train"
```

```python
# Cell 3 — persist artifacts. Same layout as data/README.md.
import os
from pathlib import Path
from google.colab import drive

drive.mount("/content/drive")
ROOT = Path("/content/drive/MyDrive/jev-runs")
for name in ("data", "runs", "reports"):
    (ROOT / name).mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(ROOT / "hf-cache")
```

```text
# Cell 4 — convert, train, calibrate, evaluate. Paths on Drive.
# Use python -m jev so the console script PATH does not matter.
!python -m jev data-convert synthetic --out /content/drive/MyDrive/jev-runs/data --n 256

# Hashing-head CLI smoke (CPU-capable). Not a replacement for the v0
# 0.5B synthetic frozen-head cells already in reports/v0/.
!python -m jev train-head \
  /content/drive/MyDrive/jev-runs/data/synthetic/jsonl/train.jsonl \
  --val-jsonl /content/drive/MyDrive/jev-runs/data/synthetic/jsonl/validation.jsonl \
  --encoder hashing \
  --out /content/drive/MyDrive/jev-runs/runs/synthetic-hashing-head.pt \
  --epochs 12
!python -m jev calibrate \
  /content/drive/MyDrive/jev-runs/data/synthetic/jsonl/calibration.jsonl \
  --checkpoint /content/drive/MyDrive/jev-runs/runs/synthetic-hashing-head.pt \
  --out /content/drive/MyDrive/jev-runs/runs/synthetic-hashing-temperature.json
!python -m jev evaluate \
  /content/drive/MyDrive/jev-runs/data/synthetic/jsonl/test.jsonl \
  --backend option-head \
  --checkpoint /content/drive/MyDrive/jev-runs/runs/synthetic-hashing-head.pt \
  --out /content/drive/MyDrive/jev-runs/reports/eval-synthetic-hashing.json

!python -m jev data-convert banking77 --out /content/drive/MyDrive/jev-runs/data

!python -m jev train-head \
  /content/drive/MyDrive/jev-runs/data/banking77/jsonl/train.jsonl \
  --val-jsonl /content/drive/MyDrive/jev-runs/data/banking77/jsonl/validation.jsonl \
  --encoder hf \
  --model-id Qwen/Qwen2.5-0.5B \
  --out /content/drive/MyDrive/jev-runs/runs/banking77-hf-head.pt \
  --epochs 12 \
  --batch-size 4 \
  --save-every 50 \
  --resume /content/drive/MyDrive/jev-runs/runs/banking77-hf-head.pt

!python -m jev calibrate \
  /content/drive/MyDrive/jev-runs/data/banking77/jsonl/calibration.jsonl \
  --checkpoint /content/drive/MyDrive/jev-runs/runs/banking77-hf-head.pt \
  --out /content/drive/MyDrive/jev-runs/runs/banking77-temperature.json

!python -m jev evaluate \
  /content/drive/MyDrive/jev-runs/data/banking77/jsonl/test.jsonl \
  --backend option-head \
  --checkpoint /content/drive/MyDrive/jev-runs/runs/banking77-hf-head.pt \
  --temperature-json /content/drive/MyDrive/jev-runs/runs/banking77-temperature.json \
  --limit 300 \
  --seed 0 \
  --out /content/drive/MyDrive/jev-runs/reports/eval-banking77-hf-head.json

# If the VM died, continue the same --out with --resume.
# The notebook then runs SST-5, BoolQ, Wikispeedia, CLINC150 the same way
# (run_frozen_split_job), plus hf-logprob / majority / tfidf-linear on the
# same --limit 300 seed=0 slice. Do not paste those rows into reports/v0/.
```

Never pass the calibration split to `--val-jsonl`. Validation is early
stopping only ([Training](04-training.md) stage 3).

If `data-convert banking77` fails on `PolyAI/banking77` (unauthenticated
HF), that is the same issue the v0 report hit; the converter already
tries `mteb/banking77`. Do not vendor the raw corpus.

Smoke-test the GPU path first:

```text
!python -m pytest -m "cuda and not hf" -q
```

Full HF CUDA gates download Qwen and need VRAM; run them only after
the Drive cache is set.

## Stepping up from 0.5B

The 1650 pin exists because 3B fp16 weights exceed 4 GiB. On a T4:

1. Keep the encoder frozen. Train only the option-attention head.
2. Start with `Qwen/Qwen2.5-0.5B` and the same JSONL you will use for
   3B, so the only changed variable is model size.
3. Then `--model-id Qwen/Qwen2.5-3B` with `batch_size` 1 and a short
   `--max-steps` VRAM probe. If that completes, train for real.
4. Do not jump to 7B/8B fp16 on free Colab.
5. Write a **new** hardware note (GPU name, VRAM, driver, PyTorch,
   dtype, `max_state_tokens`, batch size). Do not edit the v0 1650
   pin to match the Colab box.

Release CUDA between serialized stages the same way the local report
did (`jev.hardware.release_cuda`): train the head, unload, then
zero-shot eval if both run in one session.

## What to check into git

- Converted datasets stay out of git ([Datasets](10-datasets.md)).
- Checkpoints (`.pt`) stay out of git unless a model card explicitly
  vendors a tiny proof artifact.
- Eval JSON, a hardware note, and a model card *do* belong in
  `reports/` for a named run, with the Colab GPU recorded.
- This recipe. Not a claim that Colab numbers equal hosted Jev or the
  v0 1650 table.

## Related files

- [`configs/hardware.yaml`](../configs/hardware.yaml) — v0 local pin
- [`reports/v0/hardware.md`](../reports/v0/hardware.md) — measured box
- [Training](04-training.md) — loss stages (unchanged on Colab)
- [Evaluation](05-evaluation.md) — metrics; label the GPU pin per row
- [Implementation](06-implementation.md) — week 3 is the Colab week
- [`notebooks/jev_colab_hosted_gpu.ipynb`](../notebooks/jev_colab_hosted_gpu.ipynb) — runnable cells
- [`reports/v0/colab.md`](../reports/v0/colab.md) — v0 gate and what was (not) executed
