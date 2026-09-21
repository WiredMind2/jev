# Colab T4 hardware (pending live session)

This file is the **T4 pin placeholder**. `jev hardware` still prints the
**repo pin** (GTX 1650 / Qwen2.5-0.5B). After a live T4 run, replace this
table with `nvidia-smi` and `torch.cuda.get_device_properties(0)` from Drive
`jev-runs/reports/hardware.md`.

Do **not** use the local laptop GPU for remaining Qwen jobs. BANKING77,
Wikispeedia, CLINC150, frozen-head training, and the 3B probe run on
Colab T4 via [`notebooks/jev_colab_hosted_gpu.ipynb`](../../notebooks/jev_colab_hosted_gpu.ipynb).

| Field | Value |
|---|---|
| GPU | not executed (Colab T4 required) |
| VRAM | |
| PyTorch | |
| dtype | float16 (planned) |
| batch / max_state_tokens | 4 / 256 (BANKING77 starting point) |
| notebook | notebooks/jev_colab_hosted_gpu.ipynb |

A short RTX 4050 zero-shot smoke (BoolQ, SST-5) exists in `metrics/` and is
labeled as that GPU. It is not a T4 row and not a reason to keep scoring on
the laptop.

Do not edit `configs/hardware.yaml` or `reports/v0/hardware.md` to match Colab.
