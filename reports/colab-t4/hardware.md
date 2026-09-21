# Colab T4 hardware (pending live session)

This file is a placeholder. `jev hardware` still prints the **repo pin**
(GTX 1650 / Qwen2.5-0.5B). After a live T4 run, replace this table with
`nvidia-smi` and `torch.cuda.get_device_properties(0)` from Drive
`jev-runs/reports/hardware.md`.

| Field | Value |
|---|---|
| GPU | not executed |
| VRAM | |
| PyTorch | |
| dtype | float16 (planned) |
| batch / max_state_tokens | 4 / 256 (BANKING77 starting point) |
| notebook | notebooks/jev_colab_hosted_gpu.ipynb |

Do not edit `configs/hardware.yaml` or `reports/v0/hardware.md` to match Colab.
