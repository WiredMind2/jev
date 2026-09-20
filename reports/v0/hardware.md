# v0 hardware pin

Machine used for this report (20 September 2026):

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce GTX 1650 |
| VRAM | 4096 MiB (driver reports ~3896 MiB usable) |
| Compute | sm_75 Turing |
| Driver | 570.211.01 |
| CUDA driver | 12.8 |
| PyTorch | 2.6.0+cu124 |
| Python | 3.12 (.venv). System python 3.9 is not used. |
| fp16 | yes |
| bf16 | no |

Pinned models (`configs/hardware.yaml`, `configs/models.yaml`):

- Zero-shot: `Qwen/Qwen2.5-0.5B` fp16
- Frozen encoder + option head: `Qwen/Qwen2.5-0.5B` fp16

Reduction reason: the plan's consumer-GPU starting class included Qwen 2.5 3B–8B. 3B fp16 weights alone exceed 4 GiB. Both roles therefore use 0.5B. Do not silently swap a larger checkpoint without a VRAM re-measure.

CUDA stages were serialized (head train, then unload, then zero-shot) via `jev.hardware.release_cuda`.
