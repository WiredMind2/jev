"""Recorded local hardware and the pinned research models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HARDWARE_PATH = REPO_ROOT / "configs" / "hardware.yaml"


@dataclass(frozen=True)
class HardwarePin:
    gpu_name: str
    vram_mib: int
    compute_capability: str
    driver: str
    cuda_driver: str
    bf16: bool
    fp16: bool
    zero_shot_model: str
    frozen_head_model: str
    dtype: str
    reduction_reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# GTX 1650 is Turing sm_75 with 4 GiB. Qwen2.5-3B FP16 does not fit.
DEFAULT_PIN = HardwarePin(
    gpu_name="NVIDIA GeForce GTX 1650",
    vram_mib=4096,
    compute_capability="7.5",
    driver="570.211.01",
    cuda_driver="12.8",
    bf16=False,
    fp16=True,
    zero_shot_model="Qwen/Qwen2.5-0.5B",
    frozen_head_model="Qwen/Qwen2.5-0.5B",
    dtype="float16",
    reduction_reason=(
        "Pinned Qwen2.5-3B zero-shot would need ~6 GiB FP16 weights plus KV; "
        "this machine has 4096 MiB on sm_75 (no BF16). Reduced zero-shot and "
        "frozen-head to Qwen2.5-0.5B."
    ),
)


def release_cuda() -> None:
    """Drop unused CUDA allocations between serialized GPU stages."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_hardware_pin(path: Path | None = None) -> HardwarePin:
    target = path or DEFAULT_HARDWARE_PATH
    if not target.exists():
        return DEFAULT_PIN
    raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    gpu = raw.get("gpu", {})
    models = raw.get("models", {})
    return HardwarePin(
        gpu_name=str(gpu.get("name", DEFAULT_PIN.gpu_name)),
        vram_mib=int(gpu.get("vram_mib", DEFAULT_PIN.vram_mib)),
        compute_capability=str(gpu.get("compute_capability", DEFAULT_PIN.compute_capability)),
        driver=str(gpu.get("driver", DEFAULT_PIN.driver)),
        cuda_driver=str(gpu.get("cuda_driver", DEFAULT_PIN.cuda_driver)),
        bf16=bool(gpu.get("bf16", False)),
        fp16=bool(gpu.get("fp16", True)),
        zero_shot_model=str(models.get("zero_shot", DEFAULT_PIN.zero_shot_model)),
        frozen_head_model=str(models.get("frozen_head", DEFAULT_PIN.frozen_head_model)),
        dtype=str(models.get("dtype", DEFAULT_PIN.dtype)),
        reduction_reason=str(raw.get("reduction_reason", DEFAULT_PIN.reduction_reason)),
    )
