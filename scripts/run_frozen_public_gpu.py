"""Orchestrate python -m jev public-split GPU jobs. Not a second trainer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DATA = ROOT / "data"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports" / "colab-t4"
SLICE = {
    "banking77": 300,
    "sst5": 300,
    "boolq": 300,
    "wikispeedia": 300,
    "clinc150": 300,
}
BATCH = {
    "banking77": "2",
    "sst5": "4",
    "boolq": "4",
    "wikispeedia": "1",
    "clinc150": "2",
}
MAX_STEPS = {
    "banking77": "400",
    "sst5": "400",
    "boolq": "400",
    "wikispeedia": "200",
    "clinc150": "400",
}
ORDER = ["boolq", "sst5", "banking77", "clinc150", "wikispeedia"]
MODEL = "Qwen/Qwen2.5-0.5B"


def refuse_local_gpu() -> None:
    """Hosted Qwen jobs belong on Colab T4, not the laptop GPU."""
    in_colab = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
    if in_colab or os.environ.get("JEV_ALLOW_LOCAL_GPU") == "1":
        return
    raise SystemExit(
        "Refusing hosted-GPU jobs on this machine. "
        "Open notebooks/jev_colab_hosted_gpu.ipynb with a Colab T4, "
        "or set JEV_ALLOW_LOCAL_GPU=1."
    )


def run_jev(*args: str) -> None:
    cmd = [PY, "-m", "jev", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def write_hardware() -> None:
    import torch

    REPORTS.mkdir(parents=True, exist_ok=True)
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    lines = [
        "# Live GPU (this agent)",
        "",
        "This is **not** the GTX 1650 `reports/v0` pin and **not** a signed-in Colab T4 session.",
        "`jev hardware` still prints the 1650 repo pin; that is expected.",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| nvidia-smi | {smi} |",
        f"| torch.cuda.is_available | {torch.cuda.is_available()} |",
        f"| PyTorch | {torch.__version__} |",
        f"| device | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'} |",
        f"| total_memory_bytes | {getattr(props, 'total_memory', None)} |",
        "| dtype | float16 |",
        f"| max_state_tokens | 256 |",
        f"| notebook | notebooks/jev_colab_hosted_gpu.ipynb |",
        "",
        "Do not edit `configs/hardware.yaml` or `reports/v0/hardware.md` to match this box.",
        "",
    ]
    (REPORTS / "hardware.md").write_text("\n".join(lines), encoding="utf-8")


def zeroshot(name: str) -> None:
    jsonl = DATA / name / "jsonl"
    limit = str(SLICE[name])
    t_zs = RUNS / f"{name}-zeroshot-temperature.json"
    run_jev(
        "calibrate",
        str(jsonl / "calibration.jsonl"),
        "--backend",
        "hf-logprob",
        "--limit",
        limit,
        "--seed",
        "0",
        "--out",
        str(t_zs),
    )
    run_jev(
        "evaluate",
        str(jsonl / "test.jsonl"),
        "--backend",
        "hf-logprob",
        "--temperature-json",
        str(t_zs),
        "--limit",
        limit,
        "--seed",
        "0",
        "--out",
        str(REPORTS / "metrics" / f"eval-{name}-hf-logprob.json"),
    )


def train_and_eval_head(name: str) -> None:
    jsonl = DATA / name / "jsonl"
    limit = str(SLICE[name])
    ckpt = RUNS / f"{name}-hf-head.pt"
    t_head = RUNS / f"{name}-head-temperature.json"
    train_args = [
        "train-head",
        str(jsonl / "train.jsonl"),
        "--val-jsonl",
        str(jsonl / "validation.jsonl"),
        "--encoder",
        "hf",
        "--model-id",
        MODEL,
        "--out",
        str(ckpt),
        "--epochs",
        "12",
        "--batch-size",
        BATCH[name],
        "--save-every",
        "50",
        "--max-steps",
        MAX_STEPS[name],
    ]
    if ckpt.exists():
        train_args.extend(["--resume", str(ckpt)])
    run_jev(*train_args)
    run_jev(
        "calibrate",
        str(jsonl / "calibration.jsonl"),
        "--backend",
        "option-head",
        "--checkpoint",
        str(ckpt),
        "--limit",
        limit,
        "--seed",
        "0",
        "--out",
        str(t_head),
    )
    run_jev(
        "evaluate",
        str(jsonl / "test.jsonl"),
        "--backend",
        "option-head",
        "--checkpoint",
        str(ckpt),
        "--temperature-json",
        str(t_head),
        "--limit",
        limit,
        "--seed",
        "0",
        "--out",
        str(REPORTS / "metrics" / f"eval-{name}-hf-head.json"),
    )


def probe_3b() -> None:
    jsonl = DATA / "banking77" / "jsonl"
    ckpt = RUNS / "banking77-3b-probe.pt"
    note = REPORTS / "model-cards" / "qwen25-3b-probe.md"
    try:
        run_jev(
            "train-head",
            str(jsonl / "train.jsonl"),
            "--encoder",
            "hf",
            "--model-id",
            "Qwen/Qwen2.5-3B",
            "--out",
            str(ckpt),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--max-steps",
            "2",
            "--save-every",
            "1",
        )
        outcome = "fit (2 steps completed)"
    except subprocess.CalledProcessError as exc:
        outcome = f"failed exit={exc.returncode} (likely OOM or load error)"
    note.write_text(
        "\n".join(
            [
                "# Model card — Qwen2.5-3B VRAM probe",
                "",
                f"**Outcome:** {outcome}",
                "",
                "Command: `jev train-head --encoder hf --model-id Qwen/Qwen2.5-3B --batch-size 1 --max-steps 2`.",
                "Encoder stays frozen. This is not TypeSafe Jev and not RLCD.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    refuse_local_gpu()
    os.chdir(ROOT)
    (RUNS).mkdir(parents=True, exist_ok=True)
    (REPORTS / "metrics").mkdir(parents=True, exist_ok=True)
    write_hardware()
    names = sys.argv[1:] or ORDER
    stage = os.environ.get("JEV_GPU_STAGE", "all")
    if stage in {"all", "zeroshot"}:
        for name in names:
            zeroshot(name)
    if stage in {"all", "head"}:
        for name in names:
            train_and_eval_head(name)
    if stage in {"all", "probe3b"} and os.environ.get("RUN_3B_PROBE", "1") == "1":
        probe_3b()
    print("done", json.dumps({"names": names, "stage": stage}))


if __name__ == "__main__":
    main()
