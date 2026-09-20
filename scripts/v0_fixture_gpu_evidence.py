"""Qwen2.5-0.5B fixture evidence on GTX 1650. OOM is recorded, never faked."""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

from jev.calibration import collect_logit_gold, fit_temperature
from jev.canonical import canonical_dumps
from jev.evaluation import evaluate_scorer, report_as_dict
from jev.hardware import release_cuda
from jev.scoring.factory import build_scorer
from jev.scoring.option_head import (
    OptionHeadScorer,
    TrainConfig,
    load_jsonl_examples,
    save_checkpoint,
    train_option_head,
)

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "converted-fixtures"
REPORTS = REPO / "reports" / "v0" / "metrics"
RUNS = REPO / "runs"


def _smi() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def _gpu_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    torch.cuda.synchronize()
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_miB": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "reserved_miB": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        "max_allocated_miB": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "nvidia_smi": _smi(),
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    print("WROTE", path, "acc", payload.get("accuracy"), "n", payload.get("n"), flush=True)


def _eval_logprob(dataset: str, limit: int = 0) -> None:
    from jev.evaluation import stratified_sample

    test = load_jsonl_examples(OUT / dataset / "jsonl" / "test.jsonl")
    used = stratified_sample(test, limit, seed=0) if limit else test
    print(f"==== hf-logprob {dataset} n={len(used)}/{len(test)} ====", flush=True)
    print("gpu_before", _gpu_stats(), flush=True)
    try:
        scorer = build_scorer("hf-logprob")
        report = evaluate_scorer(scorer, used, temperature=1.0, shuffled=True)
        payload = {
            "backend": "hf-logprob",
            "dataset": dataset,
            "temperature": 1.0,
            "model_id": scorer.model_id,
            "limit": limit or None,
            "n_full_test": len(test),
            **report_as_dict(report),
            "gpu_after": _gpu_stats(),
        }
        _write(REPORTS / f"eval-hf-logprob-{dataset}-fixture-test.json", payload)
    except torch.cuda.OutOfMemoryError as exc:
        payload = {
            "backend": "hf-logprob",
            "dataset": dataset,
            "oom": True,
            "error": str(exc),
            "gpu": _gpu_stats(),
            "nvidia_smi": _smi(),
        }
        _write(REPORTS / f"eval-hf-logprob-{dataset}-fixture-test-oom.json", payload)
        print("OOM", dataset, exc, flush=True)
        release_cuda()
        torch.cuda.empty_cache()
    except Exception as exc:
        payload = {
            "backend": "hf-logprob",
            "dataset": dataset,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gpu": _gpu_stats(),
            "nvidia_smi": _smi(),
        }
        _write(REPORTS / f"eval-hf-logprob-{dataset}-fixture-test-error.json", payload)
        print("ERROR", dataset, exc, flush=True)
        release_cuda()


def _train_hf_head(dataset: str, epochs: int = 12, batch_size: int = 1) -> None:
    print(f"==== train-head hf {dataset} ====", flush=True)
    print("gpu_before", _gpu_stats(), flush=True)
    print(_smi(), flush=True)
    train = load_jsonl_examples(OUT / dataset / "jsonl" / "train.jsonl")
    val = load_jsonl_examples(OUT / dataset / "jsonl" / "validation.jsonl")
    calib = load_jsonl_examples(OUT / dataset / "jsonl" / "calibration.jsonl")
    test = load_jsonl_examples(OUT / dataset / "jsonl" / "test.jsonl")
    device = torch.device("cuda")
    cfg = TrainConfig(
        epochs=epochs,
        encoder_kind="hf",
        hf_model_id="Qwen/Qwen2.5-0.5B",
        batch_size=batch_size,
    )
    try:
        encoder, head, history = train_option_head(train, cfg, device=device, val_examples=val)
        ckpt = RUNS / f"{dataset}-hf-head.pt"
        save_checkpoint(
            ckpt,
            encoder,
            head,
            extra={
                "encoder_kind": "hf",
                "model_id": "Qwen/Qwen2.5-0.5B",
                "hf_model_id": "Qwen/Qwen2.5-0.5B",
                **history,
            },
        )
        scorer = OptionHeadScorer(encoder, head, model_id="Qwen/Qwen2.5-0.5B")
        pairs = collect_logit_gold(scorer, calib)
        cal = fit_temperature(pairs)
        _write(
            REPORTS / f"calibrate-hf-head-{dataset}.json",
            {
                "temperature": cal.temperature,
                "n": len(pairs),
                "split": "calibration",
                "dataset": dataset,
            },
        )
        report = evaluate_scorer(scorer, test, temperature=cal.temperature, shuffled=True)
        payload = {
            "backend": "option-head",
            "dataset": dataset,
            "temperature": cal.temperature,
            "model_id": "Qwen/Qwen2.5-0.5B",
            "encoder_kind": "hf",
            **report_as_dict(report),
            **history,
            "gpu_after": _gpu_stats(),
        }
        _write(REPORTS / f"eval-hf-head-{dataset}-fixture-test.json", payload)
        print("history", history, flush=True)
        print(_smi(), flush=True)
    except torch.cuda.OutOfMemoryError as exc:
        payload = {
            "backend": "option-head",
            "dataset": dataset,
            "encoder_kind": "hf",
            "oom": True,
            "error": str(exc),
            "gpu": _gpu_stats(),
            "nvidia_smi": _smi(),
        }
        _write(REPORTS / f"eval-hf-head-{dataset}-fixture-test-oom.json", payload)
        print("OOM train-head", dataset, exc, flush=True)
        print(_smi(), flush=True)
    except Exception as exc:
        payload = {
            "backend": "option-head",
            "dataset": dataset,
            "encoder_kind": "hf",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gpu": _gpu_stats(),
            "nvidia_smi": _smi(),
        }
        _write(REPORTS / f"eval-hf-head-{dataset}-fixture-test-error.json", payload)
        print("ERROR train-head", dataset, exc, flush=True)
        print(traceback.format_exc(), flush=True)
        print(_smi(), flush=True)
    finally:
        release_cuda()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _hf_probe() -> None:
    import os

    import jev.data.convert as convert_mod
    from jev.data.banking77 import convert_banking77
    from jev.data.boolq import try_original_boolq

    tmp = REPO / "data" / "hf-probe"
    tmp.mkdir(parents=True, exist_ok=True)
    dest = tmp / "banking77"
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    dest.mkdir()
    convert_mod.LAST_HF_SOURCE = None
    convert_mod.LAST_HF_ERROR = None
    convert_mod.LAST_DOWNLOAD_ERROR = None
    try:
        convert_banking77(dest)
        banking77_ok = True
        banking77_error = None
    except Exception as exc:
        banking77_ok = False
        banking77_error = f"{type(exc).__name__}: {exc}"
    original = try_original_boolq()
    payload = {
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "banking77_ok": banking77_ok,
        "banking77_source": convert_mod.LAST_HF_SOURCE,
        "banking77_error": banking77_error or convert_mod.LAST_HF_ERROR,
        "boolq_original_ok": original is not None,
        "boolq_download_error": convert_mod.LAST_DOWNLOAD_ERROR,
    }
    _write(REPORTS / "hf-download-probe.json", payload)


def main() -> None:
    print("START", _smi(), flush=True)
    _hf_probe()
    release_cuda()
    # Frozen-encoder option head on small fixture menus that fit 4 GiB.
    _train_hf_head("sst5")
    _train_hf_head("boolq")
    # Zero-shot continuation scorer on fixture frozen tests.
    _eval_logprob("boolq")
    release_cuda()
    _eval_logprob("clinc150")
    release_cuda()
    _eval_logprob("wikispeedia")
    release_cuda()
    # BANKING77 77-way: try full fixture test; OOM is evidence.
    _eval_logprob("banking77")
    release_cuda()
    # Wide-menu frozen head: likely OOM on 1650; record either result or OOM.
    _train_hf_head("banking77")
    release_cuda()
    _train_hf_head("clinc150")
    print("GPU_DONE", _smi(), flush=True)


if __name__ == "__main__":
    main()
