"""Official frozen-split evidence on the GTX 1650.

Does not overwrite fixture JSON under reports/v0/metrics/eval-*-fixture-*.
OOM is recorded with nvidia-smi / torch stats. Never invents metrics.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any

from jev.calibration import collect_logit_gold, fit_temperature
from jev.canonical import canonical_dumps
from jev.evaluation import evaluate_scorer, report_as_dict, stratified_sample
from jev.hardware import release_cuda
from jev.scoring.baselines import MajorityScorer, TfidfLinearScorer
from jev.scoring.fake import FakeScorer
from jev.scoring.option_head import (
    OptionHeadScorer,
    TrainConfig,
    load_jsonl_examples,
    save_checkpoint,
    train_option_head,
)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "hf-attempt"
REPORTS = REPO / "reports" / "v0" / "metrics"
SLICES = REPO / "reports" / "v0" / "slices"
RUNS = REPO / "runs"

# jev-eval-style slices. Full official tests are converted; scoring 3076×77
# (BANKING77) or 39786 Wikispeedia rows is the Colab T4 job.
JOBS: list[dict[str, Any]] = [
    {
        "name": "sst5",
        "test_limit": 300,
        "head_train_cap": 1024,
        "head_val_cap": 128,
        "head_calib_cap": 128,
        "hash_train_cap": 4000,
        "hash_epochs": 8,
        "head_epochs": 8,
    },
    {
        "name": "boolq",
        "test_limit": 150,
        "head_train_cap": 1024,
        "head_val_cap": 128,
        "head_calib_cap": 128,
        "hash_train_cap": 4000,
        "hash_epochs": 8,
        "head_epochs": 8,
    },
    {
        "name": "banking77",
        "test_limit": 300,
        "head_train_cap": 768,
        "head_val_cap": 96,
        "head_calib_cap": 96,
        "hash_train_cap": 4000,
        "hash_epochs": 8,
        "head_epochs": 8,
    },
    {
        "name": "clinc150",
        "test_limit": 300,
        "head_train_cap": 768,
        "head_val_cap": 96,
        "head_calib_cap": 96,
        "hash_train_cap": 3000,
        "hash_epochs": 6,
        "head_epochs": 8,
    },
    {
        "name": "wikispeedia",
        "test_limit": 80,
        "head_train_cap": 256,
        "head_val_cap": 64,
        "head_calib_cap": 64,
        "hash_train_cap": 2000,
        "hash_epochs": 6,
        "head_epochs": 6,
    },
]


def _smi() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def _gpu_stats() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "nvidia_smi": _smi()}
    torch.cuda.synchronize()
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(0),
        "allocated_miB": round(torch.cuda.memory_allocated() / 1024 / 1024, 1),
        "reserved_miB": round(torch.cuda.memory_reserved() / 1024 / 1024, 1),
        "max_allocated_miB": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1),
        "nvidia_smi": _smi(),
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "wrote": str(path),
                "accuracy": payload.get("accuracy"),
                "n": payload.get("n"),
                "oom": payload.get("oom"),
                "error": payload.get("error"),
            },
            default=str,
        ),
        flush=True,
    )


def _progress(label: str, every: int = 10):
    t0 = time.time()

    def inner(i: int, n: int, stage: str) -> None:
        if i == 0 or (i + 1) % every == 0 or i + 1 == n:
            print(
                f"  {label} {stage} {i + 1}/{n} elapsed={time.time() - t0:.0f}s {_smi().splitlines()[-1] if _smi() else ''}",
                flush=True,
            )

    return inner


def _load_capped(path: Path, cap: int = 0) -> list[Any]:
    if not path.exists():
        return []
    if cap <= 0:
        return load_jsonl_examples(path)
    from jev.schema import parse_training_example

    rows: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(parse_training_example(json.loads(line)))
            if len(rows) >= cap:
                break
    return rows


def _paths(name: str) -> dict[str, Path]:
    root = DATA / name
    return {
        "root": root,
        "train": root / "jsonl" / "train.jsonl",
        "validation": root / "jsonl" / "validation.jsonl",
        "calibration": root / "jsonl" / "calibration.jsonl",
        "test": root / "jsonl" / "test.jsonl",
        "manifest": root / "manifest.json",
    }


def _load_splits(name: str, *, train_cap: int = 0, extra_splits: bool = True) -> dict[str, list[Any]]:
    paths = _paths(name)
    out = {
        "train": _load_capped(paths["train"], train_cap),
        "test": load_jsonl_examples(paths["test"]),
    }
    if extra_splits:
        out["validation"] = load_jsonl_examples(paths["validation"]) if paths["validation"].exists() else []
        out["calibration"] = load_jsonl_examples(paths["calibration"]) if paths["calibration"].exists() else []
    else:
        out["validation"] = []
        out["calibration"] = []
    return out


def _slice_test(name: str, test: list[Any], limit: int) -> list[Any]:
    used = stratified_sample(test, limit, seed=0) if limit else list(test)
    SLICES.mkdir(parents=True, exist_ok=True)
    _write(
        SLICES / f"{name}-official-test-{len(used)}.json",
        {
            "dataset": name,
            "seed": 0,
            "n": len(used),
            "n_full_test": len(test),
            "ids": [str(getattr(ex, "id", i)) for i, ex in enumerate(used)],
        },
    )
    return used


def hygiene() -> None:
    payload: dict[str, Any] = {"ok": True, "datasets": {}}
    for job in JOBS:
        name = job["name"]
        paths = _paths(name)
        manifest = json.loads(paths["manifest"].read_text())
        counts = {k: v.get("n") for k, v in manifest.get("splits", {}).items()}
        notes: list[str] = []
        test_rows = _load_capped(paths["test"], 250)
        train_rows = _load_capped(paths["train"], 80)
        gold_counts: dict[str, int] = {}
        with paths["test"].open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                gold = str(json.loads(line).get("gold"))
                gold_counts[gold] = gold_counts.get(gold, 0) + 1
        if name == "banking77":
            bad = [r.id for r in test_rows + train_rows if r.metadata.group_id == r.gold]
            if bad:
                notes.append(f"group_id==intent n={len(bad)}")
            widths = {len(r.question.criteria) for r in test_rows[:20]}
            if widths - {77}:
                notes.append(f"menu_sizes={sorted(widths)}")
        if name == "boolq":
            test_groups = {r.metadata.group_id for r in test_rows}
            train_groups = {r.metadata.group_id for r in train_rows}
            overlap = test_groups & train_groups
            if overlap:
                notes.append(f"group overlap in sample n={len(overlap)}")
            notes.append(f"test_title_sample={[r.state.get('title') for r in test_rows[:4]]}")
        if name == "wikispeedia":
            for r in test_rows + train_rows:
                if r.metadata.group_id != r.state["target"]:
                    notes.append("group_id != target")
                    break
                if len(r.question.criteria) > 255:
                    notes.append("menu > 255")
                    break
                if r.gold not in r.question.criteria:
                    notes.append("gold missing from menu")
                    break
            notes.append(f"sample_menu_max={max(len(r.question.criteria) for r in test_rows)}")
        if name == "clinc150":
            n_oos = gold_counts.get("out_of_scope", 0)
            notes.append(f"test_oos={n_oos}")
            notes.append(f"test_unique_gold={len(gold_counts)}")
            if len(gold_counts) < 140:
                notes.append("too few unique golds; ClassLabel ints likely unmapped")
            if any(len(r.question.criteria) != 151 for r in test_rows[:20]):
                notes.append("menu != 151")
        dataset_ok = not any(
            x in n
            for n in notes
            for x in (
                "==intent",
                "overlap",
                "missing",
                "!= target",
                "> 255",
                "!= 151",
                "too few unique golds",
            )
        )
        nonempty = all(int(counts.get(k) or 0) > 0 for k in ("train", "validation", "calibration", "test"))
        payload["datasets"][name] = {
            "counts": counts,
            "source": manifest.get("source"),
            "split_rule": manifest.get("split_rule"),
            "notes": notes,
            "ok": dataset_ok and nonempty,
        }
        payload["ok"] = payload["ok"] and payload["datasets"][name]["ok"]
    _write(REPO / "reports" / "v0" / "official-split-hygiene.json", payload)


def cpu_baselines(only: str | None = None) -> None:
    for job in JOBS:
        name = job["name"]
        if only and name != only:
            continue
        print(f"==== cpu {name} ====", flush=True)
        paths = _paths(name)
        train = _load_capped(paths["train"], job["hash_train_cap"] * 2 if job["hash_train_cap"] else 0)
        val = _load_capped(paths["validation"], max(64, job["hash_train_cap"] // 8) if job["hash_train_cap"] else 0)
        calib = _load_capped(paths["calibration"], max(64, job["hash_train_cap"] // 8) if job["hash_train_cap"] else 0)
        test_full = load_jsonl_examples(paths["test"])
        if job["hash_train_cap"]:
            train = stratified_sample(train, min(len(train), job["hash_train_cap"]), seed=0)
        test = _slice_test(name, test_full, job["test_limit"])

        fake = FakeScorer()
        _write(
            REPORTS / f"eval-fake-{name}-official-slice.json",
            {
                "backend": "fake",
                "dataset": name,
                "split": "official-test-slice",
                **report_as_dict(evaluate_scorer(fake, test, shuffled=True, progress=_progress(f"fake-{name}", 50))),
            },
        )
        maj = MajorityScorer.fit(train)
        _write(
            REPORTS / f"eval-majority-{name}-official-slice.json",
            {
                "backend": "majority",
                "dataset": name,
                "split": "official-test-slice",
                "train_n": len(train),
                **report_as_dict(evaluate_scorer(maj, test, shuffled=True)),
            },
        )
        t0 = time.time()
        tfidf = TfidfLinearScorer.fit(train)
        _write(
            REPORTS / f"eval-tfidf-{name}-official-slice.json",
            {
                "backend": "tfidf-linear",
                "dataset": name,
                "split": "official-test-slice",
                "train_n": len(train),
                "fit_seconds": round(time.time() - t0, 2),
                **report_as_dict(evaluate_scorer(tfidf, test, shuffled=True)),
            },
        )
        ckpt = RUNS / f"hashing-{name}-official.pt"
        t0 = time.time()
        encoder, head, history = train_option_head(
            train,
            TrainConfig(epochs=job["hash_epochs"], batch_size=8, seed=0, lr=3e-3, patience=4),
            val_examples=val or None,
        )
        save_checkpoint(ckpt, encoder, head, extra={"encoder_kind": "hashing", "dataset": name, **history})
        scorer = OptionHeadScorer(encoder, head, model_id=f"option-attention-hashing-{name}")
        temperature = 1.0
        if calib:
            cal = fit_temperature(collect_logit_gold(scorer, calib))
            temperature = cal.temperature
            _write(
                REPORTS / f"calibrate-hashing-{name}-official.json",
                {
                    "temperature": temperature,
                    "n": len(calib),
                    "split": "calibration",
                    "dataset": name,
                    "note": "calibration split only; not used for early stopping",
                },
            )
        report = evaluate_scorer(scorer, test, temperature=temperature, shuffled=True)
        _write(
            REPORTS / f"eval-hashing-{name}-official-slice.json",
            {
                "backend": "option-head-hashing",
                "dataset": name,
                "split": "official-test-slice",
                "temperature": temperature,
                "train_n": len(train),
                "test_n": len(test),
                "train_seconds": round(time.time() - t0, 2),
                **history,
                **report_as_dict(report),
            },
        )


def cuda_zero_shot(only: str | None = None) -> None:
    import torch

    from jev.hardware import load_hardware_pin
    from jev.scoring.factory import build_scorer

    pin = load_hardware_pin()
    print("zero-shot load", pin.zero_shot_model, _smi(), flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    try:
        scorer = build_scorer("hf-logprob", hf_model_id=pin.zero_shot_model, device="cuda")
    except Exception as exc:
        _write(
            REPORTS / "eval-hf-logprob-official-error.json",
            {"error": f"{type(exc).__name__}: {exc}", "gpu": _gpu_stats(), "nvidia_smi": _smi()},
        )
        release_cuda()
        return
    try:
        for job in JOBS:
            name = job["name"]
            if only and name != only:
                continue
            paths = _paths(name)
            splits_test = load_jsonl_examples(paths["test"])
            n_full = int(json.loads(paths["manifest"].read_text())["splits"]["test"]["n"])
            test = _slice_test(name, splits_test, job["test_limit"])
            print(f"==== hf-logprob {name} n={len(test)}/{n_full} ====", flush=True)
            print("gpu_before", _gpu_stats(), flush=True)
            t0 = time.time()
            try:
                report = evaluate_scorer(
                    scorer,
                    test,
                    temperature=1.0,
                    shuffled=True,
                    progress=_progress(f"logprob-{name}", 5),
                )
                _write(
                    REPORTS / f"eval-hf-logprob-{name}-official-slice.json",
                    {
                        "backend": "hf-logprob",
                        "dataset": name,
                        "split": "official-test-slice",
                        "model_id": pin.zero_shot_model,
                        "temperature": 1.0,
                        "n_full_test": n_full,
                        "seconds": round(time.time() - t0, 2),
                        **report_as_dict(report),
                        "gpu_after": _gpu_stats(),
                    },
                )
            except torch.cuda.OutOfMemoryError as exc:
                _write(
                    REPORTS / f"eval-hf-logprob-{name}-official-slice-oom.json",
                    {
                        "backend": "hf-logprob",
                        "dataset": name,
                        "oom": True,
                        "error": str(exc),
                        "gpu": _gpu_stats(),
                        "nvidia_smi": _smi(),
                    },
                )
                print("OOM", name, exc, flush=True)
                release_cuda()
                break
            except Exception as exc:
                _write(
                    REPORTS / f"eval-hf-logprob-{name}-official-slice-error.json",
                    {
                        "backend": "hf-logprob",
                        "dataset": name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                        "gpu": _gpu_stats(),
                    },
                )
                print("ERROR", name, exc, flush=True)
                release_cuda()
                break
    finally:
        del scorer
        release_cuda()


def cuda_heads(only: str | None = None) -> None:
    import torch

    device = torch.device("cuda")
    for job in JOBS:
        name = job["name"]
        if only and name != only:
            continue
        print(f"==== train-head hf {name} ====", flush=True)
        print(_smi(), flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        paths = _paths(name)
        n_full = int(json.loads(paths["manifest"].read_text())["splits"]["test"]["n"])
        train = _load_capped(paths["train"], job["head_train_cap"] * 2)
        val = _load_capped(paths["validation"], job["head_val_cap"] * 2)
        calib = _load_capped(paths["calibration"], job["head_calib_cap"] * 2)
        test_full = load_jsonl_examples(paths["test"])
        train = stratified_sample(train, min(len(train), job["head_train_cap"]), seed=0)
        val = stratified_sample(val, min(len(val), job["head_val_cap"]), seed=2) if val else []
        calib = stratified_sample(calib, min(len(calib), job["head_calib_cap"]), seed=3) if calib else []
        test = _slice_test(name, test_full, job["test_limit"])
        cfg = TrainConfig(
            epochs=job["head_epochs"],
            encoder_kind="hf",
            hf_model_id="Qwen/Qwen2.5-0.5B",
            batch_size=1,
            patience=3,
        )
        t0 = time.time()
        try:
            encoder, head, history = train_option_head(train, cfg, device=device, val_examples=val or None)
            ckpt = RUNS / f"{name}-hf-head-official.pt"
            save_checkpoint(
                ckpt,
                encoder,
                head,
                extra={
                    "encoder_kind": "hf",
                    "model_id": "Qwen/Qwen2.5-0.5B",
                    "hf_model_id": "Qwen/Qwen2.5-0.5B",
                    "dataset": name,
                    "train_n": len(train),
                    **history,
                },
            )
            scorer = OptionHeadScorer(encoder, head, model_id="Qwen/Qwen2.5-0.5B")
            temperature = 1.0
            if calib:
                cal = fit_temperature(collect_logit_gold(scorer, calib))
                temperature = cal.temperature
                _write(
                    REPORTS / f"calibrate-hf-head-{name}-official.json",
                    {
                        "temperature": temperature,
                        "n": len(calib),
                        "split": "calibration",
                        "dataset": name,
                        "note": "calibration split only; validation used for early stopping",
                    },
                )
            report = evaluate_scorer(
                scorer,
                test,
                temperature=temperature,
                shuffled=True,
                progress=_progress(f"head-{name}", 10),
            )
            _write(
                REPORTS / f"eval-hf-head-{name}-official-slice.json",
                {
                    "backend": "option-head",
                    "dataset": name,
                    "split": "official-test-slice",
                    "temperature": temperature,
                    "model_id": "Qwen/Qwen2.5-0.5B",
                    "encoder_kind": "hf",
                    "train_n": len(train),
                    "val_n": len(val),
                    "calib_n": len(calib),
                    "n_full_test": n_full,
                    "seconds": round(time.time() - t0, 2),
                    **report_as_dict(report),
                    **history,
                    "gpu_after": _gpu_stats(),
                },
            )
            print("history", history, _smi(), flush=True)
        except torch.cuda.OutOfMemoryError as exc:
            _write(
                REPORTS / f"eval-hf-head-{name}-official-slice-oom.json",
                {
                    "backend": "option-head",
                    "dataset": name,
                    "encoder_kind": "hf",
                    "oom": True,
                    "error": str(exc),
                    "gpu": _gpu_stats(),
                    "nvidia_smi": _smi(),
                    "seconds": round(time.time() - t0, 2),
                },
            )
            print("OOM train-head", name, exc, flush=True)
            print(_smi(), flush=True)
        except Exception as exc:
            _write(
                REPORTS / f"eval-hf-head-{name}-official-slice-error.json",
                {
                    "backend": "option-head",
                    "dataset": name,
                    "encoder_kind": "hf",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "gpu": _gpu_stats(),
                    "seconds": round(time.time() - t0, 2),
                },
            )
            print("ERROR train-head", name, exc, flush=True)
            print(traceback.format_exc(), flush=True)
        finally:
            release_cuda()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["hygiene", "cpu", "cuda-zero-shot", "cuda-head", "all"],
        default="all",
    )
    parser.add_argument("--only", default=None, help="single dataset name")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    print("START", _smi(), flush=True)
    if args.phase in {"hygiene", "all"}:
        hygiene()
    if args.phase in {"cpu", "all"}:
        cpu_baselines(args.only)
    if args.phase in {"cuda-zero-shot", "all"}:
        cuda_zero_shot(args.only)
    if args.phase in {"cuda-head", "all"}:
        cuda_heads(args.only)
    print("DONE", _smi(), flush=True)


if __name__ == "__main__":
    main()
