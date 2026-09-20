"""Reproduce v0 public-task CPU numbers (majority, TF-IDF, hashing head)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from jev.calibration import collect_logit_gold, fit_temperature
from jev.canonical import canonical_dumps
from jev.evaluation import evaluate_scorer, report_as_dict, stratified_sample
from jev.scoring.baselines import MajorityScorer, TfidfLinearScorer
from jev.scoring.option_head import (
    OptionHeadScorer,
    TrainConfig,
    load_jsonl_examples,
    save_checkpoint,
    train_option_head,
)

REPO = Path(__file__).resolve().parents[2]
REPORTS = REPO / "reports" / "v0" / "metrics"
RUNS = REPO / "runs"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    summary = {k: payload.get(k) for k in ("accuracy", "n", "temperature")}
    print(json.dumps({"wrote": str(path), **summary}, default=str))


def run_dataset(
    name: str,
    data_dir: Path,
    *,
    train_cap: int = 0,
    test_cap: int = 0,
    epochs: int = 12,
) -> None:
    train_p = data_dir / "jsonl" / "train.jsonl"
    val_p = data_dir / "jsonl" / "validation.jsonl"
    calib_p = data_dir / "jsonl" / "calibration.jsonl"
    test_p = data_dir / "jsonl" / "test.jsonl"
    if not train_p.exists() or not test_p.exists():
        print(f"skip {name}: missing jsonl")
        return
    train = load_jsonl_examples(train_p)
    val = load_jsonl_examples(val_p) if val_p.exists() else []
    calib = load_jsonl_examples(calib_p) if calib_p.exists() else []
    test = load_jsonl_examples(test_p)
    if train_cap:
        train = stratified_sample(train, train_cap, seed=0)
    if test_cap:
        test = stratified_sample(test, test_cap, seed=1)
    if val and train_cap:
        val = stratified_sample(val, min(len(val), max(32, train_cap // 8)), seed=2)
    if calib and train_cap:
        calib = stratified_sample(calib, min(len(calib), max(32, train_cap // 8)), seed=3)

    maj = MajorityScorer.fit(train)
    _write(
        REPORTS / f"eval-majority-{name}-test.json",
        {"backend": "majority", "dataset": name, **report_as_dict(evaluate_scorer(maj, test, shuffled=True))},
    )
    t0 = time.time()
    tfidf = TfidfLinearScorer.fit(train)
    _write(
        REPORTS / f"eval-tfidf-{name}-test.json",
        {
            "backend": "tfidf-linear",
            "dataset": name,
            "fit_seconds": round(time.time() - t0, 2),
            **report_as_dict(evaluate_scorer(tfidf, test, shuffled=True)),
        },
    )

    ckpt = RUNS / f"hashing-{name}.pt"
    t0 = time.time()
    encoder, head, history = train_option_head(
        train,
        TrainConfig(epochs=epochs, batch_size=8, seed=0, lr=3e-3, patience=4),
        val_examples=val or None,
    )
    save_checkpoint(ckpt, encoder, head, extra={"encoder_kind": "hashing", "dataset": name, **history})
    scorer = OptionHeadScorer(encoder, head, model_id=f"option-attention-hashing-{name}")
    temperature = 1.0
    if calib:
        cal = fit_temperature(collect_logit_gold(scorer, calib))
        temperature = cal.temperature
        _write(
            REPORTS / f"calibrate-hashing-{name}.json",
            {"temperature": temperature, "n": len(calib), "split": "calibration", "dataset": name},
        )
    report = evaluate_scorer(scorer, test, temperature=temperature, shuffled=True)
    _write(
        REPORTS / f"eval-hashing-{name}-hf-test.json",
        {
            "backend": "option-head-hashing",
            "dataset": name,
            "temperature": temperature,
            "train_n": len(train),
            "test_n": len(test),
            "train_seconds": round(time.time() - t0, 2),
            **history,
            **report_as_dict(report),
        },
    )


def run_qwen_slices() -> None:
    from jev.hardware import load_hardware_pin, release_cuda
    from jev.scoring.factory import build_scorer

    pin = load_hardware_pin()
    jobs = [
        ("banking77", REPO / "data" / "hf-attempt" / "banking77" / "jsonl" / "test.jsonl", 300),
        ("sst5", REPO / "data" / "hf-attempt" / "sst5" / "jsonl" / "test.jsonl", 300),
        ("boolq", REPO / "data" / "hf-attempt" / "boolq" / "jsonl" / "test.jsonl", 150),
        ("wikispeedia", REPO / "data" / "hf-attempt" / "wikispeedia" / "jsonl" / "test.jsonl", 80),
    ]
    print("loading", pin.zero_shot_model)
    t0 = time.time()
    try:
        scorer = build_scorer("hf-logprob", hf_model_id=pin.zero_shot_model, device="cuda")
    except Exception as exc:
        _write(REPORTS / "eval-hf-logprob-public-error.json", {"error": str(exc)})
        release_cuda()
        return
    print("loaded in", round(time.time() - t0, 2), "s")
    try:
        for name, path, limit in jobs:
            if not path.exists():
                print("skip missing", path)
                continue
            examples = load_jsonl_examples(path)
            examples = stratified_sample(examples, limit, seed=0)
            print(f"qwen eval {name} n={len(examples)}")
            t1 = time.time()
            try:
                report = evaluate_scorer(scorer, examples, temperature=1.0, shuffled=True)
                _write(
                    REPORTS / f"eval-hf-logprob-{name}-slice.json",
                    {
                        "backend": "hf-logprob",
                        "dataset": name,
                        "model_id": pin.zero_shot_model,
                        "temperature": 1.0,
                        "seconds": round(time.time() - t1, 2),
                        **report_as_dict(report),
                    },
                )
            except Exception as exc:
                _write(
                    REPORTS / f"eval-hf-logprob-{name}-slice.json",
                    {"error": str(type(exc).__name__) + ": " + str(exc), "dataset": name, "n_attempted": len(examples)},
                )
                print("OOM/error", name, exc)
                release_cuda()
                break
    finally:
        del scorer
        release_cuda()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["cpu", "cuda", "all"], default="cpu")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if args.phase in {"cpu", "all"}:
        jobs = [
            ("banking77", REPO / "data" / "hf-attempt" / "banking77", 4000, 500, 8),
            ("sst5", REPO / "data" / "hf-attempt" / "sst5", 0, 0, 12),
            ("boolq", REPO / "data" / "hf-attempt" / "boolq", 0, 500, 8),
            ("clinc150", REPO / "data" / "hf-attempt" / "clinc150", 3000, 400, 6),
            ("wikispeedia", REPO / "data" / "hf-attempt" / "wikispeedia", 4000, 400, 10),
        ]
        for name, path, train_cap, test_cap, epochs in jobs:
            print(f"=== {name} {path} ===")
            run_dataset(name, path, train_cap=train_cap, test_cap=test_cap, epochs=epochs)
    if args.phase in {"cuda", "all"}:
        run_qwen_slices()


if __name__ == "__main__":
    main()
