"""Convert fixtures, train hashing heads, evaluate fake/hashing. CPU only."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from jev.canonical import canonical_dumps
from jev.data.banking77 import convert_banking77
from jev.data.boolq import convert_boolq
from jev.data.clinc150 import convert_clinc150
from jev.data.sst5 import convert_sst5
from jev.data.wikispeedia import convert_wikispeedia
from jev.evaluation import evaluate_scorer, report_as_dict
from jev.scoring.factory import build_scorer
from jev.scoring.option_head import (
    OptionHeadScorer,
    TrainConfig,
    load_checkpoint,
    load_jsonl_examples,
    save_checkpoint,
    train_option_head,
)
from jev.calibration import collect_logit_gold, fit_temperature
import torch

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures" / "data"
OUT = REPO / "data" / "converted-fixtures"
REPORTS = REPO / "reports" / "v0"
RUNS = REPO / "runs"


def _convert() -> None:
    jobs = {
        "banking77": convert_banking77,
        "sst5": convert_sst5,
        "boolq": convert_boolq,
        "clinc150": convert_clinc150,
        "wikispeedia": convert_wikispeedia,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in jobs.items():
        dest = OUT / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        fn(dest, fixture=FIXTURES / f"{name}.json")
        manifest_src = dest / "manifest.json"
        manifest_dst = REPORTS / "manifests" / f"{name}-fixture.json"
        shutil.copy2(manifest_src, manifest_dst)
        payload = json.loads(manifest_src.read_text())
        print(
            name,
            {k: v.get("n") for k, v in payload["splits"].items()},
            "source",
            payload.get("source"),
        )


def _write_eval(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    print(path.name, payload.get("backend"), "n", payload.get("n"), "acc", payload.get("accuracy"))


def _run_one(dataset: str) -> None:
    root = OUT / dataset
    train = load_jsonl_examples(root / "jsonl" / "train.jsonl")
    val = load_jsonl_examples(root / "jsonl" / "validation.jsonl")
    calib = load_jsonl_examples(root / "jsonl" / "calibration.jsonl")
    test = load_jsonl_examples(root / "jsonl" / "test.jsonl")
    device = torch.device("cpu")
    cfg = TrainConfig(epochs=12, encoder_kind="hashing", batch_size=16)
    encoder, head, history = train_option_head(train, cfg, device=device, val_examples=val)
    ckpt = RUNS / f"{dataset}-hashing-head.pt"
    save_checkpoint(
        ckpt,
        encoder,
        head,
        extra={"encoder_kind": "hashing", "model_id": "option-attention-hashing", **history},
    )
    scorer = OptionHeadScorer(encoder, head, model_id="option-attention-hashing")
    pairs = collect_logit_gold(scorer, calib)
    cal = fit_temperature(pairs)
    cal_path = REPORTS / "metrics" / f"calibrate-hashing-{dataset}.json"
    _write_eval(cal_path, {"temperature": cal.temperature, "n": len(pairs), "split": "calibration", "dataset": dataset})
    report = evaluate_scorer(scorer, test, temperature=cal.temperature, shuffled=True)
    payload = {
        "backend": "option-head-hashing",
        "dataset": dataset,
        "temperature": cal.temperature,
        "model_id": "option-attention-hashing",
        **report_as_dict(report),
        **history,
    }
    _write_eval(REPORTS / "metrics" / f"eval-hashing-{dataset}-test.json", payload)
    fake = build_scorer("fake")
    fake_report = evaluate_scorer(fake, test, temperature=1.0, shuffled=True)
    fake_payload = {
        "backend": "fake",
        "dataset": dataset,
        "model_id": fake.model_id,
        **report_as_dict(fake_report),
    }
    _write_eval(REPORTS / "metrics" / f"eval-fake-{dataset}-test.json", fake_payload)


def main() -> None:
    _convert()
    RUNS.mkdir(parents=True, exist_ok=True)
    for dataset in ("banking77", "sst5", "boolq", "clinc150", "wikispeedia"):
        print("==== hashing", dataset, "====")
        _run_one(dataset)


if __name__ == "__main__":
    main()
