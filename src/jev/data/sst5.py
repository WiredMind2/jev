"""SST-5 Score conversion. Sentence-level only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.data.convert import freeze_and_write, load_records_jsonl, try_load_hf
from jev.data.manifest import criteria_path, load_criteria
from jev.data.splits import stratified_indices
from jev.schema import FORMAT_VERSION, ExampleMetadata, ScoreQuestion, ScoreTrainingExample

LABEL_MAP = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4,
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
}


def _from_hf() -> dict[str, list[dict[str, Any]]] | None:
    ds = try_load_hf("SetFit/sst5")
    if ds is None:
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ds:
        rows = []
        for i, row in enumerate(ds[split]):
            label = row.get("label")
            if isinstance(label, str):
                gold = LABEL_MAP[label.lower()]
            else:
                gold = int(label)
            rows.append({"id": f"sst5_{split}_{i}", "text": row["text"], "label": gold, "split": split})
        out[split] = rows
    return out


def convert_sst5(out_dir: Path, fixture: Path | None = None) -> Path:
    spec = load_criteria(criteria_path("sst5"))
    if fixture:
        raw = json.loads(fixture.read_text(encoding="utf-8")) if fixture.suffix == ".json" else None
        if raw is None:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in load_records_jsonl(fixture):
                grouped.setdefault(row.get("split", "train"), []).append(row)
            raw = grouped
    else:
        raw = _from_hf()
    if raw is None:
        raise FileNotFoundError("SST-5 not available: pass --fixture")

    def to_ex(row: dict[str, Any], split: str) -> ScoreTrainingExample:
        gold = LABEL_MAP[row["label"]] if not isinstance(row["label"], int) else int(row["label"])
        return ScoreTrainingExample(
            id=str(row.get("id") or row["text"][:40]),
            type="score",
            format_version=FORMAT_VERSION,
            criteria_version=str(spec["criteria_version"]),
            state={"text": row["text"]},
            question=ScoreQuestion(type="score", instructions=spec["instructions"], criteria=spec["criteria"]),
            gold=gold,
            metadata=ExampleMetadata(domain="sst5", group_id=str(row.get("id")), source="SetFit/sst5", split=split),  # type: ignore[arg-type]
        )

    test_rows = raw.get("test") or raw.get("validation") or []
    train_rows = raw.get("train", [])
    if "validation" in raw and raw.get("test"):
        val_official = raw["validation"]
        labels = [str(r["label"]) for r in train_rows]
        carved = stratified_indices(labels, seed=1, fractions=(0.9, 0.0, 0.1))
        splits = {
            "train": [to_ex(train_rows[i], "train") for i in carved["train"]],
            "validation": [to_ex(r, "validation") for r in val_official],
            "calibration": [to_ex(train_rows[i], "calibration") for i in carved["calibration"]],
            "test": [to_ex(r, "test") for r in test_rows],
        }
    else:
        labels = [str(r["label"]) for r in train_rows]
        carved = stratified_indices(labels, seed=1, fractions=(0.8, 0.1, 0.1))
        splits = {
            "train": [to_ex(train_rows[i], "train") for i in carved["train"]],
            "validation": [to_ex(train_rows[i], "validation") for i in carved["validation"]],
            "calibration": [to_ex(train_rows[i], "calibration") for i in carved["calibration"]],
            "test": [to_ex(r, "test") for r in test_rows],
        }
    return freeze_and_write(
        dataset="sst5",
        primitive="score",
        criteria_file=criteria_path("sst5"),
        converter="jev.data.sst5",
        license_name="LicenseRef-SST-research",
        source="SetFit/sst5",
        split_rule="official test/validation preferred; 10% of train carved as calibration",
        examples_by_split=splits,
        out_dir=out_dir,
        notes="Sentence-level only. Do not shuffle Score level order.",
    )
