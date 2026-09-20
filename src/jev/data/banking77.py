"""BANKING77 Choice conversion (Casanueva et al., CC BY 4.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.data import convert as convert_mod
from jev.data.convert import freeze_and_write, load_records_jsonl, try_load_hf_first
from jev.data.manifest import criteria_path, load_criteria
from jev.data.splits import stratified_indices
from jev.schema import FORMAT_VERSION, ChoiceQuestion, ChoiceTrainingExample, ExampleMetadata


def _rows_from_hf() -> dict[str, list[dict[str, Any]]] | None:
    ds = try_load_hf_first(
        [
            ("PolyAI/banking77", {}),
            ("mteb/banking77", {}),
        ]
    )
    if ds is None:
        return None
    out: dict[str, list[dict[str, Any]]] = {}
    names = getattr(ds["train"].features.get("label"), "names", None)
    for split in ds:
        rows = []
        for i, row in enumerate(ds[split]):
            raw_label = row.get("label_text") or row.get("label")
            if names is not None and not isinstance(raw_label, str):
                label = names[int(raw_label)]
            else:
                label = str(raw_label)
            rows.append({"id": f"banking77_{split}_{i}", "text": row["text"], "label": label})
        out[split] = rows
    return out


def _rows_from_fixture(path: Path) -> dict[str, list[dict[str, Any]]]:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    # jsonl with split field
    grouped: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for row in load_records_jsonl(path):
        grouped.setdefault(row.get("split", "train"), []).append(row)
    return grouped


def convert_banking77(out_dir: Path, fixture: Path | None = None) -> Path:
    criteria_file = criteria_path("banking77")
    spec = load_criteria(criteria_file)
    criteria = spec["criteria"]
    instructions = spec["instructions"]
    source = "PolyAI/banking77"
    if fixture:
        raw = _rows_from_fixture(fixture)
        source = str(fixture)
    else:
        raw = _rows_from_hf()
        source = convert_mod.LAST_HF_SOURCE or "mteb/banking77"
    if raw is None:
        raise FileNotFoundError(
            "BANKING77 not available: pass --fixture or install datasets + network "
            "(tries PolyAI/banking77 then mteb/banking77)"
        )
    official_test = raw.get("test", [])
    official_train = raw.get("train", raw.get("validation", []))

    def to_example(row: dict[str, Any], split: str) -> ChoiceTrainingExample:
        gold = str(row["label"])
        if gold not in criteria:
            raise ValueError(f"unknown banking77 label {gold!r}")
        return ChoiceTrainingExample(
            id=str(row.get("id") or row["text"][:40]),
            type="choice",
            format_version=FORMAT_VERSION,
            criteria_version=str(spec["criteria_version"]),
            state={"text": row["text"]},
            question=ChoiceQuestion(type="choice", instructions=instructions, criteria=criteria),
            gold=gold,
            metadata=ExampleMetadata(
                domain="banking77",
                group_id=str(row.get("id") or row["text"]),
                source=source,
                split=split,  # type: ignore[arg-type]
            ),
        )

    labels = [str(r["label"]) for r in official_train]
    carved = stratified_indices(labels, seed=0, fractions=(0.8, 0.1, 0.1))
    splits = {
        "train": [to_example(official_train[i], "train") for i in carved["train"]],
        "validation": [to_example(official_train[i], "validation") for i in carved["validation"]],
        "calibration": [to_example(official_train[i], "calibration") for i in carved["calibration"]],
        "test": [to_example(r, "test") for r in official_test],
    }
    return freeze_and_write(
        dataset="banking77",
        primitive="choice",
        criteria_file=criteria_file,
        converter="jev.data.banking77",
        license_name="CC BY 4.0",
        source=source,
        split_rule="official test frozen; official train 80/10/10 stratified by intent",
        examples_by_split=splits,
        out_dir=out_dir,
        notes="Do not use intent as group_id. Full 77-option menu on every row.",
    )
