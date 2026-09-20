"""BoolQ Noul conversion. Group by Wikipedia title."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.data.convert import freeze_and_write, load_records_jsonl, try_load_hf
from jev.data.manifest import criteria_path, load_criteria
from jev.data.splits import assign_groups
from jev.schema import (
    FORMAT_VERSION,
    ExampleMetadata,
    NoulCriteria,
    NoulQuestion,
    NoulTrainingExample,
)


def convert_boolq(out_dir: Path, fixture: Path | None = None) -> Path:
    spec = load_criteria(criteria_path("boolq"))
    if fixture:
        if fixture.suffix == ".json":
            raw = json.loads(fixture.read_text(encoding="utf-8"))
        else:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in load_records_jsonl(fixture):
                grouped.setdefault(row.get("split", "train"), []).append(row)
            raw = grouped
    else:
        ds = try_load_hf("google/boolq")
        if ds is None:
            raise FileNotFoundError("BoolQ not available: pass --fixture")
        raw = {}
        for split in ds:
            raw[split] = [
                {
                    "id": f"boolq_{split}_{i}",
                    "title": row.get("title") or f"page-{i}",
                    "passage": row["passage"],
                    "question": row["question"],
                    "answer": bool(row["answer"]),
                }
                for i, row in enumerate(ds[split])
            ]
    test_src = [r for r in (raw.get("validation") or raw.get("test") or []) if r.get("answer") is not None]
    train_src = [r for r in raw.get("train", []) if r.get("answer") is not None]
    test_titles = {str(r.get("title") or r["id"]).lower() for r in test_src}
    train_kept = [r for r in train_src if str(r.get("title") or r["id"]).lower() not in test_titles]

    def to_ex(row: dict[str, Any], split: str) -> NoulTrainingExample:
        title = str(row.get("title") or "unknown")
        return NoulTrainingExample(
            id=str(row.get("id") or title),
            type="noul",
            format_version=FORMAT_VERSION,
            criteria_version=str(spec["criteria_version"]),
            state={"title": title, "passage": row["passage"], "question": row["question"]},
            question=NoulQuestion(
                type="noul",
                instructions=spec["instructions"],
                criteria=NoulCriteria(true=spec["criteria"]["true"], false=spec["criteria"]["false"]),
            ),
            gold=bool(row["answer"]),
            metadata=ExampleMetadata(domain="boolq", group_id=title.lower(), source="google/boolq", split=split),  # type: ignore[arg-type]
        )

    groups = [str(r.get("title") or r["id"]).lower() for r in train_kept]
    assigned = assign_groups(groups, seed=2, fractions=(0.8, 0.1, 0.1))
    splits: dict[str, list[NoulTrainingExample]] = {"train": [], "validation": [], "calibration": [], "test": []}
    for row in train_kept:
        dest = assigned[str(row.get("title") or row["id"]).lower()]
        splits[dest].append(to_ex(row, dest))
    splits["test"] = [to_ex(r, "test") for r in test_src]
    return freeze_and_write(
        dataset="boolq",
        primitive="noul",
        criteria_file=criteria_path("boolq"),
        converter="jev.data.boolq",
        license_name="CC BY-SA 3.0",
        source="google/boolq",
        split_rule="official validation frozen as test; remaining titles 80/10/10; no title overlap",
        examples_by_split=splits,
        out_dir=out_dir,
    )
