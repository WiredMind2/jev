"""CLINC150 Choice + out_of_scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.data.clinc_intents import clinc_criteria as frozen_clinc_criteria
from jev.data.convert import freeze_and_write, try_load_hf_first
from jev.data.manifest import criteria_path, load_criteria
from jev.schema import FORMAT_VERSION, ChoiceQuestion, ChoiceTrainingExample, ExampleMetadata


def clinc_criteria() -> dict[str, str]:
    criteria_file = criteria_path("clinc150")
    if criteria_file.exists():
        return load_criteria(criteria_file)["criteria"]
    return frozen_clinc_criteria()


def convert_clinc150(out_dir: Path, fixture: Path | None = None) -> Path:
    criteria = clinc_criteria()
    spec_path = criteria_path("clinc150")
    if not spec_path.exists():
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(
            json.dumps(
                {
                    "criteria_version": "clinc150-v0",
                    "dataset": "clinc150",
                    "instructions": "Which intent matches `text`? Use out_of_scope if none do.",
                    "criteria": criteria,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    spec = load_criteria(spec_path)
    if fixture:
        raw = json.loads(fixture.read_text(encoding="utf-8"))
    else:
        ds = try_load_hf_first(
            [
                ("clinc_oos", {"name": "plus"}),
                ("clinc/clinc_oos", {"name": "plus"}),
            ]
        )
        if ds is None:
            raise FileNotFoundError("CLINC150 not available: pass --fixture")
        raw = {}
        for split in ds:
            rows = []
            for i, row in enumerate(ds[split]):
                intent = row.get("intent") or row.get("label")
                if intent in (None, "oos", "OOS"):
                    intent = "out_of_scope"
                rows.append({"id": f"clinc_{split}_{i}", "text": row["text"], "label": str(intent)})
            raw[split] = rows

    def to_ex(row: dict[str, Any], split: str) -> ChoiceTrainingExample:
        gold = str(row["label"])
        if gold not in spec["criteria"]:
            gold = "out_of_scope"
        return ChoiceTrainingExample(
            id=str(row.get("id")),
            type="choice",
            format_version=FORMAT_VERSION,
            criteria_version=str(spec["criteria_version"]),
            state={"text": row["text"]},
            question=ChoiceQuestion(
                type="choice",
                instructions=spec["instructions"],
                criteria=spec["criteria"],
            ),
            gold=gold,
            metadata=ExampleMetadata(domain="clinc150", group_id=str(row.get("id")), source="clinc_oos", split=split),  # type: ignore[arg-type]
        )

    train = [to_ex(r, "train") for r in raw.get("train", [])]
    val_src = raw.get("validation") or raw.get("val") or []
    # Split validation evenly into validation/calibration by index hash
    val, calib = [], []
    for i, r in enumerate(val_src):
        (calib if i % 2 else val).append(to_ex(r, "calibration" if i % 2 else "validation"))
    test = [to_ex(r, "test") for r in raw.get("test", [])]
    return freeze_and_write(
        dataset="clinc150",
        primitive="choice",
        criteria_file=spec_path,
        converter="jev.data.clinc150",
        license_name="CC BY 3.0",
        source="clinc/clinc_oos",
        split_rule="official train+test; official validation split even/odd into val/calib",
        examples_by_split={"train": train, "validation": val, "calibration": calib, "test": test},
        out_dir=out_dir,
        notes="Includes genuine out_of_scope gold rows.",
    )
