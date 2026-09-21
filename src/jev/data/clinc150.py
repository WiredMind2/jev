"""CLINC150 Choice + out_of_scope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.data.clinc_intents import OOS_KEY, clinc_criteria as frozen_clinc_criteria
from jev.data.convert import freeze_and_write, try_load_hf_first
from jev.data.manifest import criteria_path, load_criteria
from jev.schema import FORMAT_VERSION, ChoiceQuestion, ChoiceTrainingExample, ExampleMetadata

OOS_ALIASES = {"oos", "ood", "OOS", "OOD", OOS_KEY}


def decode_clinc_intent(raw: Any, names: list[str] | None) -> str:
    """Map HF ClassLabel integers and `oos` onto frozen criteria keys."""
    value: Any = raw
    if names is not None and not isinstance(value, str):
        value = names[int(value)]
    elif names is not None and isinstance(value, str) and value.isdigit():
        idx = int(value)
        if 0 <= idx < len(names):
            value = names[idx]
    label = str(value)
    if label in OOS_ALIASES or label.lower() in {"oos", "ood"}:
        return OOS_KEY
    return label


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
                ("clinc/clinc_oos", {"name": "plus"}),
                ("clinc_oos", {"name": "plus"}),
            ]
        )
        if ds is None:
            raise FileNotFoundError("CLINC150 not available: pass --fixture")
        intent_feat = ds["train"].features.get("intent") or ds["train"].features.get("label")
        names = list(getattr(intent_feat, "names", None) or []) or None
        raw = {}
        for split in ds:
            rows = []
            for i, row in enumerate(ds[split]):
                intent = row.get("intent") if row.get("intent") is not None else row.get("label")
                rows.append(
                    {
                        "id": f"clinc_{split}_{i}",
                        "text": row["text"],
                        "label": decode_clinc_intent(intent, names),
                    }
                )
            raw[split] = rows

    def to_ex(row: dict[str, Any], split: str) -> ChoiceTrainingExample:
        gold = decode_clinc_intent(row["label"], None)
        if gold not in spec["criteria"]:
            raise ValueError(f"unknown clinc150 label {gold!r} (id={row.get('id')})")
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
