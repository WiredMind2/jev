"""Shared conversion helpers and HF/fixture loaders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.canonical import sha256_json
from jev.data.io import write_example_jsonl
from jev.data.manifest import build_manifest, criteria_sha256, write_manifest
from jev.data.splits import assert_no_group_overlap, group_key
from jev.schema import parse_training_example


def load_records_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def try_load_hf(path_or_name: str, **kwargs: Any) -> Any | None:
    try:
        from datasets import load_dataset
    except ImportError:
        return None
    try:
        return load_dataset(path_or_name, **kwargs)
    except Exception:
        return None


def freeze_and_write(
    *,
    dataset: str,
    primitive: str,
    criteria_file: Path,
    converter: str,
    license_name: str,
    source: str,
    split_rule: str,
    examples_by_split: dict[str, list[Any]],
    out_dir: Path,
    notes: str = "",
) -> Path:
    jsonl_dir = out_dir / "jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, Path] = {}
    parsed = {
        k: [parse_training_example(e.model_dump(mode="json") if hasattr(e, "model_dump") else e) for e in v]
        for k, v in examples_by_split.items()
    }
    assert_no_group_overlap(parsed)
    digest = criteria_sha256(criteria_file)
    version = json.loads(criteria_file.read_text(encoding="utf-8")).get("criteria_version", "unknown")
    stamped = f"{version}:{digest[:12]}"
    n_groups: dict[str, int] = {}
    for name, rows in parsed.items():
        updated = []
        for row in rows:
            meta_update = {"split": name}
            if getattr(row, "type", None) == "choice":
                gold = row.gold
                if gold not in row.question.criteria:
                    raise ValueError(f"{name} gold {gold!r} missing from criteria")
                meta_update["candidate_sha256"] = sha256_json(list(row.question.criteria))
            meta = row.metadata.model_copy(update=meta_update)
            updated.append(row.model_copy(update={"criteria_version": stamped, "metadata": meta}))
        parsed[name] = updated
        n_groups[name] = len({group_key(r) for r in updated})
        dest = jsonl_dir / f"{name}.jsonl"
        write_example_jsonl(dest, updated)
        split_files[name] = dest
    manifest = build_manifest(
        dataset=dataset,
        primitive=primitive,
        criteria_file=criteria_file,
        converter=converter,
        license_name=license_name,
        source=source,
        split_rule=split_rule,
        split_files=split_files,
        notes=notes,
        n_groups=n_groups,
    )
    dest_m = out_dir / "manifest.json"
    write_manifest(dest_m, manifest)
    return dest_m


def attach_criteria_hash(example: Any, criteria_obj: dict[str, Any]) -> Any:
    version = str(criteria_obj.get("criteria_version", "unknown"))
    digest = sha256_json(criteria_obj)
    return example.model_copy(update={"criteria_version": f"{version}:{digest[:12]}"})
