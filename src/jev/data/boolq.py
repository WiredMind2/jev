"""BoolQ Noul conversion. Group by Wikipedia title (or passage hash if omitted)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jev.canonical import sha256_text
from jev.data.convert import download_file, freeze_and_write, load_records_jsonl, try_load_hf
from jev.data.manifest import criteria_path, load_criteria, repo_root
from jev.data.splits import assign_groups
from jev.schema import (
    FORMAT_VERSION,
    ExampleMetadata,
    NoulCriteria,
    NoulQuestion,
    NoulTrainingExample,
)


def group_key_for_row(row: dict[str, Any], fallback_i: int = 0) -> str:
    title = row.get("title")
    if title:
        return str(title)
    passage = str(row.get("passage") or "")
    if passage:
        return f"passage:{sha256_text(passage[:800])[:16]}"
    return f"page-{fallback_i}"


BOOLQ_ORIGINAL = {
    "train": "https://storage.googleapis.com/boolq/train.jsonl",
    "validation": "https://storage.googleapis.com/boolq/dev.jsonl",
}


def _load_original_boolq_jsonl(path: Path, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append(
                {
                    "id": f"boolq_{split}_{i}",
                    "title": str(row.get("title") or group_key_for_row(row, i)),
                    "passage": row["passage"],
                    "question": row["question"],
                    "answer": bool(row["answer"]),
                }
            )
    return rows


def try_original_boolq() -> dict[str, list[dict[str, Any]]] | None:
    """Prefer the original release (has Wikipedia titles) over google/boolq."""
    cache = repo_root() / "data" / "raw" / "boolq"
    files = {"train": cache / "train.jsonl", "validation": cache / "dev.jsonl"}
    for split, dest in files.items():
        if dest.exists() and dest.stat().st_size > 0:
            continue
        if not download_file(BOOLQ_ORIGINAL[split], dest):
            return None
    return {split: _load_original_boolq_jsonl(path, split) for split, path in files.items()}


def convert_boolq(out_dir: Path, fixture: Path | None = None) -> Path:
    spec = load_criteria(criteria_path("boolq"))
    source_name = "google/boolq"
    if fixture:
        if fixture.suffix == ".json":
            raw = json.loads(fixture.read_text(encoding="utf-8"))
        else:
            grouped: dict[str, list[dict[str, Any]]] = {}
            for row in load_records_jsonl(fixture):
                grouped.setdefault(row.get("split", "train"), []).append(row)
            raw = grouped
        source_name = "fixture"
    else:
        raw = try_original_boolq()
        if raw is not None:
            source_name = "storage.googleapis.com/boolq (titles present)"
        else:
            ds = try_load_hf("google/boolq")
            if ds is None:
                raise FileNotFoundError("BoolQ not available: pass --fixture")
            raw = {}
            for split in ds:
                raw[split] = [
                    {
                        "id": f"boolq_{split}_{i}",
                        "title": group_key_for_row(row, i),
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
            metadata=ExampleMetadata(
                domain="boolq",
                group_id=title.lower(),
                source=source_name,
                split=split,  # type: ignore[arg-type]
            ),
        )

    groups = [str(r.get("title") or r["id"]).lower() for r in train_kept]
    assigned = assign_groups(groups, seed=2, fractions=(0.8, 0.1, 0.1))
    splits: dict[str, list[NoulTrainingExample]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
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
        source=source_name,
        split_rule=(
            "official validation/dev frozen as test; remaining groups 80/10/10; "
            "group_id is Wikipedia title when present, else a hash of the passage "
            "(google/boolq omits title; original jsonl includes it)"
        ),
        examples_by_split=splits,
        out_dir=out_dir,
        notes=(
            "Original BoolQ jsonl includes Wikipedia titles used as group_id."
            if "titles present" in source_name
            else "google/boolq has no title column; passage-hash grouping is the leakage control."
        ),
    )
