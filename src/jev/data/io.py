"""JSONL IO for training examples."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from jev.canonical import canonical_dumps
from jev.schema import parse_training_example


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_dumps(row) if not isinstance(row, str) else row)
            if not isinstance(row, str):
                # already dumped
                pass
            handle.write("\n")


def write_example_jsonl(path: Path, examples: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for ex in examples:
            payload = ex.model_dump(mode="json") if hasattr(ex, "model_dump") else ex
            handle.write(canonical_dumps(payload))
            handle.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_examples(path: Path) -> list[Any]:
    return [parse_training_example(row) for row in read_jsonl(path)]
