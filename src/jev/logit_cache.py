"""Append-only logit sidecar so Colab evaluate/calibrate can resume after a disconnect."""

from __future__ import annotations

import json
from pathlib import Path


def cache_path_for(out: Path) -> Path:
    return out.with_suffix(".partial.jsonl")


def load_rows(path: Path | None) -> dict[tuple[str, str], tuple[list[float], int]]:
    rows: dict[tuple[str, str], tuple[list[float], int]] = {}
    if path is None or not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        stage = str(obj["stage"])
        example_id = str(obj["id"])
        logits = [float(x) for x in obj["logits"]]
        gold = int(obj["gold"])
        rows[(stage, example_id)] = (logits, gold)
    return rows


def append_row(
    path: Path,
    *,
    stage: str,
    example_id: str,
    logits: list[float],
    gold: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "id": example_id, "logits": logits, "gold": gold}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
