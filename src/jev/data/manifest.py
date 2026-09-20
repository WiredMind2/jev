"""Dataset manifests with content hashes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jev.canonical import sha256_file, sha256_json
from jev.schema import FORMAT_VERSION

try:
    from importlib.resources import files as pkg_files
except ImportError:  # pragma: no cover
    pkg_files = None  # type: ignore


def load_criteria(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def criteria_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return sha256_json(payload)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def criteria_path(name: str) -> Path:
    return repo_root() / "assets" / "criteria" / f"{name}.json"


def build_manifest(
    *,
    dataset: str,
    primitive: str,
    criteria_file: Path,
    converter: str,
    license_name: str,
    source: str,
    split_rule: str,
    split_files: dict[str, Path],
    notes: str = "",
    n_groups: dict[str, int] | None = None,
) -> dict[str, Any]:
    criteria = load_criteria(criteria_file)
    splits = {}
    for name, path in split_files.items():
        n = 0
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                n = sum(1 for line in handle if line.strip())
        entry = {
            "path": str(path.as_posix()),
            "n": n,
            "sha256": sha256_file(path) if path.exists() else "0" * 64,
        }
        if n_groups and name in n_groups:
            entry["n_groups"] = int(n_groups[name])
        splits[name] = entry
    return {
        "dataset": dataset,
        "primitive": primitive,
        "format_version": FORMAT_VERSION,
        "criteria_version": str(criteria.get("criteria_version", "unknown")),
        "criteria_path": str(criteria_file.as_posix()),
        "criteria_sha256": criteria_sha256(criteria_file),
        "converter": converter,
        "converter_version": "0.1.0",
        "license": license_name,
        "source": source,
        "split_rule": split_rule,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": notes,
        "splits": splits,
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    from jev.canonical import canonical_dumps

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_dumps(manifest) + "\n", encoding="utf-8")
