"""Canonical JSON for cache keys, hashes, and byte-identical reconversion."""

from __future__ import annotations

import hashlib
import json
from typing import Any

CANONICAL_SEPARATORS = (",", ":")


def canonical_dumps(value: Any) -> str:
    """UTF-8 JSON with sorted keys and no insignificant whitespace."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=CANONICAL_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_dumps_bytes(value: Any) -> bytes:
    return canonical_dumps(value).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_dumps_bytes(value))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
