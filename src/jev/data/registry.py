"""Named dataset converters. Prefer local fixtures; HF is optional."""

from __future__ import annotations

from pathlib import Path

from jev.data.banking77 import convert_banking77
from jev.data.boolq import convert_boolq
from jev.data.clinc150 import convert_clinc150
from jev.data.sst5 import convert_sst5
from jev.data.wikispeedia import convert_wikispeedia


def convert_named(dataset: str, out_dir: Path, fixture: Path | None = None) -> Path:
    mapping = {
        "banking77": convert_banking77,
        "sst5": convert_sst5,
        "boolq": convert_boolq,
        "clinc150": convert_clinc150,
        "wikispeedia": convert_wikispeedia,
    }
    if dataset not in mapping:
        raise ValueError(f"unknown dataset {dataset!r}")
    return mapping[dataset](out_dir, fixture=fixture)
