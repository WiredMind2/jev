"""Synthetic menus that prove the option-attention trainer.

Each state uniquely names the gold option. Shuffled-state accuracy should
fall near chance if the head uses state-option compatibility.
"""

from __future__ import annotations

import hashlib
from typing import Any

from jev.data.splits import assign_groups, group_key
from jev.schema import (
    FORMAT_VERSION,
    ChoiceQuestion,
    ChoiceTrainingExample,
    ExampleMetadata,
)

CRITERIA_VERSION = "synthetic-v0"
OPTIONS = ("crimson", "azure", "ochre", "violet", "emerald", "amber", "indigo", "ivory")


def _pick(seed_key: str, items: tuple[str, ...], k: int) -> list[str]:
    ranked = sorted(items, key=lambda x: hashlib.sha256(f"{seed_key}:{x}".encode()).hexdigest())
    return ranked[:k]


def make_synthetic_choice(n: int = 256, k: int = 4, seed: int = 0) -> list[ChoiceTrainingExample]:
    rows: list[ChoiceTrainingExample] = []
    for i in range(n):
        menu = _pick(f"{seed}:menu:{i}", OPTIONS, k)
        gold = menu[int(hashlib.sha256(f"{seed}:gold:{i}".encode()).hexdigest(), 16) % k]
        others = [c for c in menu if c != gold]
        state = {
            "ticket": f"Policy names the matching bucket as {gold}.",
            "hint": f"Do not pick {', '.join(others)}.",
        }
        criteria = {c: f"The named bucket is {c}." for c in menu}
        rows.append(
            ChoiceTrainingExample(
                id=f"synthetic_{i:04d}",
                type="choice",
                format_version=FORMAT_VERSION,
                criteria_version=CRITERIA_VERSION,
                state=state,
                question=ChoiceQuestion(
                    type="choice",
                    instructions="Which bucket does `ticket` name?",
                    criteria=criteria,
                ),
                gold=gold,
                metadata=ExampleMetadata(
                    domain="synthetic",
                    group_id=f"synthetic:{i // 4}",
                    source="generated",
                ),
            )
        )
    return rows


def split_synthetic(
    rows: list[ChoiceTrainingExample], seed: int = 0
) -> dict[str, list[ChoiceTrainingExample]]:
    groups = [group_key(r) for r in rows]
    assigned = assign_groups(groups, seed=seed, fractions=(0.7, 0.1, 0.1))
    # leftover ~0.1 goes to test by hashing
    splits: dict[str, list[ChoiceTrainingExample]] = {
        "train": [],
        "validation": [],
        "calibration": [],
        "test": [],
    }
    for row in rows:
        g = group_key(row)
        # 10% of groups reserved for test via a second hash bit
        test_bit = int(hashlib.sha256(f"test:{seed}:{g}".encode()).hexdigest()[:2], 16) < 26
        if test_bit:
            dest = "test"
        else:
            dest = assigned[g]
        payload = row.model_copy(update={"metadata": row.metadata.model_copy(update={"split": dest})})
        splits[dest].append(payload)
    return splits


def to_dicts(splits: dict[str, list[Any]]) -> dict[str, list[dict[str, Any]]]:
    return {k: [r.model_dump(mode="json") for r in v] for k, v in splits.items()}
