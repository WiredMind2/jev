"""Grouped split helpers. Calibration is never used for early stopping."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any, TypeVar

T = TypeVar("T")

SPLIT_NAMES = ("train", "validation", "calibration", "test")


def group_key(example: Any) -> str:
    meta = example.metadata if hasattr(example, "metadata") else example.get("metadata", {})
    gid = meta.group_id if hasattr(meta, "group_id") else meta.get("group_id")
    if gid:
        return str(gid)
    ident = example.id if hasattr(example, "id") else example["id"]
    return f"id:{ident}"


def groups_of(examples: Sequence[T]) -> dict[str, list[T]]:
    grouped: dict[str, list[T]] = defaultdict(list)
    for ex in examples:
        grouped[group_key(ex)].append(ex)
    return dict(grouped)


def assert_no_group_overlap(splits: dict[str, Sequence[T]]) -> None:
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for ex in rows:
            g = group_key(ex)
            prev = seen.get(g)
            if prev is not None and prev != name:
                raise ValueError(f"group {g!r} appears in both {prev} and {name}")
            seen[g] = name


def assign_groups(
    group_ids: Sequence[str],
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, str]:
    """Assign unique groups to train/validation/calibration. Test is separate."""
    import hashlib

    labeled: dict[str, str] = {}
    names = ("train", "validation", "calibration")
    cuts = (fractions[0], fractions[0] + fractions[1], 1.0)
    for gid in sorted(set(group_ids)):
        digest = hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()
        u = int(digest[:8], 16) / 0xFFFFFFFF
        if u < cuts[0]:
            labeled[gid] = names[0]
        elif u < cuts[1]:
            labeled[gid] = names[1]
        else:
            labeled[gid] = names[2]
    return labeled


def stratified_indices(
    labels: Sequence[str], *, seed: int, fractions: tuple[float, float, float]
) -> dict[str, list[int]]:
    """Row-level stratified carve of train into train/val/calib. Not for grouped tasks."""
    import hashlib
    from collections import defaultdict

    by_label: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_label[str(lab)].append(i)
    out: dict[str, list[int]] = {"train": [], "validation": [], "calibration": []}
    cuts = (fractions[0], fractions[0] + fractions[1])
    for lab, idxs in by_label.items():
        ordered = sorted(idxs, key=lambda i: hashlib.sha256(f"{seed}:{lab}:{i}".encode()).hexdigest())
        n = len(ordered)
        n_train = int(n * cuts[0])
        n_val = int(n * (cuts[1] - cuts[0]))
        out["train"].extend(ordered[:n_train])
        out["validation"].extend(ordered[n_train : n_train + n_val])
        out["calibration"].extend(ordered[n_train + n_val :])
        # Keep at least one train row when n>=1
        if n >= 1 and not out["train"] and ordered:
            moved = ordered[0]
            for key in ("calibration", "validation"):
                if moved in out[key]:
                    out[key].remove(moved)
                    out["train"].append(moved)
                    break
    for key in out:
        out[key].sort()
    return out
