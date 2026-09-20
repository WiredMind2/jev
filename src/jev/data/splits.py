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
    unique = sorted(set(group_ids))
    for gid in unique:
        digest = hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()
        u = int(digest[:8], 16) / 0xFFFFFFFF
        if u < cuts[0]:
            labeled[gid] = names[0]
        elif u < cuts[1]:
            labeled[gid] = names[1]
        else:
            labeled[gid] = names[2]
    if len(unique) >= 3:
        present = set(labeled.values())
        for needed in names:
            if needed in present:
                continue
            donor = max(names, key=lambda n: sum(1 for v in labeled.values() if v == n))
            steal = next(g for g in unique if labeled[g] == donor)
            labeled[steal] = needed
            present.add(needed)
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
    frac_train, frac_val, _frac_cal = fractions
    for lab, idxs in by_label.items():
        ordered = sorted(idxs, key=lambda i: hashlib.sha256(f"{seed}:{lab}:{i}".encode()).hexdigest())
        n = len(ordered)
        if n >= 3:
            n_val = max(1, int(n * frac_val))
            n_cal = max(1, int(n * (1.0 - frac_train - frac_val)))
            n_train = n - n_val - n_cal
            if n_train < 1:
                n_train = 1
                leftover = n - 1
                n_val = max(1, leftover // 2)
                n_cal = leftover - n_val
        elif n == 2:
            n_train, n_val, n_cal = 1, 0, 1
        else:
            n_train, n_val, n_cal = n, 0, 0
        out["train"].extend(ordered[:n_train])
        out["validation"].extend(ordered[n_train : n_train + n_val])
        out["calibration"].extend(ordered[n_train + n_val : n_train + n_val + n_cal])
    for key in out:
        out[key].sort()
    return out
