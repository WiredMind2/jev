"""Wikispeedia next-click conversion with back-click stack replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jev.data.convert import freeze_and_write
from jev.data.manifest import criteria_path, load_criteria, repo_root
from jev.data.splits import assign_groups
from jev.schema import FORMAT_VERSION, ChoiceQuestion, ChoiceTrainingExample, ExampleMetadata

MAX_OPTIONS = 255


def replay_path(tokens: list[str]) -> list[tuple[str, str]]:
    """Yield (current, next_forward) after applying '<' as back."""
    stack: list[str] = []
    pairs: list[tuple[str, str]] = []
    for tok in tokens:
        if tok == "<":
            if stack:
                stack.pop()
            continue
        if stack:
            pairs.append((stack[-1], tok))
        stack.append(tok)
    return pairs


def sample_menu(gold: str, outgoing: list[str], *, cap: int = MAX_OPTIONS, seed_key: str) -> list[str]:
    uniq = []
    seen = set()
    for x in outgoing:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    if gold not in uniq:
        uniq.append(gold)
    if len(uniq) <= cap:
        return uniq
    others = [x for x in uniq if x != gold]
    ranked = sorted(others, key=lambda x: hashlib.sha256(f"{seed_key}:{x}".encode()).hexdigest())
    return [gold] + ranked[: cap - 1]


def parse_fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_wikispeedia(out_dir: Path, fixture: Path | None = None) -> Path:
    spec_path = criteria_path("wikispeedia")
    if not spec_path.exists():
        spec_path.write_text(
            json.dumps(
                {
                    "criteria_version": "wikispeedia-v0",
                    "instructions": (
                        "Which outgoing article should a player click next "
                        "to reach `target` from `current`?"
                    ),
                    "title_template": "Outgoing link: {title}",
                    "max_options": 255,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    spec = load_criteria(spec_path)
    if fixture is None:
        default = repo_root() / "tests" / "fixtures" / "data" / "wikispeedia.json"
        if not default.exists():
            raise FileNotFoundError("Wikispeedia SNAP dump is not vendored; pass --fixture JSON")
        fixture = default
    payload = parse_fixture(fixture)
    articles: dict[str, str] = payload.get("articles", {})
    graph: dict[str, list[str]] = payload["graph"]
    paths: list[dict[str, Any]] = payload["paths"]
    rows: list[ChoiceTrainingExample] = []
    skipped = 0
    for p_i, path in enumerate(paths):
        target = path["target"]
        tokens = path["tokens"]
        for step_i, (current, nxt) in enumerate(replay_path(tokens)):
            outgoing = list(graph.get(current, []))
            menu = sample_menu(nxt, outgoing, cap=int(spec.get("max_options", 255)), seed_key=f"{target}:{current}")
            if len(menu) < 2:
                skipped += 1
                continue
            if nxt not in menu:
                skipped += 1
                continue
            template = spec.get("title_template", "Outgoing link: {title}")
            criteria = {title: template.format(title=title) for title in menu}
            current_text = articles.get(current, current)
            rows.append(
                ChoiceTrainingExample(
                    id=f"wiki_{p_i}_{step_i}",
                    type="choice",
                    format_version=FORMAT_VERSION,
                    criteria_version=str(spec["criteria_version"]),
                    state={"target": target, "current": current, "current_text": current_text},
                    question=ChoiceQuestion(type="choice", instructions=spec["instructions"], criteria=criteria),
                    gold=nxt,
                    metadata=ExampleMetadata(
                        domain="wikispeedia",
                        group_id=target,
                        source="SNAP-wikispeedia",
                        candidate_sha256=hashlib.sha256("|".join(sorted(menu)).encode()).hexdigest(),
                    ),
                )
            )
    groups = [r.metadata.group_id or r.id for r in rows]
    assigned = assign_groups(groups, seed=3, fractions=(0.7, 0.1, 0.1))
    splits: dict[str, list[ChoiceTrainingExample]] = {"train": [], "validation": [], "calibration": [], "test": []}
    for row in rows:
        g = row.metadata.group_id or row.id
        test_bit = int(hashlib.sha256(f"wiki-test:{g}".encode()).hexdigest()[:2], 16) < 26
        dest = "test" if test_bit else assigned[g]
        splits[dest].append(row.model_copy(update={"metadata": row.metadata.model_copy(update={"split": dest})}))
    return freeze_and_write(
        dataset="wikispeedia",
        primitive="choice",
        criteria_file=spec_path,
        converter="jev.data.wikispeedia",
        license_name="SNAP + Wikipedia CC BY-SA",
        source="snap.stanford.edu/data/wikispeedia",
        split_rule="split by target article; replay '<' via navigation stack; cap 255 keeping gold",
        examples_by_split=splits,
        out_dir=out_dir,
        notes=f"skipped_rows={skipped}",
    )
