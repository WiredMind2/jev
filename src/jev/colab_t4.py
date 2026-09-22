"""File Colab T4 eval JSON into reports/colab-t4 without touching reports/v0."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

DATASETS: tuple[tuple[str, str, bool, bool], ...] = (
    ("BANKING77", "banking77", True, False),
    ("SST-5", "sst5", False, True),
    ("BoolQ", "boolq", False, False),
    ("Wikispeedia", "wikispeedia", False, False),
    ("CLINC150", "clinc150", False, False),
)
HEAD_ROW = "Frozen Qwen2.5-0.5B option head"
ZS_ROW = "Zero-shot Qwen2.5-0.5B logprob"
V0_FORBIDDEN = "reports/v0"


def _fmt(value: float, digits: int) -> str:
    return f"{float(value):.{digits}f}"


def gpu_label(hardware_text: str, live: dict[str, Any] | None) -> str:
    name = ""
    if live:
        gpu = live.get("live_gpu") or {}
        if isinstance(gpu, dict):
            name = str(gpu.get("name") or "")
    if not name:
        for line in hardware_text.splitlines():
            if "Tesla" in line or "T4" in line or "RTX" in line or "name:" in line.lower():
                name = line
                break
    blob = f"{name} {hardware_text}"
    if re.search(r"\bT4\b", blob) or "Tesla T4" in blob:
        return "Colab T4"
    if "4050" in blob:
        return "RTX 4050, not Colab T4"
    if "1650" in blob:
        return "GTX 1650, not Colab T4"
    return name.strip() or "unknown GPU"


def discover_eval_json(src: Path, dataset: str, kind: str) -> Path | None:
    names = (
        src / "metrics" / f"eval-{dataset}-{kind}.json",
        src / f"eval-{dataset}-{kind}.json",
        src / "reports" / "metrics" / f"eval-{dataset}-{kind}.json",
        src / "reports" / f"eval-{dataset}-{kind}.json",
    )
    for path in names:
        if path.is_file():
            return path
    return None


def _row_cells(payload: dict[str, Any], *, coverage: bool, mae: bool) -> str:
    cells = [
        _fmt(payload["accuracy"], 3),
        _fmt(payload["nll"], 3),
        _fmt(payload["brier"], 3),
        _fmt(payload["ece"], 3),
    ]
    if coverage:
        cells.append(_fmt(payload.get("coverage_at_1pct") or 0.0, 2))
    if mae:
        cells.append(_fmt(payload.get("mae") or 0.0, 3))
    cells.append(_fmt(payload["shuffled_accuracy"], 3))
    return " | ".join(cells)


def _replace_model_row(section: str, label: str, cells: str) -> str:
    pattern = rf"^(\| {re.escape(label)} \|)[^\n]*$"
    return re.sub(pattern, rf"\1 {cells} |", section, count=1, flags=re.M)


def render_metrics_md(
    template: str,
    *,
    rows: dict[str, dict[str, dict[str, Any]]],
    gpu: str,
) -> str:
    parts = re.split(r"(?=^## )", template, flags=re.M)
    out: list[str] = []
    for part in parts:
        heading = part.splitlines()[0] if part.strip() else ""
        matched = None
        for title, key, coverage, mae in DATASETS:
            if heading.startswith(f"## {title}"):
                matched = (key, coverage, mae)
                break
        if matched is None:
            if heading.startswith("## 3B VRAM probe") and rows.get("probe3b"):
                outcome = str(rows["probe3b"].get("outcome") or "executed")
                part = re.sub(
                    r"(\| Qwen2\.5-3B `--max-steps` 2 batch 1 \|)[^\n]*",
                    rf"\1 {outcome} |",
                    part,
                    count=1,
                )
            out.append(part)
            continue
        key, coverage, mae = matched
        block = rows.get(key) or {}
        for kind, label in (("hf-head", HEAD_ROW), ("hf-logprob", ZS_ROW)):
            payload = block.get(kind)
            if payload:
                part = _replace_model_row(part, label, _row_cells(payload, coverage=coverage, mae=mae))
        if gpu == "Colab T4":
            part = re.sub(r"GPU: RTX 4050[^\n]*", "GPU: Colab T4.", part)
        out.append(part)
    text = "".join(out)
    if gpu == "Colab T4" and "GPU pin for remaining Qwen rows: **Colab T4**" in text:
        text = text.replace(
            "Qwen rows are filled only after a live CUDA run; empty cells are unused,\n"
            "not invented.",
            "Qwen rows below are from a live Colab T4 session. Empty cells remain unused,\n"
            "not invented.",
        )
    return text


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_colab_t4(src: Path, dest: Path, *, v0_metrics: Path | None = None) -> dict[str, Any]:
    src = src.resolve()
    dest = dest.resolve()
    if V0_FORBIDDEN in dest.as_posix():
        raise ValueError("refusing to ingest into reports/v0")
    dest.mkdir(parents=True, exist_ok=True)
    metrics_dir = dest / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (dest / "model-cards").mkdir(parents=True, exist_ok=True)

    hardware_src = None
    for candidate in (src / "hardware.md", src / "reports" / "hardware.md"):
        if candidate.is_file():
            hardware_src = candidate
            break
    live = None
    for candidate in (
        src / "colab-live-hardware.json",
        src / "reports" / "colab-live-hardware.json",
    ):
        if candidate.is_file():
            live = _load_json(candidate)
            shutil.copy2(candidate, dest / "colab-live-hardware.json")
            break
    hardware_text = hardware_src.read_text(encoding="utf-8") if hardware_src else ""
    gpu = gpu_label(hardware_text, live)
    if hardware_src is not None:
        shutil.copy2(hardware_src, dest / "hardware.md")

    filled: dict[str, list[str]] = {}
    row_payloads: dict[str, Any] = {}
    for _title, key, _coverage, _mae in DATASETS:
        for kind in ("hf-head", "hf-logprob"):
            found = discover_eval_json(src, key, kind)
            if found is None:
                continue
            target = metrics_dir / f"eval-{key}-{kind}.json"
            shutil.copy2(found, target)
            payload = _load_json(target)
            row_payloads.setdefault(key, {})[kind] = payload
            filled.setdefault(key, []).append(kind)

    probe_card = None
    for candidate in (
        src / "model-cards" / "qwen25-3b-probe.md",
        src / "reports" / "model-cards" / "qwen25-3b-probe.md",
    ):
        if candidate.is_file():
            probe_card = candidate
            break
    if probe_card is not None:
        shutil.copy2(probe_card, dest / "model-cards" / "qwen25-3b-probe.md")
        text = probe_card.read_text(encoding="utf-8")
        outcome = "executed"
        for line in text.splitlines():
            if line.startswith("**Outcome:**"):
                outcome = line.split(":", 1)[1].strip()
                break
        row_payloads["probe3b"] = {"outcome": outcome}

    template_path = dest / "metrics.md"
    if not template_path.is_file():
        raise FileNotFoundError(f"missing metrics template: {template_path}")
    before_v0 = v0_metrics.read_bytes() if v0_metrics and v0_metrics.is_file() else None
    template_path.write_text(
        render_metrics_md(template_path.read_text(encoding="utf-8"), rows=row_payloads, gpu=gpu),
        encoding="utf-8",
    )
    if before_v0 is not None and v0_metrics is not None and v0_metrics.read_bytes() != before_v0:
        raise RuntimeError("ingest mutated reports/v0/metrics.md")

    return {"gpu": gpu, "filled": filled, "dest": str(dest)}


def comparison_status(root: Path) -> dict[str, Any]:
    """Inventory required T4 comparison files under a Drive reports dump."""
    root = root.resolve()
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for _title, key, _coverage, _mae in DATASETS:
        for kind in ("hf-head", "hf-logprob"):
            found = discover_eval_json(root, key, kind)
            ok = found is not None
            rows.append(
                {
                    "dataset": key,
                    "kind": kind,
                    "ok": ok,
                    "path": str(found) if found else None,
                }
            )
            if not ok:
                missing.append(f"{key}/{kind}")
    hardware = None
    for candidate in (root / "hardware.md", root / "reports" / "hardware.md"):
        if candidate.is_file():
            hardware = str(candidate)
            break
    live = None
    for candidate in (
        root / "colab-live-hardware.json",
        root / "reports" / "colab-live-hardware.json",
    ):
        if candidate.is_file():
            live = str(candidate)
            break
    probe = None
    for candidate in (
        root / "model-cards" / "qwen25-3b-probe.md",
        root / "reports" / "model-cards" / "qwen25-3b-probe.md",
    ):
        if candidate.is_file():
            probe = str(candidate)
            break
    if hardware is None:
        missing.append("hardware.md")
    return {
        "complete": not missing,
        "missing": missing,
        "rows": rows,
        "hardware": hardware,
        "live_gpu": live,
        "probe3b": probe,
        "root": str(root),
    }
