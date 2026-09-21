"""Stderr progress bars with ETA. Colab is often not a TTY, so we always use \\r."""

from __future__ import annotations

import math
import os
import sys
import time
from typing import TextIO


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _bar(frac: float, width: int = 20) -> str:
    frac = min(1.0, max(0.0, frac))
    filled = int(round(frac * width))
    if filled >= width:
        return "[" + "=" * width + "]"
    if filled <= 0:
        return "[" + ">" + " " * (width - 1) + "]" if frac > 0 else "[" + " " * width + "]"
    return "[" + "=" * (filled - 1) + ">" + " " * (width - filled) + "]"


class TaskProgress:
    """One bar: `desc  12/300  [===>    ]  4.0%  0.50 it/s  ETA 9m36s`."""

    def __init__(
        self,
        desc: str,
        total: int,
        *,
        stream: TextIO | None = None,
        min_interval: float = 0.25,
        unit: str = "it",
    ) -> None:
        self.desc = desc
        self.total = max(0, int(total))
        self.unit = unit
        self._stream = stream if stream is not None else sys.stderr
        self._min_interval = min_interval
        self._n = 0
        self._t0 = time.monotonic()
        self._last_draw = -1.0
        self._last_len = 0
        self._closed = False
        self._enabled = os.environ.get("JEV_PROGRESS", "1") != "0"

    def set(self, n: int, *, total: int | None = None, force: bool = False) -> None:
        if total is not None:
            self.total = max(0, int(total))
        self._n = max(0, int(n))
        self._draw(force=force or self._n >= self.total > 0 or self._n == 0)

    def update(self, k: int = 1) -> None:
        self.set(self._n + k)

    def close(self) -> None:
        if self._closed:
            return
        self._draw(force=True)
        if self._enabled:
            self._stream.write("\n")
            self._stream.flush()
        self._closed = True

    def __enter__(self) -> TaskProgress:
        self.set(self._n, force=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def render(self, *, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self._t0)
        frac = (self._n / self.total) if self.total else 0.0
        pct = 100.0 * frac
        rate = self._n / elapsed if elapsed > 0 and self._n else 0.0
        remain = max(0, self.total - self._n)
        eta = remain / rate if rate > 0 else None
        return (
            f"{self.desc}  {self._n}/{self.total}  {_bar(frac)}  {pct:5.1f}%  "
            f"{rate:.2f} {self.unit}/s  ETA {format_duration(eta)}"
        )

    def _draw(self, *, force: bool = False) -> None:
        if not self._enabled or self._closed:
            return
        now = time.monotonic()
        if not force and self._last_draw >= 0 and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        text = self.render(now=now)
        pad = " " * max(0, self._last_len - len(text))
        self._stream.write("\r" + text + pad)
        self._stream.flush()
        self._last_len = len(text)


class StageBars:
    """Evaluate/calibrate callback: (stage, i, n) -> one bar per stage."""

    def __init__(self, *, stream: TextIO | None = None, unit: str = "ex") -> None:
        self._stream = stream
        self._unit = unit
        self._bars: dict[str, TaskProgress] = {}

    def __call__(self, stage: str, i: int, n: int) -> None:
        bar = self._bars.get(stage)
        if bar is None:
            for old in self._bars.values():
                old.close()
            bar = TaskProgress(stage, n, stream=self._stream, unit=self._unit)
            self._bars[stage] = bar
        bar.set(i)

    def close(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


def step_total(*, n_examples: int, batch_size: int, epochs: int, max_steps: int | None) -> int:
    n_batches = max(1, math.ceil(n_examples / max(1, batch_size)))
    planned = n_batches * max(1, epochs)
    if max_steps is None:
        return planned
    return min(planned, max(1, max_steps))
