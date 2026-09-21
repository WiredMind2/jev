import io

from jev.progress import TaskProgress, format_duration, step_total


def test_format_duration() -> None:
    assert format_duration(None) == "?"
    assert format_duration(9) == "9s"
    assert format_duration(75) == "1m15s"
    assert format_duration(3661) == "1h01m01s"


def test_step_total_respects_max_steps() -> None:
    assert step_total(n_examples=100, batch_size=4, epochs=12, max_steps=400) == 300
    assert step_total(n_examples=10_000, batch_size=4, epochs=12, max_steps=400) == 400
    assert step_total(n_examples=8, batch_size=16, epochs=2, max_steps=None) == 2


def test_task_progress_bar_and_eta(monkeypatch) -> None:
    clock = {"t": 100.0}
    monkeypatch.setattr("jev.progress.time.monotonic", lambda: clock["t"])
    buf = io.StringIO()
    bar = TaskProgress("eval", 10, stream=buf, min_interval=0, unit="ex")
    bar.set(0, force=True)
    clock["t"] = 110.0
    bar.set(5, force=True)
    text = buf.getvalue()
    assert "eval" in text
    assert "5/10" in text
    assert "50.0%" in text
    assert "ETA" in text
    assert "10s" in text or "0m10s" in text
    bar.close()
    assert buf.getvalue().endswith("\n")
