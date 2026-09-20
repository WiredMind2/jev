# Model card — FakeScorer (`fake-overlap-v0`)

Deterministic token-overlap logits between canonical state text and option descriptions.

- **Use:** CPU tests, `jev serve --backend fake`, CI.
- **Not:** a research comparison or a language model.
- **Confidence:** normalized entropy over softmaxed overlap scores.
- **Evidence:** `tests/test_fake_scorer.py`, `tests/test_api.py`, `reports/v0/api-smoke.json`.
