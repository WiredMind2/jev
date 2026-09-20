# Model card — Qwen2.5-0.5B zero-shot logprob

Causal continuation scoring with prefix / KV cache.

- **Checkpoint:** `Qwen/Qwen2.5-0.5B`, fp16, CUDA GTX 1650 4 GiB.
- **Method:** render state+question once; score each option as a continuation; softmax. Cached scores matched naïve re-encoding on this GPU (`tests/test_hf_cuda.py`).
- **Reductions:** sum (default), mean, PMI.
- **This run (frozen synthetic test n=12):** acc 0.75, NLL 0.606, Brier 0.353, ECE 0.259, shuffled-context 0.167, coverage@1% 0.25. Uncalibrated T=1.0.
- **Fixture frozen tests (same 0.5B, GTX 1650, no OOM):** BANKING77 n=154 acc 0.786 shuffled 0.006; SST-5 n=15 acc 0.200; BoolQ n=8 acc 0.500; CLINC150 n=26 acc 0.538 shuffled 0.038; Wikispeedia n=4 acc 0.500 shuffled 0.250.
- **Not TypeSafe Jev.** Softmaxed LM likelihoods are not correctness probabilities.
- **Evidence:** `reports/v0/metrics/eval-hf-logprob-synthetic-test.json`, `eval-hf-logprob-*-fixture-test.json`, `gpu-run.json`.
