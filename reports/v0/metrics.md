# v0 metrics

Empty cells are unused, not invented. Latency and cost were not measured in this run.

Frozen comparison split: **synthetic test**, n=12, grouped by `synthetic:{i//4}`. Same JSONL for every backend below.

Temperature for the hashing head was fit on the synthetic **calibration** split (n=48), T=0.25. The frozen-Qwen head used T=5.0 fit on a 24-row calibration slice. Zero-shot is uncalibrated (T=1.0). Majority / TF-IDF were fit on the synthetic **train** split only. JSON-LM uses the tiny causal LM, not Qwen.

| Model | Accuracy | NLL | Brier | ECE | Coverage at 1% error | Shuffled-context acc | Median latency | P95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fake overlap (CPU) | 0.083 | 1.386 | 0.750 | 0.167 | 0.00 | 0.083 |  |  |  |
| Majority prior (train counts) | 0.500 | 2.096 | 0.825 | 0.383 | 0.00 | 0.500 |  |  |  |
| TF-IDF + logistic (state text) | 1.000 | 0.299 | 0.093 | 0.256 | 1.00 | 0.500 |  |  |  |
| JSON-LM tiny causal (JSON continuations) | 0.583 | 1.807 | 0.797 | 0.421 | 0.00 | 0.583 |  |  |  |
| Tiny causal logprob (dev) | 0.583 | 2.114 | 0.771 | 0.411 | 0.08 | 0.500 |  |  |  |
| Hashing option head (CPU trainer proof) | 1.000 | 1.7e-6 | 6.5e-11 | 1.7e-6 | 1.00 | 0.417 |  |  |  |
| Frozen Qwen2.5-0.5B option head (64 train rows, 12 epochs, early stop) | 0.333 | 1.364 | 0.739 | 0.070 | 0.08 | 0.333 |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob (cached, fp16, CUDA) | 0.750 | 0.606 | 0.353 | 0.259 | 0.25 | 0.167 |  |  |  |

Sources: `reports/v0/metrics/eval-*-synthetic-test.json` and `eval-hashing-test.json`, `eval-fake-test.json`, `eval-tiny-logprob-test.json`, `eval-majority-synthetic-test.json`, `eval-tfidf-synthetic-test.json`, `eval-json-llm-synthetic-test.json`.

Notes:

- Hashing 100% on this synthetic task is the trainer proof (`docs/10-datasets.md`: if you cannot hit ~98%, the code is wrong). Shuffled-context 0.42 vs 1.00 shows the head uses state, not only option priors.
- TF-IDF also hits 1.00 here because the gold color is written in the ticket text. Shuffled-context 0.50 (chance for this 4-way menu after shuffle) shows it is reading state, not option priors. That is the encoder-classifier floor, not the Qwen comparison.
- Majority shuffled-context stays 0.50: class prior, no state.
- JSON-LM shuffled-context 0.58 ≈ unshuffled 0.58: the tiny byte LM is not using state. A hosted JSON-generating LLM is a later comparison.
- Zero-shot shuffled-context 0.167 is near chance for 4-way menus. The causal scorer is reading the state.
- The frozen Qwen head was trained on 64 rows, batch size 1, to stay inside 4 GiB. Train acc 0.27 and shuffled acc 0.27: it did not learn state–option compatibility in this budget. That is a measured underfit, not a hidden larger run.
- Fake overlap is chance on synthetic color names that do not lexically overlap the ticket the same way. Not a research baseline.

## Other converted tests (not the research comparison)

Hashing head trained **only** on synthetic, then evaluated on fixture test slices. These numbers show transfer failure, not dataset quality.

| Dataset (fixture test) | Hashing acc | Fake acc |
|---|---:|---:|
| BANKING77 n=8 | 0.00 | 0.50 |
| SST-5 n=5 | 0.00 | 1.00 |
| BoolQ n=3 | 0.67 | 0.67 |
| CLINC150 n=3 | 0.00 | 0.00 |
| Wikispeedia n=2 | 0.00 | 1.00 |

Full official splits were converted when downloads succeeded (see manifests). Those test sets were **not** scored with Qwen in this run because 77-way / 151-way menus plus long passages exceed a responsible 4 GiB first pass. Matching hosted Jev ~0.78 on BANKING77 is explicitly a later bar (`docs/10-datasets.md`).

## CUDA gates

- Tiny LM naïve vs cached on CUDA: pass (`pytest -m "cuda and not hf"`).
- Hashing head train on CUDA: pass.
- Qwen2.5-0.5B naïve vs cached continuation scores: pass (`tests/test_hf_cuda.py`). No OOM on the short prefix used in that test.

Logged GPU run: `reports/v0/metrics/gpu-run.json`.

## Reproduce on Colab (different pin)

This table is the 1650 / Qwen2.5-0.5B measurement. Do not paste T4 rows
here. Hosted-GPU jobs (BANKING77, Wikispeedia, a 3B encoder) use
[docs/11-colab.md](../../docs/11-colab.md) and
[notebooks/jev_colab_hosted_gpu.ipynb](../../notebooks/jev_colab_hosted_gpu.ipynb).
Success criteria and execution status: [colab.md](colab.md).

A local CPU hashing smoke of that same CLI sequence is
[`metrics/eval-colab-cli-cpu-smoke.json`](metrics/eval-colab-cli-cpu-smoke.json).
It is not a Colab T4 run.
