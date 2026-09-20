# v0 metrics

Empty cells are unused, not invented. Latency and cost were not measured in this run.

Frozen comparison split: **synthetic test**, n=12, grouped by `synthetic:{i//4}`. Same JSONL for every backend below.

CLI sequence on this machine: `jev data-convert synthetic` → `jev train-head --encoder hf` (Qwen2.5-0.5B, batch 1, CUDA) → `jev calibrate` on the **calibration** split → `jev evaluate --out`. Hashing is an extra CPU trainer proof, not the Qwen substitute.

Temperature for the hashing head was fit on the synthetic **calibration** split (n=48), T=0.25. The frozen-Qwen head used T=5.0 fit on the same 48-row calibration split. Zero-shot is uncalibrated (T=1.0).

| Model | Accuracy | NLL | Brier | ECE | Coverage at 1% error | Shuffled-context acc | Median latency | P95 latency | Cost / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fake overlap (CPU) | 0.083 | 1.386 | 0.750 | 0.167 | 0.00 | 0.083 |  |  |  |
| Majority prior (train counts) | 0.500 | 2.096 | 0.825 | 0.383 | 0.00 | 0.500 |  |  |  |
| TF-IDF + logistic (state text) | 1.000 | 0.299 | 0.093 | 0.256 | 1.00 | 0.500 |  |  |  |
| JSON-LM tiny causal (JSON continuations) | 0.583 | 1.807 | 0.797 | 0.421 | 0.00 | 0.583 |  |  |  |
| Tiny causal logprob (dev) | 0.583 | 2.114 | 0.771 | 0.411 | 0.08 | 0.500 |  |  |  |
| Hashing option head (CPU trainer proof) | 1.000 | 1.7e-6 | 6.5e-11 | 1.7e-6 | 1.00 | 0.417 |  |  |  |
| Frozen Qwen2.5-0.5B option head (CLI `--encoder hf`, full train, batch 1) | 0.083 | 1.436 | 0.777 | 0.214 | 0.00 | 0.083 |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob (cached, fp16, CUDA) | 0.750 | 0.606 | 0.353 | 0.259 | 0.25 | 0.167 |  |  |  |

Sources: `reports/v0/metrics/eval-hf-head-synthetic-test.json`, `eval-hf-logprob-synthetic-test.json`, `eval-hashing-test.json`, `eval-fake-test.json`, `eval-tiny-logprob-test.json`, `eval-majority-synthetic-test.json`, `eval-tfidf-synthetic-test.json`, `eval-json-llm-synthetic-test.json`.

Notes:

- Hashing 100% on this synthetic task is the trainer proof (`docs/10-datasets.md`: if you cannot hit ~98%, the code is wrong). Shuffled-context 0.42 vs 1.00 shows the head uses state, not only option priors.
- The frozen Qwen head was trained with the public CLI (`train-head --encoder hf --batch-size 1 --epochs 12`, early stop on validation, not calibration). Train acc 0.27 = shuffled 0.27. Frozen test acc 0.083 = shuffled 0.083. Measured underfit on 4 GiB, **not** a hidden larger run and **not** an OOM (nvidia-smi stayed inside 4096 MiB). Zero-shot on the same test is the stronger Qwen result.
- Zero-shot shuffled-context 0.167 is near chance for 4-way menus. The causal scorer is reading the state.
- TF-IDF also hits 1.00 here because the gold color is written in the ticket text. Shuffled-context 0.50 shows it is reading state.
- Do not mix Colab T4 rows into this table.

## Other converted tests (not the research comparison)

Hashing head trained **only** on synthetic, then evaluated on fixture test slices. These numbers show transfer failure, not dataset quality.

| Dataset (fixture test) | Hashing acc | Fake acc |
|---|---:|---:|
| BANKING77 n=8 | 0.00 | 0.50 |
| SST-5 n=5 | 0.00 | 1.00 |
| BoolQ n=3 | 0.67 | 0.67 |
| CLINC150 n=3 | 0.00 | 0.00 |
| Wikispeedia n=2 | 0.00 | 1.00 |

Full official splits converted on this host (manifests under `reports/v0/manifests/`): BANKING77 `mteb/banking77` train 8067 / test 3076; SST-5; BoolQ `google/boolq` (original JSONL HTTP 403); CLINC150; Wikispeedia SNAP 159134 train. Those large tests were **not** scored with Qwen here: 77-way / 151-way menus plus long passages exceed a responsible 4 GiB first pass. That is a Colab T4 job with a **new** hardware note.

## CUDA gates

- Tiny LM naïve vs cached on CUDA: pass (`pytest -m "cuda and not hf"`).
- Hashing head train on CUDA: pass.
- Qwen2.5-0.5B naïve vs cached continuation scores: pass (`tests/test_hf_cuda.py`). No OOM.
- CLI `train-head --encoder hf` on full synthetic train: completed, no OOM.

Logged GPU run: `reports/v0/metrics/gpu-run.json`. CLI HF train/eval: `eval-hf-head-synthetic-test.json`, `verification-commands.txt`.

## Reproduce on Colab (different pin)

This table is the 1650 / Qwen2.5-0.5B measurement. Do not paste T4 rows
here. Hosted-GPU jobs (BANKING77, Wikispeedia, a 3B encoder) use
[docs/11-colab.md](../../docs/11-colab.md) and
[notebooks/jev_colab_hosted_gpu.ipynb](../../notebooks/jev_colab_hosted_gpu.ipynb).
Success criteria and execution status: [colab.md](colab.md).
