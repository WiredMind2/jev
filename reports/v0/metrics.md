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

## Fixture frozen tests (Qwen 0.5B on the 1650, plus hashing CPU)

These are **per-dataset** heads and scorers on the expanded converter fixtures, not transfer from synthetic. Official full splits (BANKING77 3076 test, CLINC150 5500, Wikispeedia SNAP, …) were converted on this host; scoring those large tests with Qwen is the Colab T4 job. No T4 numbers are in this file.

| Dataset | n | Model | Acc | NLL | Brier | ECE | Cov@1% | Shuffled |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| BANKING77 | 154 | Fake overlap | 0.792 | 3.121 | 0.925 | 0.746 | 0.182 | 0.013 |
| BANKING77 | 154 | Hashing head (T=1.00, calib n=77) | 0.013 | 14.485 | 1.425 | 0.568 | 0.000 | 0.019 |
| BANKING77 | 154 | Zero-shot Qwen logprob (T=1.0) | 0.786 | 1.278 | 0.437 | 0.353 | 0.377 | 0.006 |
| BANKING77 | 154 | Frozen Qwen head | — | — | — | — | — | — |
| SST-5 | 15 | Fake overlap | 1.000 | 0.842 | 0.421 | 0.567 | 1.000 | 0.400 |
| SST-5 | 15 | Hashing head (T=0.25, calib n=5) | 0.667 | 1.134 | 0.576 | 0.321 | 0.467 | 0.267 |
| SST-5 | 15 | Zero-shot Qwen logprob (T=1.0) | 0.200 | 1.354 | 0.682 | 0.328 | 0.200 | 0.200 |
| SST-5 | 15 | Frozen Qwen head (T=5.0, calib n=5) | 0.200 | 2.150 | 0.907 | 0.182 | 0.200 | 0.200 |
| BoolQ | 8 | Fake overlap | 0.500 | 0.614 | 0.423 | 0.122 | 0.375 | 0.500 |
| BoolQ | 8 | Hashing head (T=5.0, calib n=3) | 0.500 | 0.694 | 0.501 | 0.014 | 0.000 | 0.500 |
| BoolQ | 8 | Zero-shot Qwen logprob (T=1.0) | 0.500 | 0.677 | 0.484 | 0.165 | 0.125 | 0.500 |
| BoolQ | 8 | Frozen Qwen head (T=2.0, calib n=3) | 0.500 | 0.739 | 0.544 | 0.149 | 0.000 | 0.500 |
| CLINC150 | 26 | Fake overlap | 0.923 | 4.043 | 0.969 | 0.904 | 0.923 | 0.115 |
| CLINC150 | 26 | Hashing head (T=0.75, calib n=13) | 0.769 | 0.824 | 0.366 | 0.141 | 0.538 | 0.154 |
| CLINC150 | 26 | Zero-shot Qwen logprob (T=1.0) | 0.538 | 1.731 | 0.666 | 0.211 | 0.385 | 0.038 |
| Wikispeedia | 4 | Fake overlap | 0.750 | 1.882 | 0.617 | 0.417 | 0.500 | 0.250 |
| Wikispeedia | 4 | Hashing head (T=0.50, calib n=5) | 0.500 | 1.947 | 0.869 | 0.483 | 0.000 | 0.500 |
| Wikispeedia | 4 | Zero-shot Qwen logprob (T=1.0) | 0.500 | 1.547 | 0.590 | 0.264 | 0.250 | 0.250 |

BANKING77 frozen Qwen head: **not OOM**. Training on the 154-row fixture (77-way menus, batch 1, fp16) held **1198 MiB / 4096 MiB** at 100% GPU for ~80 minutes and did not finish 12 epochs; the process was stopped. Evidence: `eval-hf-head-banking77-fixture-test-timeout.json`. Official BANKING77 / CLINC150 / Wikispeedia frozen-head jobs stay on Colab T4 ([colab.md](colab.md)).

BANKING77 zero-shot 0.786 vs shuffled 0.006 (77-way chance ≈ 0.013) shows the cached continuation scorer is reading the utterance, not the menu prior. CLINC150 zero-shot 0.538 vs shuffled 0.038 is the same control.

Sources: `eval-hf-logprob-{banking77,sst5,boolq,clinc150,wikispeedia}-fixture-test.json`, `eval-hf-head-{sst5,boolq}-fixture-test.json`, `eval-hashing-*-test.json`, `eval-fake-*-test.json`, `calibrate-hashing-*.json`, `calibrate-hf-head-{sst5,boolq}.json`.

## Converted official splits (not scored with Qwen here)

Manifests under `reports/v0/manifests/`: BANKING77 `mteb/banking77` train 8067 / test 3076; SST-5 `SetFit/sst5`; BoolQ `google/boolq` (original JSONL HTTP 403); CLINC150; Wikispeedia SNAP 159134 train. Hub probe: `metrics/hf-download-probe.json` (`hf_token_present: false`, `banking77_source: mteb/banking77`, BoolQ original 403).

## CUDA gates

- Tiny LM naïve vs cached on CUDA: pass (`pytest -m "cuda and not hf"`).
- Hashing head train on CUDA: pass.
- Qwen2.5-0.5B naïve vs cached continuation scores: pass (`tests/test_hf_cuda.py`). No OOM.
- CLI `train-head --encoder hf` on full synthetic train: completed, no OOM.
- Fixture SST-5 and BoolQ `train-head --encoder hf`: completed, peak ≈ 1138 MiB, no OOM.
- Fixture BANKING77 / CLINC150 / Wikispeedia / BoolQ / SST-5 cached zero-shot: completed on this GPU. BANKING77 n=154 max allocated 1821 MiB. No OOM.

Logged GPU run: `reports/v0/metrics/gpu-run.json`. CLI HF train/eval: `eval-hf-head-synthetic-test.json`, `verification-commands.txt`.

## Reproduce on Colab (different pin)

This table is the 1650 / Qwen2.5-0.5B measurement. Do not paste T4 rows
here. Hosted-GPU jobs (BANKING77, Wikispeedia, a 3B encoder) use
[docs/11-colab.md](../../docs/11-colab.md) and
[notebooks/jev_colab_hosted_gpu.ipynb](../../notebooks/jev_colab_hosted_gpu.ipynb).
Success criteria and execution status: [colab.md](colab.md).
