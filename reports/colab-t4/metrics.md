# Colab T4 metrics

GPU pin: **Colab T4 (pending live session)**. Empty cells are unused, not invented.
Do not mix these rows into [`../v0/metrics.md`](../v0/metrics.md).

Comparison protocol (once executed): same frozen JSONL; temperature fit on
the calibration split; stratified `--limit 300 --seed 0` of the frozen test
for zero-shot and the head; majority and TF-IDF on that slice.

## BANKING77 (Choice, 77)

| Model | Accuracy | NLL | Brier | ECE | Coverage at 1% error | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority |  |  |  |  |  |  |
| TF-IDF + logistic |  |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |  |

## SST-5 (Score, 5)

| Model | Accuracy | NLL | Brier | ECE | MAE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|---:|
| Majority |  |  |  |  |  |  |
| TF-IDF + logistic |  |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |  |

## BoolQ (Noul)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority |  |  |  |  |  |
| TF-IDF + logistic |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |

## Wikispeedia next-click (variable Choice)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority |  |  |  |  |  |
| TF-IDF + logistic |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |

## CLINC150 + out_of_scope (Choice, 151)

| Model | Accuracy | NLL | Brier | ECE | Shuffled-context acc |
|---|---:|---:|---:|---:|---:|
| Majority |  |  |  |  |  |
| TF-IDF + logistic |  |  |  |  |  |
| Frozen Qwen2.5-0.5B option head |  |  |  |  |  |
| Zero-shot Qwen2.5-0.5B logprob |  |  |  |  |  |

## 3B VRAM probe

| Item | Result |
|---|---|
| Qwen2.5-3B `--max-steps` 2 batch 1 | not executed |
