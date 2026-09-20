# Model card — frozen Qwen2.5-0.5B option-attention head

`AutoModel` last hidden states, all encoder parameters frozen. Trainable: low-rank option query + MLP logit.

- **Checkpoint:** encoder `Qwen/Qwen2.5-0.5B` fp16; head saved without encoder weights (`runs/synthetic-hf-head.pt`).
- **Train budget this run:** 64 synthetic rows, batch size 1, max 12 epochs, early stop on validation (not calibration), max_state_tokens 192, 4 GiB cap.
- **This run:** train acc 0.27 = shuffled 0.27 (no state use). Frozen test acc 0.333 = shuffled 0.333. T=5.0. Wall time ~126 s. No OOM.
- **Interpretation:** underfit. Zero-shot on the same test is the stronger Qwen result. A larger train set or unfrozen last layer is future work, not claimed here.
- **Evidence:** `reports/v0/metrics/eval-hf-head-synthetic-test.json`, `gpu-run.json`.
