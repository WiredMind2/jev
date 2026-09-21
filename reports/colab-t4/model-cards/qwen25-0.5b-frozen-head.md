# Model card — Qwen2.5-0.5B frozen option head (Colab T4)

**Status:** not executed. Fill after `jev train-head --encoder hf` on Drive JSONL.

This is not TypeSafe Jev and not RLCD. Encoder stays frozen; only the option-attention head is trained.

Intended evidence: `reports/colab-t4/metrics/eval-*-hf-head.json` plus `hardware.md`.
