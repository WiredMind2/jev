# Model card — Qwen2.5-3B frozen-head VRAM probe (Colab T4)

**Status:** not executed. Set `RUN_3B_PROBE=1` after 0.5B BANKING77 succeeds.

Short `--max-steps` probe, batch size 1, encoder frozen. Record fit / OOM /
skipped. Do not jump to 7B/8B fp16 on free T4. Do not overwrite the 1650 pin.
