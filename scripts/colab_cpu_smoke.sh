#!/usr/bin/env bash
# CPU proof that the Colab notebook's CLI sequence exists.
# Does not use a Colab T4 and must not overwrite reports/v0/metrics.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

OUT="${JEV_RUNS:-$ROOT/.jev-colab-smoke}"
DATA="$OUT/data"
RUNS="$OUT/runs"
REPORTS="$OUT/reports"
N="${SYNTHETIC_N:-128}"

mkdir -p "$DATA" "$RUNS" "$REPORTS"

echo "== python =="
"$PYTHON" --version
echo "== jev hardware (repo pin, not a live Colab GPU) =="
"$PYTHON" -m jev hardware

echo "== convert synthetic =="
"$PYTHON" -m jev data-convert synthetic --out "$DATA" --n "$N"

TRAIN="$DATA/synthetic/jsonl/train.jsonl"
VAL="$DATA/synthetic/jsonl/validation.jsonl"
CALIB="$DATA/synthetic/jsonl/calibration.jsonl"
TEST="$DATA/synthetic/jsonl/test.jsonl"
CKPT="$RUNS/synthetic-hashing-head.pt"
CAL_JSON="$RUNS/synthetic-hashing-temperature.json"
EVAL_JSON="$REPORTS/eval-synthetic-hashing-cpu-smoke.json"

echo "== train-head hashing (CPU trainer proof; not the Qwen comparison) =="
"$PYTHON" -m jev train-head "$TRAIN" \
  --val-jsonl "$VAL" \
  --encoder hashing \
  --out "$CKPT" \
  --epochs 12 \
  --batch-size 16

echo "== calibrate (calibration split only) =="
"$PYTHON" -m jev calibrate "$CALIB" \
  --checkpoint "$CKPT" \
  --out "$CAL_JSON"

echo "== evaluate =="
"$PYTHON" -m jev evaluate "$TEST" \
  --backend option-head \
  --checkpoint "$CKPT" \
  --out "$EVAL_JSON"

echo "== wrote $EVAL_JSON =="
cat "$EVAL_JSON"
echo
echo "CPU smoke ok. This is not a Colab T4 run and not a rewrite of reports/v0/metrics.md."
