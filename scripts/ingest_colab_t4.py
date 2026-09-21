"""CLI wrapper: python scripts/ingest_colab_t4.py --src Drive/jev-runs --dest reports/colab-t4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jev.colab_t4 import ingest_colab_t4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy Colab Drive eval JSON into reports/colab-t4. Never writes reports/v0."
    )
    parser.add_argument("--src", type=Path, required=True, help="Drive jev-runs or a reports dump")
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("reports/colab-t4"),
        help="Git report directory (must not be reports/v0)",
    )
    parser.add_argument(
        "--v0-metrics",
        type=Path,
        default=Path("reports/v0/metrics.md"),
        help="Checksum guard; ingest aborts if this file changes",
    )
    args = parser.parse_args()
    result = ingest_colab_t4(args.src, args.dest, v0_metrics=args.v0_metrics)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
