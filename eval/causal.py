"""Causal evaluation driver: runs the interchange + steering experiments.

Wrapper around src/interchange.py. Outputs results/causal/causal_results.json.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS  # noqa: E402


def main() -> None:
    (RESULTS / "causal").mkdir(parents=True, exist_ok=True)
    runpy.run_path(str(ROOT / "src" / "interchange.py"), run_name="__main__")


if __name__ == "__main__":
    main()
