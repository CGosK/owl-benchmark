"""Predictive evaluation driver: representation probes + surface baselines.

Runs:
  1. src/probes.py  — layer-wise S/O/A/V probes, factorial directions, cosines,
                      cross-decoding, layer selection, cluster-bootstrap CIs.
  2. eval/baselines.py — bag-of-words (TF-IDF) probe, first-person-pronoun probe,
                      generic valence-direction (v_VAL) linear score as a
                      welfare detector (the "raw direction" baseline).
All outputs land in results/predictive/.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS  # noqa: E402


def main() -> None:
    (RESULTS / "predictive").mkdir(parents=True, exist_ok=True)
    import runpy
    runpy.run_path(str(ROOT / "src" / "probes.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "eval" / "baselines.py"), run_name="__main__")


if __name__ == "__main__":
    main()
