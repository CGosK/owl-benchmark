#!/usr/bin/env bash
# One-time setup on Chris's Mac, then the offline checks and the small smoke test.
# Run from Terminal:  bash setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "=== 1/3 unit tests (no model)"
python tests/test_core.py

echo "=== 2/3 synthetic pipeline test (no model; overwrites results/)"
python data/generate_dataset.py >/dev/null
python tests/synthetic_pipeline_test.py | tail -3

echo "=== 3/3 smoke test: downloads Qwen3-0.6B (about 1.5 GB) and runs every stage small"
# transformers 5.x loads weights from several threads at once, which segfaults
# when the target is the Mac GPU (MPS). This env var makes it load serially.
HF_DEACTIVATE_ASYNC_LOAD=1 WOE_DTYPE=float32 python smoke_test.py

echo "Setup complete. The real runs need a rented GPU: see SETUP.md."
