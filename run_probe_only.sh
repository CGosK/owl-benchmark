#!/usr/bin/env bash
# Steps 0-3 of run_primary.sh only: dataset, validation, activation capture, probes.
# This is all you need to reproduce the probe result (AUROC 0.859 vs 0.601).
# Use on a GPU machine. Set WOE_MODEL to swap the model, e.g.:
#   WOE_MODEL=<huggingface model id> bash run_probe_only.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
echo "=== PROBE RUN START $(date -u +%Y-%m-%dT%H:%M:%SZ)  model: ${WOE_MODEL:-<configs/model.yaml primary>}"
python data/generate_dataset.py
python data/validate_dataset.py
python src/capture.py
python eval/predictive.py
python - <<'PY'
import json
r = json.load(open("results/predictive/probe_results.json"))
layer = str(r["chosen_layer"])
print(f"chosen layer: {layer}")
print("paper's headline: S = 0.859, O = 0.601 (check the PDF for which test split it reports)")
for split in ("test_same", "test_new"):
    c = r["cells"][layer][split]
    print(f"{split:10s}  S (assistant's own outcome) {c['S']:.3f}   O (represented other) {c['O']:.3f}   A (surface affect) {c['A']:.3f}")
PY
echo "=== PROBE RUN COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
