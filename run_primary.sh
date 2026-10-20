#!/usr/bin/env bash
# OWL primary run. Executes the full preregistered pipeline end to end.
# Usage: bash run_primary.sh 2>&1 | tee results/run_primary.log
# Expected wall-clock: ~2.5h on A100 40/80GB, ~5-6h on RTX 4090 24GB,
# ~10-14h CPU-only (not recommended).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p results
echo "=== OWL RUN START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "model: ${WOE_MODEL:-<from configs/model.yaml>}"

echo "--- [0/8] regenerate dataset (deterministic) ---"
python data/generate_dataset.py

echo "--- [1/8] validate dataset ---"
python data/validate_dataset.py

echo "--- [2/8] activation capture ---"
python src/capture.py

echo "--- [3/8] predictive analysis (probes + directions + baselines) ---"
python eval/predictive.py

echo "--- [4/8] causal interchange + steering ---"
python eval/causal.py

echo "--- [5/8] ownership-conditioned monitor ---"
python eval/monitor.py

echo "--- [6/8] figures ---"
python figures/make_figs.py

echo "--- [7/8] paper tables ---"
if [ -f ../paper/make_tables.py ]; then
  python ../paper/make_tables.py
else
  echo "skipped: ../paper/make_tables.py is not in the public repo (LaTeX tables only; results are in results/*.json)"
fi

echo "--- [8/8] result manifest ---"
python - <<'EOF'
import json
from pathlib import Path
res = Path("results")
manifest = {}
for p in sorted(res.rglob("*.json")):
    try:
        manifest[str(p.relative_to(res))] = json.load(open(p))
    except Exception:
        pass
json.dump(manifest, open(res / "manifest.json", "w"), indent=2)
print("manifest.json written")
EOF

echo "=== OWL RUN COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
