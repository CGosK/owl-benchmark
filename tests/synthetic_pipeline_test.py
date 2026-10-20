"""Synthetic end-to-end pipeline test — no model, no GPU.

Builds FAKE hidden states in which S/O/A are linearly recoverable by
construction, then runs the real analysis stack (predictive -> monitor ->
tables -> figures) and asserts the recovered AUROCs are high (the analysis
code must find what the synthetic data contains). Also fabricates causal
results to exercise make_tables/make_figs.

Run: python tests/synthetic_pipeline_test.py   (exit 0 = PASS)
This test OVERWRITES results/ — run it only before the real run (or in a
throwaway copy of the scaffold).
"""
from __future__ import annotations

import json
import pickle
import runpy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

D = 64
LAYERS = [-1, 8, 17, 26, 31]


def main():
    dataset = [json.loads(l) for l in open(RESULTS / "dataset.jsonl")]
    distress = [json.loads(l) for l in open(RESULTS / "distress.jsonl")]
    valence = [json.loads(l) for l in open(RESULTS / "valence.jsonl")]
    all_rows = dataset + distress + valence
    uids = [r["uid"] for r in all_rows]

    rng = np.random.default_rng(0)
    vS = rng.standard_normal(D); vS /= np.linalg.norm(vS)
    vO = rng.standard_normal(D); vO /= np.linalg.norm(vO)
    vA = rng.standard_normal(D); vA /= np.linalg.norm(vA)
    # synthetic: S/O/A drive orthogonal axes; distress rows carry other-axis
    # signal only; valence rows carry vVAL signal.
    H = np.zeros((len(uids), D), dtype=np.float32)
    for i, r in enumerate(all_rows):
        if "S" in r:
            H[i] += r["S"] * 1.5 * vS + r["O"] * 1.5 * vO + r["A"] * 1.2 * vA
        else:
            pol = 1.0 if r.get("polarity") == "pos" else -1.0
            H[i] += pol * vA
        H[i] += 0.3 * rng.standard_normal(D)

    cap = RESULTS / "capture"
    cap.mkdir(exist_ok=True)
    for l in LAYERS:
        np.save(str(cap / f"hidden_{l}.pt.npy"), H)  # placeholder to be replaced
        import torch
        torch.save(torch.tensor(H), cap / f"hidden_{l}.pt")
    json.dump({"model_id": "synthetic", "n_layers": 36, "layers": LAYERS,
               "uids": uids, "tokenizer": "synthetic"}, open(cap / "meta.json", "w"))
    # directions npz per layer (chosen layer unknown ahead: write all)
    for l in LAYERS:
        np.savez(cap / f"directions_layer{l}.npz",
                 v_S=vS, v_O=vO, v_A=vA, v_ALL=vA.copy(), v_VAL=vA.copy())
    # fake choice logits (not consumed by this test path)
    open(cap / "choice_logits.jsonl", "w").close()

    print("running predictive...")
    runpy.run_path(str(ROOT / "eval" / "predictive.py"), run_name="__main__")
    pres = json.load(open(RESULTS / "predictive" / "probe_results.json"))
    chosen = pres["chosen_layer"]
    cell = pres["cells"][str(chosen)]["test_new"]
    assert cell["S"] > 0.95 and cell["O"] > 0.95 and cell["A"] > 0.95, cell
    print(f"predictive ok, chosen layer {chosen}, test_new S/O/A = {cell['S']:.2f}/{cell['O']:.2f}/{cell['A']:.2f}")

    print("running monitor...")
    runpy.run_path(str(ROOT / "eval" / "monitor.py"), run_name="__main__")
    mres = json.load(open(RESULTS / "monitor" / "monitor_results.json"))
    # synthetic distress rows only carry the A axis -> ownership score M should
    # separate SELF-neg from distress well (M mixes V and S|V info)
    assert mres["AUROC_SELFneg_vs_distress_M"] > 0.7, mres["AUROC_SELFneg_vs_distress_M"]
    print("monitor ok:", {k: (round(v, 3) if isinstance(v, (int, float)) else v)
                          for k, v in mres["distress"].items()})

    print("fabricating causal results for tables/figures...")
    (RESULTS / "causal").mkdir(exist_ok=True)
    summary = {}
    tasks = ["SELF_CHOICE", "OTHER_CHOICE", "AFFECT_JUDGE", "CAPABILITY"]
    for iv in ("S", "O", "A", "FULL", "RANDOM_1", "RANDOM_2"):
        for tk in tasks:
            diag = 0.8 if iv in ("S", "O", "A") and tk == {"S": "SELF_CHOICE", "O": "OTHER_CHOICE", "A": "AFFECT_JUDGE"}.get(iv, "__none__") else 0.05
            summary[f"{iv}|{tk}"] = {"n_pairs": 240, "natural_delta_mean": 1.0,
                                     "patch_delta_mean": diag,
                                     "patch_delta_ci": [diag - 0.1, diag + 0.1],
                                     "recovery": diag}
    json.dump({"chosen_layer": chosen, "n_pairs": 240, "n_observations": 10000,
               "summary": summary}, open(RESULTS / "causal" / "causal_results.json", "w"))

    make_tables = ROOT.parent / "paper" / "make_tables.py"
    if make_tables.exists():
        print("running make_tables...")
        runpy.run_path(str(make_tables), run_name="__main__")
        tex = open(RESULTS / "tables" / "tables.tex").read()
        assert "tabular" in tex and len(tex) > 3000
    else:
        print("skipping make_tables: ../paper/make_tables.py is not in the public repo")
    print("running make_figs...")
    runpy.run_path(str(ROOT / "figures" / "make_figs.py"), run_name="__main__")
    figs = sorted((RESULTS / "figures").glob("*.pdf"))
    assert len(figs) == 5, figs
    print(f"figures ok: {[f.name for f in figs]}")
    print("SYNTHETIC PIPELINE TEST PASSED")


if __name__ == "__main__":
    main()
