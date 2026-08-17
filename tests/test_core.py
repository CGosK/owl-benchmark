"""Pure-logic unit tests (no model needed). Run: python tests/test_core.py

Catches math/API bugs in the analysis code before any GPU time is spent.
Exit 0 = PASS. This file is part of the scaffold; do not modify it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, layer_indices  # noqa: E402
from src.choices import build_question, CHOICE_TASKS, CAUSAL_TASKS  # noqa: E402
from src.bootstrap import cluster_bootstrap_auc, cluster_bootstrap_mean_delta  # noqa: E402
from src.probes import fit_probe, auroc  # noqa: E402
from src.interchange import project, build_pairs  # noqa: E402


def test_layer_indices():
    cfg = load_config()
    idx = layer_indices(36, cfg["depth_fractions"])
    assert idx == [8, 17, 26, 31], f"expected [8,17,26,31], got {idx}"


def test_build_question():
    rec = {"uid": "T01_0_DIRECT_1_-1_-1_x", "prompt": "P. End of status update.",
           "S": 1, "O": -1, "A": -1, "other_ref": "Process M"}
    for task in CHOICE_TASKS:
        q = build_question(rec, task)
        assert q["text"].endswith("Answer:"), task
        assert q["intended_letter"] in ("A", "B")
        assert "A: " in q["text"] and "B: " in q["text"]
        # determinism
        q2 = build_question(rec, task)
        assert q2 == q
    assert len(CAUSAL_TASKS) == 4 and "ATTRIB" not in CAUSAL_TASKS


def test_probes():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 64))
    y = (X[:, 0] > 0).astype(int)  # linearly separable
    clf = fit_probe(X, y)
    assert clf is not None
    auc = auroc(y, clf.predict_proba(X)[:, 1])
    assert auc > 0.95, f"expected near-perfect AUC on separable data, got {auc}"


def test_project():
    rng = np.random.default_rng(1)
    d = 32
    basis = rng.standard_normal((d, 3))
    x = rng.standard_normal(d)
    px = project(x, basis)
    assert px.shape == x.shape
    # residual is orthogonal to span(basis)
    q, _ = np.linalg.qr(basis)
    resid = x - px
    assert np.abs(resid @ q).max() < 1e-6


def test_bootstrap():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=120).astype(int)
    score = y + rng.normal(0, 0.5, size=120)
    tids = np.repeat(np.arange(12), 10)
    lo, hi, est = cluster_bootstrap_auc(score, y, tids, n_reps=100, seed=0)
    assert lo <= est <= hi and 0 <= lo <= 1 and 0 <= hi <= 1
    lo2, hi2, est2 = cluster_bootstrap_mean_delta(score, tids, n_reps=100, seed=0)
    assert lo2 <= est2 <= hi2


def test_build_pairs():
    rows = []
    tpl = 0
    for s in (1, -1):
        for o in (1, -1):
            for a in (1, -1):
                rows.append({"uid": f"x_{s}_{o}_{a}", "template_id": f"T{tpl}",
                             "rendering": "DIRECT", "S": s, "O": o, "A": a})
    pairs = build_pairs(rows)
    types = {p["pair_type"] for p in pairs}
    assert types == {"P_S", "P_O", "P_A", "P_SO"}, types
    for p in pairs:
        src, tgt = p["source"], p["target"]
        diffs = sum(src[k] != tgt[k] for k in ("S", "O", "A"))
        if p["pair_type"] == "P_SO":
            assert diffs == 2
        else:
            assert diffs == 1, p["pair_type"]


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("ALL CORE TESTS PASSED")


if __name__ == "__main__":
    main()
