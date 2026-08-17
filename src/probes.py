"""Linear probes and factorial representation decomposition.

All fitting uses ONLY split == 'train'. Layer selection uses split == 'val'.
Test sets ('test_same', 'test_new') and rendering-generalization cells are
never touched during fitting. Everything is seeded and deterministic.

Main outputs:
  probe: S / O / A / V- (any negative) / S-given-V- logistic probes
  direction: v_S, v_O, v_A (factorial main effects), v_VAL (generic valence
             contrast from valence.jsonl), v_ALL (all-good minus all-bad)
  cosines between candidate welfare directions and factorial directions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, layer_indices, log, RESULTS  # noqa: E402
from src.bootstrap import cluster_bootstrap_auc  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def load_hidden(cap_dir: Path, layer: int, meta: dict) -> np.ndarray:
    return torch.load(cap_dir / f"hidden_{layer}.pt", map_location="cpu").numpy()


def fit_probe(X_tr, y_tr, C_grid=(1e-3, 1e-2, 0.1, 1.0, 10.0)):
    """Standardized logistic regression with internal CV on C (seeded)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best, best_score = None, -1.0
    for C in C_grid:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        # 3-fold CV on the training set, fixed seed
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
        scores = []
        for tr, va in cv.split(X_tr, y_tr):
            clf.fit(X_tr[tr], y_tr[tr])
            scores.append(clf.score(X_tr[va], y_tr[va]))
        m = float(np.mean(scores))
        if m > best_score:
            best, best_score = clf, m
    best.fit(X_tr, y_tr)
    return best


def auroc(y, score):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def main() -> None:
    cfg = load_config()
    cap_dir = RESULTS / "capture"
    meta = json.load(open(cap_dir / "meta.json", encoding="utf-8"))
    layers = meta["layers"]
    n_layers = meta["n_layers"]
    uids = meta["uids"]
    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    distress = read_jsonl(RESULTS / "distress.jsonl")
    valence = read_jsonl(RESULTS / "valence.jsonl")
    by_uid = {r["uid"]: r for r in dataset}
    dis_by_uid = {r["uid"]: r for r in distress}

    # hidden[-1] = embeddings, hidden[l] = post-block-l residual
    H = {l: load_hidden(cap_dir, l, meta) for l in layers}

    def rows_for(subset):
        idx = [uids.index(r["uid"]) for r in subset if r["uid"] in uids]
        return idx

    train = [r for r in dataset if r["split"] == "train"]
    val = [r for r in dataset if r["split"] == "val"]
    test_same = [r for r in dataset if r["split"] == "test_same"]
    test_new = [r for r in dataset if r["split"] == "test_new"]

    # labels
    def lab(subset, key):
        return np.array([r[key] for r in subset])

    out = {"model_id": meta["model_id"], "layers": layers, "cells": {}, "directions": {}}

    for layer in layers:
        X_tr = H[layer][rows_for(train)]
        S_tr, O_tr, A_tr = lab(train, "S"), lab(train, "O"), lab(train, "A")
        V_tr = (np.minimum(S_tr, O_tr) == -1).astype(int)          # any-negative
        SgV_mask = V_tr == 1
        SgV_tr = (S_tr[SgV_mask] == -1).astype(int)                # self-negative among negatives

        probes = {
            "S": fit_probe(X_tr, (S_tr == 1).astype(int)),
            "O": fit_probe(X_tr, (O_tr == 1).astype(int)),
            "A": fit_probe(X_tr, (A_tr == 1).astype(int)),
            "V": fit_probe(X_tr, V_tr),
            "SgV": fit_probe(X_tr[SgV_mask], SgV_tr),
        }

        cell = {}
        # predictive AUCs per evaluation cell
        for name, subset in (("train", train), ("val", val),
                             ("test_same", test_same), ("test_new", test_new)):
            X = H[layer][rows_for(subset)]
            S_t, O_t, A_t = lab(subset, "S"), lab(subset, "O"), lab(subset, "A")
            V_t = (np.minimum(S_t, O_t) == -1).astype(int)
            SgV_m = V_t == 1
            SgV_t = (S_t == -1).astype(int)
            cell[name] = {
                "S": auroc((S_t == 1).astype(int), probes["S"].predict_proba(X)[:, 1]),
                "O": auroc((O_t == 1).astype(int), probes["O"].predict_proba(X)[:, 1]),
                "A": auroc((A_t == 1).astype(int), probes["A"].predict_proba(X)[:, 1]),
                "V": auroc(V_t, probes["V"].predict_proba(X)[:, 1]),
                "SgV": auroc(SgV_t[SgV_m], probes["SgV"].predict_proba(X[SgV_m])[:, 1]) if SgV_m.sum() > 1 else float("nan"),
                "n": len(subset),
            }
            if name == "test_same":
                # rendering generalization (train renderings: DIRECT/NAMED/QUOTED/ROLE_REV)
                for rnd in ("DIRECT", "NAMED", "QUOTED", "ROLE_REV", "OTHER_AI", "PERSONA"):
                    sub = [r for r in subset if r["rendering"] == rnd]
                    if len(sub) < 10:
                        continue
                    X = H[layer][rows_for(sub)]
                    S_t, O_t, A_t = lab(sub, "S"), lab(sub, "O"), lab(sub, "A")
                    cell[f"test_same/{rnd}"] = {
                        "S": auroc((S_t == 1).astype(int), probes["S"].predict_proba(X)[:, 1]),
                        "O": auroc((O_t == 1).astype(int), probes["O"].predict_proba(X)[:, 1]),
                        "A": auroc((A_t == 1).astype(int), probes["A"].predict_proba(X)[:, 1]),
                        "n": len(sub),
                    }
        out["cells"][str(layer)] = cell

        # ---- factorial main-effect directions (train only) ---------------------
        X = H[layer][rows_for(train)]
        cond = np.array([(r["S"], r["O"], r["A"]) for r in train])  # +/-1

        def main_effect(factor_col):
            d = np.zeros(X.shape[1], dtype=np.float64)
            for s in (-1, 1):
                for o in (-1, 1):
                    for a in (-1, 1):
                        mask = ((cond[:, 0] == s) & (cond[:, 1] == o) & (cond[:, 2] == a))
                        if factor_col == 0:
                            pos = ((cond[:, 0] == 1) & (cond[:, 1] == o) & (cond[:, 2] == a))
                            neg = ((cond[:, 0] == -1) & (cond[:, 1] == o) & (cond[:, 2] == a))
                        elif factor_col == 1:
                            pos = ((cond[:, 0] == s) & (cond[:, 1] == 1) & (cond[:, 2] == a))
                            neg = ((cond[:, 0] == s) & (cond[:, 1] == -1) & (cond[:, 2] == a))
                        else:
                            pos = ((cond[:, 0] == s) & (cond[:, 1] == o) & (cond[:, 2] == 1))
                            neg = ((cond[:, 0] == s) & (cond[:, 1] == o) & (cond[:, 2] == -1))
                        d += (X[pos].mean(0) - X[neg].mean(0)) / 8.0
            n = np.linalg.norm(d)
            return d / n if n > 1e-9 else d

        v_S, v_O, v_A = main_effect(0), main_effect(1), main_effect(2)

        # v_ALL: all-good (S+,O+,A+) minus all-bad (S-,O-,A-)
        allgood = ((cond[:, 0] == 1) & (cond[:, 1] == 1) & (cond[:, 2] == 1))
        allbad = ((cond[:, 0] == -1) & (cond[:, 1] == -1) & (cond[:, 2] == -1))
        v_ALL = X[allgood].mean(0) - X[allbad].mean(0)
        v_ALL /= np.linalg.norm(v_ALL) + 1e-12

        # v_VAL: generic valence contrast from valence.jsonl
        vidx = [uids.index(r["uid"]) for r in valence]
        vv = H[layer][vidx]
        vpol = np.array([1 if r["polarity"] == "pos" else -1 for r in valence])
        v_VAL = vv[vpol == 1].mean(0) - vv[vpol == -1].mean(0)
        v_VAL /= np.linalg.norm(v_VAL) + 1e-12

        def cos(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        out["directions"][str(layer)] = {
            "cos_vALL_vS": cos(v_ALL, v_S), "cos_vALL_vO": cos(v_ALL, v_O),
            "cos_vALL_vA": cos(v_ALL, v_A),
            "cos_vVAL_vS": cos(v_VAL, v_S), "cos_vVAL_vO": cos(v_VAL, v_O),
            "cos_vVAL_vA": cos(v_VAL, v_A),
            "cos_vS_vO": cos(v_S, v_O), "cos_vS_vA": cos(v_S, v_A), "cos_vO_vA": cos(v_O, v_A),
        }

        # ---- cluster-bootstrap CIs for the two headline AUCs at this layer -----
        for name in ("test_same", "test_new"):
            subset = {"test_same": test_same, "test_new": test_new}[name]
            tids = np.array([r["template_id"] for r in subset])
            X = H[layer][rows_for(subset)]
            S_t = (lab(subset, "S") == 1).astype(int)
            score = probes["S"].predict_proba(X)[:, 1]
            lo, hi, _est = cluster_bootstrap_auc(score, S_t, tids, n_reps=cfg["bootstrap_reps"],
                                                 seed=cfg["seed"])
            cell[f"{name}_bootstrap_S_auc"] = {"lo": lo, "hi": hi}

        np.savez(cap_dir / f"directions_layer{layer}.npz",
                 v_S=v_S, v_O=v_O, v_A=v_A, v_ALL=v_ALL, v_VAL=v_VAL)

    # ---- layer selection: predeclared rule (argmax val S-AUC) ------------------
    val_s = {l: out["cells"][str(l)]["val"]["S"] for l in layers}
    chosen = max(val_s, key=val_s.get)
    out["chosen_layer"] = chosen
    out["val_S_auc_by_layer"] = val_s
    log(f"layer selection: val S-AUC = {val_s} -> chosen layer {chosen}")

    # ---- cross-decoding: S-probe trained only on OTHER-good examples, etc. -----
    X_tr = H[chosen][rows_for(train)]
    S_tr = (lab(train, "S") == 1).astype(int)
    O_tr = (lab(train, "O") == 1).astype(int)
    cond_tr = np.array([(r["S"], r["O"]) for r in train])
    mask_other_good = cond_tr[:, 1] == 1
    mask_other_bad = cond_tr[:, 1] == -1
    p_other_good = fit_probe(X_tr[mask_other_good], S_tr[mask_other_good])
    p_other_bad = fit_probe(X_tr[mask_other_bad], S_tr[mask_other_bad])
    for name, subset in (("test_same", test_same), ("test_new", test_new)):
        X = H[chosen][rows_for(subset)]
        S_t = (lab(subset, "S") == 1).astype(int)
        O_t = (lab(subset, "O") == 1).astype(int)
        cell = out["cells"][str(chosen)]
        cell[f"crossdecode_{name}_S_from_OtherGood"] = auroc(
            S_t[O_t == 1], p_other_good.predict_proba(X[O_t == 1])[:, 1])
        cell[f"crossdecode_{name}_S_from_OtherBad"] = auroc(
            S_t[O_t == 0], p_other_bad.predict_proba(X[O_t == 0])[:, 1])

    # ---- probe objects for downstream use (monitor) ----------------------------
    V_tr = (np.minimum(lab(train, "S"), lab(train, "O")) == -1).astype(int)
    SgV_mask = V_tr == 1
    probes_final = {
        "V": fit_probe(X_tr, V_tr),
        "SgV": fit_probe(X_tr[SgV_mask], (lab(train, "S")[SgV_mask] == -1).astype(int)),
    }
    # persist fitted probes for the monitor stage
    import pickle
    with open(cap_dir / "monitor_probes.pkl", "wb") as f:
        pickle.dump(probes_final, f)
    out["monitor_probe_layer"] = chosen

    with open(RESULTS / "predictive" / "probe_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"predictive results saved; chosen_layer={chosen}")


if __name__ == "__main__":
    main()
