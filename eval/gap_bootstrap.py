"""Paired cluster-bootstrap CIs for the O probe AUROC and the S - O gap.

src/probes.py only bootstraps the S AUROC. This script works from saved hidden
states (no capture), refits the S and O probes exactly as src/probes.py does
(same fit_probe, train split, seed), scores test_new, and resamples template
ids with the scheme of src/bootstrap.py:cluster_bootstrap_auc. Every quantity
uses the SAME resample in each rep: S and O are paired within a model, and
because all models share results/dataset.jsonl the resample is also paired
across models, which gives a CI for the cross-model difference in gap.

Usage:
  python eval/gap_bootstrap.py results/qwen3-8b results/mistral-7b-instruct-v0.3
Writes results/gap_bootstrap.json and prints markdown tables.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, log, RESULTS  # noqa: E402
from src.probes import auroc, fit_probe, load_hidden, read_jsonl  # noqa: E402


def score_model(run_dir: Path, train, test):
    """Refit S and O probes per layer; return test scores and saved results."""
    cap_dir = run_dir / "capture"
    meta = json.load(open(cap_dir / "meta.json", encoding="utf-8"))
    saved = json.load(open(run_dir / "predictive" / "probe_results.json", encoding="utf-8"))
    pos = {u: i for i, u in enumerate(meta["uids"])}
    tr_idx = [pos[r["uid"]] for r in train]
    te_idx = [pos[r["uid"]] for r in test]
    S_tr = np.array([r["S"] for r in train]) == 1
    O_tr = np.array([r["O"] for r in train]) == 1
    scores = {}
    for layer in meta["layers"]:
        H = load_hidden(cap_dir, layer, meta)
        X_tr, X_te = H[tr_idx], H[te_idx]
        scores[layer] = {
            "S": fit_probe(X_tr, S_tr.astype(int)).predict_proba(X_te)[:, 1],
            "O": fit_probe(X_tr, O_tr.astype(int)).predict_proba(X_te)[:, 1],
        }
        log(f"{meta['model_id']} layer {layer}: refit done")
    return meta, saved, scores


def ci(reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    cfg = load_config()
    run_dirs = [Path(p) for p in sys.argv[1:]] or [RESULTS / "qwen3-8b",
                                                   RESULTS / "mistral-7b-instruct-v0.3"]
    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    train = [r for r in dataset if r["split"] == "train"]
    test = [r for r in dataset if r["split"] == "test_new"]
    y = {"S": (np.array([r["S"] for r in test]) == 1).astype(int),
         "O": (np.array([r["O"] for r in test]) == 1).astype(int)}
    tids = np.array([r["template_id"] for r in test])

    models = {}
    for d in run_dirs:
        meta, saved, scores = score_model(d, train, test)
        models[d.name] = {"meta": meta, "saved": saved, "scores": scores}

    # ---- sanity check: refit reproduces the saved point AUROCs ------------------
    for name, m in models.items():
        for layer, sc in m["scores"].items():
            for t in ("S", "O"):
                new = auroc(y[t], sc[t])
                old = m["saved"]["cells"][str(layer)]["test_new"][t]
                if round(new, 3) != round(old, 3):
                    raise SystemExit(f"refit mismatch {name} layer {layer} {t}: {new} vs {old}")
    log("sanity check passed: refit S/O AUROCs match probe_results.json to 3 decimals")

    # ---- paired cluster bootstrap (same draws as cluster_bootstrap_auc) ---------
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(cfg["seed"])
    clusters = np.unique(tids)
    members = [np.where(tids == c)[0] for c in clusters]
    n = len(clusters)
    reps = {name: {l: {"S": [], "O": []} for l in m["scores"]} for name, m in models.items()}
    skipped = 0
    for _ in range(cfg["bootstrap_reps"]):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([members[c] for c in pick])
        if len(np.unique(y["S"][idx])) < 2 or len(np.unique(y["O"][idx])) < 2:
            skipped += 1
            continue
        for name, m in models.items():
            for layer, sc in m["scores"].items():
                for t in ("S", "O"):
                    reps[name][layer][t].append(roc_auc_score(y[t][idx], sc[t][idx]))
    log(f"bootstrap: {cfg['bootstrap_reps']} reps over {n} templates, {skipped} skipped")

    out = {"n_templates": n, "n_reps": cfg["bootstrap_reps"], "skipped": skipped,
           "seed": cfg["seed"], "models": {}, "cross_model": []}
    for name, m in models.items():
        nl = m["meta"]["n_layers"]
        rows = {}
        for layer, sc in m["scores"].items():
            S_r, O_r = np.array(reps[name][layer]["S"]), np.array(reps[name][layer]["O"])
            S, O = auroc(y["S"], sc["S"]), auroc(y["O"], sc["O"])
            rows[str(layer)] = {
                "depth": (layer + 1) / nl, "S": S, "O": O, "gap": S - O,
                "S_ci": ci(S_r), "O_ci": ci(O_r), "gap_ci": ci(S_r - O_r),
                "p_gap_le_0": float(np.mean(S_r - O_r <= 0)),
                "saved_S_ci": [m["saved"]["cells"][str(layer)]["test_new_bootstrap_S_auc"][k]
                               for k in ("lo", "hi")],
            }
        out["models"][name] = {"model_id": m["meta"]["model_id"],
                               "chosen_layer": m["saved"]["chosen_layer"], "layers": rows}

    # ---- cross-model difference in gap: chosen layers, then matched depths ------
    if len(models) == 2:
        (a, ma), (b, mb) = models.items()
        pairs = [("chosen", ma["saved"]["chosen_layer"], mb["saved"]["chosen_layer"])]
        pairs += [("matched depth", la, lb) for la, lb in zip(ma["meta"]["layers"],
                                                               mb["meta"]["layers"])]
        for label, la, lb in pairs:
            ga = np.array(reps[a][la]["S"]) - np.array(reps[a][la]["O"])
            gb = np.array(reps[b][lb]["S"]) - np.array(reps[b][lb]["O"])
            est = out["models"][a]["layers"][str(la)]["gap"] - out["models"][b]["layers"][str(lb)]["gap"]
            out["cross_model"].append({"pair": label, "layer_a": la, "layer_b": lb,
                                       "diff": est, "ci": ci(ga - gb),
                                       "p_diff_le_0": float(np.mean(ga - gb <= 0))})

    with open(RESULTS / "gap_bootstrap.json", "w", encoding="utf-8") as f:
        json.dump({"models_order": list(models), **out}, f, indent=2)

    # ---- markdown tables ---------------------------------------------------------
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    for name, mo in out["models"].items():
        print(f"\n{mo['model_id']} (test_new, {n} templates, chosen layer {mo['chosen_layer']})\n")
        print("| Layer (depth) | S [95% CI] | O [95% CI] | S - O [95% CI] | P(gap <= 0) |")
        print("|---|---|---|---|---|")
        for layer, r in mo["layers"].items():
            print(f"| {layer} ({r['depth']:.0%}) | {f3(r['S'])} [{f3(r['S_ci'][0])}, {f3(r['S_ci'][1])}] "
                  f"| {f3(r['O'])} [{f3(r['O_ci'][0])}, {f3(r['O_ci'][1])}] "
                  f"| {f3(r['gap'])} [{f3(r['gap_ci'][0])}, {f3(r['gap_ci'][1])}] | {r['p_gap_le_0']:.3f} |")
    if out["cross_model"]:
        a, b = list(models)
        print(f"\nDifference in gap, {a} minus {b}\n")
        print("| Pair | Layers | Diff in gap [95% CI] | P(diff <= 0) |")
        print("|---|---|---|---|")
        for c in out["cross_model"]:
            print(f"| {c['pair']} | {c['layer_a']} vs {c['layer_b']} | {f3(c['diff'])} "
                  f"[{f3(c['ci'][0])}, {f3(c['ci'][1])}] | {c['p_diff_le_0']:.3f} |")


if __name__ == "__main__":
    main()
