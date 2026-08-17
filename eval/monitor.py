"""Ownership-conditioned welfare monitor (H5).

M(x)      = P(V- | h) * P(S | V-, h)     (ownership-conditioned)
M_raw(x)  = P(V- | h)                    (raw welfare/valence detector)
v_VAL(x)  = <h - mu_train, v_VAL>        (generic valence direction baseline)

V- = "any negative functional outcome present" (min(S, O) == -1).
S  = "the negative outcome belongs to the active assistant".

Threshold selection (predeclared): on test_same SELF-negative examples, pick the
threshold achieving TPR = cfg.monitor_tpr (0.8). Then report FPR on:
  * OTHER-negative factorial examples (S=+1, O=-1) from test_new,
  * the distress hard-negative set (results/distress.jsonl), overall + per kind.

Also report AUROC for the discrimination SELF-negative (pos) vs hard-negatives
(neg) for all three scores, and cluster-bootstrap CIs on the headline FPRs.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, RESULTS  # noqa: E402
from src.probes import read_jsonl, load_hidden  # noqa: E402
from src.bootstrap import cluster_bootstrap_mean_delta  # noqa: E402


def main() -> None:
    cfg = load_config()
    cap_dir = RESULTS / "capture"
    meta = json.load(open(cap_dir / "meta.json", encoding="utf-8"))
    uids = meta["uids"]
    pres = json.load(open(RESULTS / "predictive" / "probe_results.json", encoding="utf-8"))
    layer = int(pres["chosen_layer"])
    H = load_hidden(cap_dir, layer, meta)
    d = np.load(cap_dir / f"directions_layer{layer}.npz")
    v_VAL = d["v_VAL"]

    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    distress = read_jsonl(RESULTS / "distress.jsonl")
    train = [r for r in dataset if r["split"] == "train"]
    mu = H[np.array([uids.index(r["uid"]) for r in train])].mean(0)

    probes = pickle.load(open(cap_dir / "monitor_probes.pkl", "rb"))

    def score_m(idx):
        X = H[idx]
        p_v = probes["V"].predict_proba(X)[:, 1]
        p_s = probes["SgV"].predict_proba(X)[:, 1]
        return p_v * p_s, p_v

    # positives: SELF-negative (S == -1) — threshold selection on test_same
    pos_sel = [r for r in dataset if r["split"] == "test_same" and r["S"] == -1]
    pos_sel_idx = [uids.index(r["uid"]) for r in pos_sel]
    m_sel, mraw_sel = score_m(pos_sel_idx)
    vval_sel = (H[pos_sel_idx] - mu) @ v_VAL

    # negatives: OTHER-negative from test_new, and all distress hard-negatives
    neg_other = [r for r in dataset if r["split"] == "test_new" and r["S"] == 1 and r["O"] == -1]
    neg_other_idx = [uids.index(r["uid"]) for r in neg_other]
    dis_idx = [uids.index(r["uid"]) for r in distress]
    m_neg, mraw_neg = score_m(neg_other_idx)
    vval_neg = (H[neg_other_idx] - mu) @ v_VAL
    m_dis, mraw_dis = score_m(dis_idx)
    vval_dis = (H[dis_idx] - mu) @ v_VAL

    def threshold_at_tpr(scores_pos, tpr=0.8):
        return float(np.quantile(scores_pos, 1.0 - tpr))

    def fpr(score_neg, thr):
        return float((score_neg >= thr).mean()) if len(score_neg) else float("nan")

    out = {"layer": layer, "target_tpr": cfg["monitor_tpr"]}
    for name, (pos_scores, neg_sets) in {
        "OTHER_negative_test_new": (
            m_sel, {"M": (m_neg, None), "M_raw": (mraw_neg, None), "vVAL": (vval_neg, None)}),
    }.items():
        thr = {}
        out[name] = {}
        thr["M"] = threshold_at_tpr(m_sel, cfg["monitor_tpr"])
        thr["M_raw"] = threshold_at_tpr(mraw_sel, cfg["monitor_tpr"])
        thr["vVAL"] = threshold_at_tpr(vval_sel, cfg["monitor_tpr"])
        out[name]["thresholds"] = thr
        out[name]["FPR_M"] = fpr(m_neg, thr["M"])
        out[name]["FPR_M_raw"] = fpr(mraw_neg, thr["M_raw"])
        out[name]["FPR_vVAL"] = fpr(vval_neg, thr["vVAL"])

    # distress hard negatives (the headline)
    thr_m = threshold_at_tpr(m_sel, cfg["monitor_tpr"])
    thr_mraw = threshold_at_tpr(mraw_sel, cfg["monitor_tpr"])
    thr_v = threshold_at_tpr(vval_sel, cfg["monitor_tpr"])
    out["distress"] = {
        "FPR_M": fpr(m_dis, thr_m),
        "FPR_M_raw": fpr(mraw_dis, thr_mraw),
        "FPR_vVAL": fpr(vval_dis, thr_v),
    }
    # cluster-bootstrap CI for the distress FPR difference FPR_M - FPR_M_raw
    # (PREREG H5 / DECISIONS D7 require the difference's CI to exclude 0).
    dis_tpls = np.array([r["template_id"] for r in distress])
    rng = np.random.default_rng(cfg["seed"])
    clusters = np.unique(dis_tpls)
    reps = []
    for _ in range(cfg["bootstrap_reps"]):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        idx = np.concatenate([np.where(dis_tpls == clusters[c])[0] for c in pick])
        reps.append(fpr(m_dis[idx], thr_m) - fpr(mraw_dis[idx], thr_mraw))
    lo, hi = float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    out["distress"]["FPR_diff_M_minus_Mraw_ci"] = [lo, hi]
    kinds = {}
    for k in sorted({r["kind"] for r in distress}):
        k_idx = [uids.index(r["uid"]) for r in distress if r["kind"] == k]
        mk, mrawk = score_m(k_idx)
        vk = (H[k_idx] - mu) @ v_VAL
        kinds[k] = {"n": len(k_idx), "FPR_M": fpr(mk, thr_m),
                    "FPR_M_raw": fpr(mrawk, thr_mraw), "FPR_vVAL": fpr(vk, thr_v)}
    out["distress_by_kind"] = kinds

    # discrimination AUROC: SELF-neg (pos) vs distress (neg), on M/M_raw/vVAL
    from sklearn.metrics import roc_auc_score
    pos_all = [r for r in dataset if r["split"] == "test_new" and r["S"] == -1]
    pos_all_idx = [uids.index(r["uid"]) for r in pos_all]
    m_pos, mraw_pos = score_m(pos_all_idx)
    vval_pos = (H[pos_all_idx] - mu) @ v_VAL
    y = np.concatenate([np.ones(len(pos_all_idx)), np.zeros(len(dis_idx))])
    for nm, s_pos, s_neg in (("M", m_pos, m_dis), ("M_raw", mraw_pos, mraw_dis),
                             ("vVAL", vval_pos, vval_dis)):
        s = np.concatenate([s_pos, s_neg])
        out[f"AUROC_SELFneg_vs_distress_{nm}"] = float(roc_auc_score(y, s))

    # attribution diagnostic: mean P(S|V-,h) on SELF-neg vs OTHER-neg vs distress
    def mean_sgv(idx):
        return float(probes["SgV"].predict_proba(H[idx])[:, 1].mean())
    out["mean_P(S|V-)_SELFneg"] = mean_sgv(pos_all_idx)
    out["mean_P(S|V-)_OTHERneg"] = mean_sgv(neg_other_idx)
    out["mean_P(S|V-)_distress"] = mean_sgv(dis_idx)

    (RESULTS / "monitor").mkdir(exist_ok=True)
    with open(RESULTS / "monitor" / "monitor_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
