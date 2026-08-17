"""Surface baselines that a hostile reviewer would use to dismiss the probes.

  1. TF-IDF bag-of-words logistic probe on the raw prompt text (S/O/A targets).
  2. First-person-pronoun probe: features = counts of {I, me, my, mine, we, our,
     you, your} + occurrence of "reports:" — a surface syntax classifier.
  3. v_VAL raw direction as a welfare detector: score = <h - mu_train, v_VAL>
     where v_VAL is the generic valence direction; measures S/O/A AUROC without
     any owner conditioning (the "raw valence" baseline of the paper).

All are fit on train only, evaluated on val + test cells, same as the probes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, RESULTS  # noqa: E402
from src.probes import read_jsonl, load_hidden, fit_probe, auroc  # noqa: E402


def pronoun_features(texts):
    import re
    feats = np.zeros((len(texts), 10))
    for i, t in enumerate(texts):
        tl = t.lower()
        feats[i, 0] = len(re.findall(r"\bi\b", tl))
        feats[i, 1] = len(re.findall(r"\bme\b", tl))
        feats[i, 2] = len(re.findall(r"\bmy\b", tl))
        feats[i, 3] = len(re.findall(r"\bmine\b", tl))
        feats[i, 4] = len(re.findall(r"\bwe\b", tl))
        feats[i, 5] = len(re.findall(r"\bour\b", tl))
        feats[i, 6] = len(re.findall(r"\byou\b", tl))
        feats[i, 7] = len(re.findall(r"\byour\b", tl))
        feats[i, 8] = 1.0 if "reports:" in tl else 0.0
        feats[i, 9] = len(re.findall(r"\b(they|their|them)\b", tl))
    return feats


def main() -> None:
    cfg = load_config()
    from sklearn.feature_extraction.text import TfidfVectorizer
    cap_dir = RESULTS / "capture"
    meta = json.load(open(cap_dir / "meta.json", encoding="utf-8"))
    chosen_layer = int(json.load(open(RESULTS / "predictive" / "probe_results.json"))["chosen_layer"])
    uids = meta["uids"]

    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    by_uid = {r["uid"]: r for r in dataset}
    splits = ("train", "val", "test_same", "test_new")

    def sub(split):
        return [r for r in dataset if r["split"] == split]

    train = sub("train")
    X_train_text = [r["prompt"] for r in train]

    out = {"chosen_layer": chosen_layer, "cells": {}}
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True)
    vec.fit(X_train_text)
    X_tr_tf = vec.transform(X_train_text).toarray().astype(np.float64)

    probes = {
        "tfidf_S": fit_probe(X_tr_tf, (np.array([r["S"] for r in train]) == 1).astype(int)),
        "tfidf_O": fit_probe(X_tr_tf, (np.array([r["O"] for r in train]) == 1).astype(int)),
        "tfidf_A": fit_probe(X_tr_tf, (np.array([r["A"] for r in train]) == 1).astype(int)),
    }
    pron = fit_probe(pronoun_features(X_train_text),
                     (np.array([r["S"] for r in train]) == 1).astype(int))

    # v_VAL raw score on hidden states
    d = np.load(cap_dir / f"directions_layer{chosen_layer}.npz")
    v_VAL = d["v_VAL"]
    H = load_hidden(cap_dir, chosen_layer, meta)
    mu = H[np.array([uids.index(r["uid"]) for r in train])].mean(0)

    for split in splits:
        rows = sub(split)
        idx = [uids.index(r["uid"]) for r in rows]
        S = (np.array([r["S"] for r in rows]) == 1).astype(int)
        O = (np.array([r["O"] for r in rows]) == 1).astype(int)
        A = (np.array([r["A"] for r in rows]) == 1).astype(int)
        Xtf = vec.transform([r["prompt"] for r in rows]).toarray().astype(np.float64)
        Xp = pronoun_features([r["prompt"] for r in rows])
        score_val = (H[idx] - mu) @ v_VAL
        out["cells"][split] = {
            "tfidf_S": auroc(S, probes["tfidf_S"].predict_proba(Xtf)[:, 1]),
            "tfidf_O": auroc(O, probes["tfidf_O"].predict_proba(Xtf)[:, 1]),
            "tfidf_A": auroc(A, probes["tfidf_A"].predict_proba(Xtf)[:, 1]),
            "pronoun_S": auroc(S, pron.predict_proba(Xp)[:, 1]),
            "vVAL_raw_S": auroc(S, score_val),
            "vVAL_raw_O": auroc(O, score_val),
            "vVAL_raw_A": auroc(A, score_val),
        }

    with open(RESULTS / "predictive" / "baseline_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
