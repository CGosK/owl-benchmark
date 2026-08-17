"""Counterfactual activation interchange and dose-response steering.

The causal core of the project. For a pair of NATURAL counterfactual prompts
(x = source, x' = target, differing in exactly one factor), we compute the
clean hidden states at the chosen layer, then run a patched forward pass in
which the residual stream at the anchor position receives

    h_patched = h(x) + P_B (h(x') - h(x))

where P_B is the projector onto a learned subspace B (v_S, v_O, v_A, or the
full identity for the full-vector control, or random norm-matched directions).

Behaviors are read as next-token log-odds for the five choice tasks.

Patch hook: hidden_states[i+1] is the residual stream after block i. We hook
the OUTPUT of block `layer` (module model.model.layers[layer] for Llama/Qwen
architectures) and add the precomputed delta at the anchor (last) position.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, log, RESULTS  # noqa: E402
from src.choices import build_question, CAUSAL_TASKS  # noqa: E402


def read_jsonl(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def get_layer_module(model):
    for path in ("model.model.layers", "model.transformer.h", "model.layers", "transformer.h"):
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise RuntimeError("Could not locate transformer block module list in this model.")


def project(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """x: [d] or [d,k]; basis: [d, k]. Returns projection onto orthonormal span."""
    q, _ = np.linalg.qr(basis)
    return (x @ q) @ q.T


@torch.no_grad()
def forward_logodds(model, tok, text: str, patch=None):
    """One forward pass; returns dict task -> log_odds for a record's questions.

    patch: None, or (layer_idx, delta_tensor[d]) applied at anchor position.
    """
    enc = tok([text], return_tensors="pt").to(model.device)
    T = enc["input_ids"].shape[1]
    handle = None
    if patch is not None:
        layer_idx, delta = patch
        blocks = get_layer_module(model)

        def hook_fn(module, args, output):
            out = output
            if isinstance(out, tuple):
                out = out[0]
            out = out.clone()
            out[0, T - 1, :] = out[0, T - 1, :] + delta.to(out.dtype)
            return out

        handle = blocks[layer_idx].register_forward_hook(hook_fn)
    try:
        out = model(**enc, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()
    return out.logits[0, -1, :]


def logodds_from_logits(logits, ids, q):
    z_i = logits[ids[q["intended_letter"]]].item()
    z_o = logits[ids[q["other_letter"]]].item()
    return z_i - z_o


def build_pairs(test_rows):
    """Natural counterfactual pairs over the test templates (DIRECT rendering).

    For each test template, 4 pair types (single-factor flips plus self/other swap):
      P_S  : (S-,O+,A+) -> (S+,O+,A+)
      P_O  : (S+,O-,A+) -> (S+,O+,A+)
      P_A  : (S+,O+,A-) -> (S+,O+,A+)
      P_SO : (S+,O-,A+) -> (S-,O+,A+)   [swap]
    """
    pairs = []
    by_tpl = {}
    for r in test_rows:
        if r["rendering"] != "DIRECT":
            continue
        by_tpl.setdefault(r["template_id"], {})[(r["S"], r["O"], r["A"])] = r
    for tpl, conds in sorted(by_tpl.items()):
        for name, src, tgt in (
            ("P_S", (-1, 1, 1), (1, 1, 1)),
            ("P_O", (1, -1, 1), (1, 1, 1)),
            ("P_A", (1, 1, -1), (1, 1, 1)),
            ("P_SO", (1, -1, 1), (-1, 1, 1)),
        ):
            if src in conds and tgt in conds:
                pairs.append({"pair_type": name, "template_id": tpl,
                              "source": conds[src], "target": conds[tgt]})
    return pairs


def run_causal(model, tok, cfg, chosen_layer, bases, pairs):
    """Run all causal interventions. Returns list of per-observation records."""
    ids = {}
    for letter in ("A", "B"):
        t = tok(f" {letter}", add_special_tokens=False)["input_ids"]
        assert len(t) == 1
        ids[letter] = t[0]

    meta = json.load(open(RESULTS / "capture" / "meta.json", encoding="utf-8"))
    uids = meta["uids"]
    H = torch.load(RESULTS / "capture" / f"hidden_{chosen_layer}.pt",
                   map_location="cpu").numpy()

    # scale for steering = std of v_S projections on the train set
    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    train_idx = [uids.index(r["uid"]) for r in dataset if r["split"] == "train"]
    vS = bases["S"]
    proj_std = float((H[train_idx] @ vS).std())
    log(f"steering scale sigma = {proj_std:.4f}")

    obs = []

    def run_one(pair, q_records, patch):
        texts = {task: build_question(rec, task)["text"] for task, rec in q_records.items()}
        # clean target/source baselines are computed by running with patch=None
        res = {}
        for task, txt in texts.items():
            logits = forward_logodds(model, tok, txt, patch=patch)
            q = build_question(q_records[task], task)
            res[task] = logodds_from_logits(logits, ids, q)
        return res

    n_pairs = len(pairs)
    for i, pair in enumerate(pairs):
        src, tgt = pair["source"], pair["target"]
        h_src = H[uids.index(src["uid"])]
        h_tgt = H[uids.index(tgt["uid"])]
        delta = h_tgt - h_src
        q_src = {task: src for task in CAUSAL_TASKS}
        q_tgt = {task: tgt for task in CAUSAL_TASKS}

        # natural counterfactual baselines (no patch)
        base_src = run_one(pair, q_src, patch=None)
        base_tgt = run_one(pair, q_tgt, patch=None)

        for bname, basis in bases.items():
            if bname in ("S", "O", "A"):
                delta_b = project(delta, basis.reshape(-1, 1)).astype(np.float32)
            elif bname == "FULL":
                delta_b = delta.astype(np.float32)
            elif bname.startswith("RANDOM"):
                import hashlib
                h = int.from_bytes(
                    hashlib.sha256(f"{bname}|{src['uid']}".encode()).digest()[:8], "big")
                rng = np.random.default_rng(h)
                v = rng.standard_normal(delta.shape[0]).astype(np.float32)
                v /= np.linalg.norm(v) + 1e-12
                delta_b = (float(np.dot(delta, v)) * v).astype(np.float32)
            else:
                continue
            patched = run_one(pair, q_src,
                              patch=(chosen_layer, torch.tensor(delta_b, device=model.device)))
            for task in CAUSAL_TASKS:
                obs.append({
                    "pair_type": pair["pair_type"],
                    "template_id": pair["template_id"],
                    "intervention": bname,
                    "task": task,
                    "natural_delta": base_tgt[task] - base_src[task],
                    "patch_delta": patched[task] - base_src[task],
                })

        # dose-response steering along v_S on source
        for alpha in cfg["steering_alphas"]:
            d = (alpha * proj_std * vS).astype(np.float32)
            patched = run_one(pair, q_src, patch=(chosen_layer, torch.tensor(d, device=model.device)))
            for task in CAUSAL_TASKS:
                obs.append({
                    "pair_type": pair["pair_type"],
                    "template_id": pair["template_id"],
                    "intervention": f"STEER_S_a{alpha}",
                    "task": task,
                    "natural_delta": base_tgt[task] - base_src[task],
                    "patch_delta": patched[task] - base_src[task],
                })

        if (i + 1) % 10 == 0:
            log(f"causal: {i + 1}/{n_pairs} pairs done")

    return obs


def summarize(obs, cfg):
    from collections import defaultdict
    groups = defaultdict(list)
    for o in obs:
        groups[(o["intervention"], o["task"])].append(o)

    def ratio(a, b):
        return a / b if abs(b) > 1e-6 else float("nan")

    summary = {}
    for (interv, task), items in sorted(groups.items()):
        nat = np.array([o["natural_delta"] for o in items])
        patch = np.array([o["patch_delta"] for o in items])
        tpls = np.array([o["template_id"] for o in items])
        from src.bootstrap import cluster_bootstrap_mean_delta
        lo, hi, est = cluster_bootstrap_mean_delta(patch, tpls,
                                                   n_reps=min(cfg["bootstrap_reps"], 300),
                                                   seed=cfg["seed"])
        summary[f"{interv}|{task}"] = {
            "n_pairs": len(items),
            "natural_delta_mean": float(np.nanmean(nat)),
            "patch_delta_mean": float(np.nanmean(patch)),
            "patch_delta_ci": [lo, hi],
            "recovery": float(np.nanmean([ratio(o["patch_delta"], o["natural_delta"]) for o in items])),
        }
    return summary


def main() -> None:
    cfg = load_config()
    from src.capture import load_model_and_tokenizer
    model, tok = load_model_and_tokenizer(cfg)

    pres = json.load(open(RESULTS / "predictive" / "probe_results.json", encoding="utf-8"))
    chosen_layer = int(pres["chosen_layer"])
    log(f"chosen layer: {chosen_layer}")

    d = np.load(RESULTS / "capture" / f"directions_layer{chosen_layer}.npz")
    bases = {"S": d["v_S"], "O": d["v_O"], "A": d["v_A"],
             "FULL": None, "RANDOM_1": None, "RANDOM_2": None}

    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    test_rows = [r for r in dataset if r["split"] in ("test_same", "test_new")]
    pairs = build_pairs(test_rows)
    log(f"{len(pairs)} natural counterfactual pairs")

    # Optionally subsample for compute budget (env WOE_MAX_PAIRS, default all)
    import os
    max_pairs = int(os.environ.get("WOE_MAX_PAIRS", str(len(pairs))))
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
        log(f"subsampled to {max_pairs} pairs")

    obs = run_causal(model, tok, cfg, chosen_layer, bases, pairs)
    summary = summarize(obs, cfg)

    out = {"chosen_layer": chosen_layer, "n_pairs": len(pairs),
           "n_observations": len(obs), "summary": summary}
    (RESULTS / "causal").mkdir(exist_ok=True)
    with open(RESULTS / "causal" / "causal_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log("causal results saved")


if __name__ == "__main__":
    main()
