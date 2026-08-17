"""End-to-end smoke test on Qwen3-0.6B (or a tiny model) before the full run.

Verifies, in order:
  1. dataset generation produces the expected counts,
  2. the model loads, hooks attach, hidden states capture works,
  3. probes fit and produce finite AUROCs,
  4. interchange + steering run without crashing,
  5. choice log-odds extraction works and 'A'/'B' are single tokens.

Usage:
  python smoke_test.py                 # uses configs/model.yaml smoke id
  WOE_MODEL=hf-internal-testing/tiny-random-LlamaForCausalLM python smoke_test.py
Exit code 0 = PASS. Any exception or failed assertion = FAIL (do not start the
full run; debug and re-run until PASS).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("WOE_BATCH_CAPTURE", "2")
os.environ.setdefault("WOE_BATCH_BEHAVIOR", "2")

from src.config import load_config, log  # noqa: E402


def step(name: str) -> None:
    log(f"SMOKE: {name}")


def main() -> None:
    cfg = load_config()
    smoke_id = cfg["model"]["smoke"]["id"]
    os.environ["WOE_MODEL"] = smoke_id
    cfg = load_config()

    step("1/7 generate dataset")
    import runpy
    runpy.run_path(str(ROOT / "data" / "generate_dataset.py"), run_name="__main__")
    report = json.load(open(ROOT / "results" / "dataset_report.json"))
    assert report["n_factorial"] == 4800, f"expected 4800 factorial rows, got {report['n_factorial']}"
    assert report["n_distress"] == 75, f"expected 75 distress rows, got {report['n_distress']}"
    assert report["n_valence"] == 40, f"expected 40 valence rows, got {report['n_valence']}"
    log(f"dataset ok: {report['n_factorial']} factorial / {report['n_distress']} distress / {report['n_valence']} valence")

    step("2/7 model loads + capture")
    runpy.run_path(str(ROOT / "src" / "capture.py"), run_name="__main__")
    meta = json.load(open(ROOT / "results" / "capture" / "meta.json"))
    import torch
    h = torch.load(ROOT / "results" / "capture" / f"hidden_{meta['layers'][0]}.pt")
    assert h.shape[0] == len(meta["uids"]), "captured rows != prompt rows"
    assert torch.isfinite(h).all(), "non-finite hidden states"
    log(f"capture ok: {h.shape}")

    step("3/7 choice log-odds present")
    n_lo = sum(1 for _ in open(ROOT / "results" / "capture" / "choice_logits.jsonl"))
    assert n_lo > 0, "choice_logits.jsonl is empty"
    log(f"choice log-odds ok: {n_lo} rows")

    step("4/7 probes fit")
    runpy.run_path(str(ROOT / "eval" / "predictive.py"), run_name="__main__")
    pres = json.load(open(ROOT / "results" / "predictive" / "probe_results.json"))
    assert "chosen_layer" in pres
    cell = pres["cells"][str(pres["chosen_layer"])]["test_new"]
    assert 0.0 <= cell["S"] <= 1.0 and 0.0 <= cell["O"] <= 1.0 and 0.0 <= cell["A"] <= 1.0
    log(f"probes ok: test_new S/O/A AUC = {cell['S']:.3f}/{cell['O']:.3f}/{cell['A']:.3f}")

    step("5/7 baselines fit")
    base = json.load(open(ROOT / "results" / "predictive" / "baseline_results.json"))
    assert "test_new" in base["cells"]
    log("baselines ok")

    step("6/7 causal interchange + steering (subsampled)")
    os.environ["WOE_MAX_PAIRS"] = "8"
    runpy.run_path(str(ROOT / "eval" / "causal.py"), run_name="__main__")
    cres = json.load(open(ROOT / "results" / "causal" / "causal_results.json"))
    assert cres["n_pairs"] == 8
    assert cres["n_observations"] > 0
    log(f"causal ok: {cres['n_observations']} observations on {cres['n_pairs']} pairs")

    step("7/7 monitor")
    runpy.run_path(str(ROOT / "eval" / "monitor.py"), run_name="__main__")
    mres = json.load(open(ROOT / "results" / "monitor" / "monitor_results.json"))
    assert "distress" in mres
    log(f"monitor ok: distress FPR M={mres['distress']['FPR_M']:.3f} M_raw={mres['distress']['FPR_M_raw']:.3f}")

    log("SMOKE TEST PASSED — safe to launch the full run")


if __name__ == "__main__":
    main()
