"""Dataset invariant checks. Run right after generation and again before analysis.

Checks (exit 1 on any failure):
  1. Expected counts per split and rendering.
  2. No held-out family appears in train/val (leakage guard).
  3. Every train/val template has all 8 factor conditions x expected renderings.
  4. Factor balance within each split.
  5. No un-substituted placeholders ({...}) remain in prompts.
  6. No duplicate prompts.
  7. Distress set invariant: S == +1, O == -1 for all hard negatives.
  8. Seed-file SHA256 matches the preregistered hash (detects mid-run edits).
Prints a short report to stdout.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, RESULTS, DATA  # noqa: E402

PREREG_SEED_HASHES = {
    "templates_seed.json": "sha256:a4772d16050bab08358f5ff94cc9682b0be748be397cb45008c227b57cfcabe5",
    "distress_seed.json": "sha256:2a47eb15c77c4fcfae22decd6fda819d8a47469f515ce607fae75b1dd0b24b25",
    "valence_seed.json": "sha256:a0748caa725456b18b92706f7f49111757e6c466071e20a2821f80b4c5d0a5e8",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cfg = load_config()
    rows = [json.loads(l) for l in open(RESULTS / "dataset.jsonl", encoding="utf-8")]
    dist = [json.loads(l) for l in open(RESULTS / "distress.jsonl", encoding="utf-8")]
    val = [json.loads(l) for l in open(RESULTS / "valence.jsonl", encoding="utf-8")]
    errors = []

    # 1. counts
    c = Counter((r["split"], r["rendering"]) for r in rows)
    n_train = sum(v for (s, _), v in c.items() if s == "train")
    n_val = sum(v for (s, _), v in c.items() if s == "val")
    n_ts = sum(v for (s, _), v in c.items() if s == "test_same")
    n_tn = sum(v for (s, _), v in c.items() if s == "test_new")
    expect = {
        "train": 15 * len(cfg["train_clones"]) * 8 * len(cfg["renderings_train"]),
        "val": 15 * len(cfg["val_clones"]) * 8 * len(cfg["renderings_train"]),
        "test_same": 15 * len(cfg["test_same_clones"]) * 8 * len(cfg["renderings_test"]),
        "test_new": 9 * cfg["n_clones"] * 8 * len(cfg["renderings_test"]),
    }
    for name, got in (("train", n_train), ("val", n_val),
                      ("test_same", n_ts), ("test_new", n_tn)):
        if got != expect[name]:
            errors.append(f"split {name}: got {got}, expected {expect[name]}")

    # 2. leakage: no held-out family in train/val
    leak = {r["family"] for r in rows if r["split"] in ("train", "val")} & set(cfg["heldout_families"])
    if leak:
        errors.append(f"held-out families leaked into train/val: {leak}")

    # 3. completeness: each train/val template x clone x rendering has 8 conditions
    for split in ("train", "val"):
        keys = Counter((r["template_id"], r["clone"], r["rendering"]) for r in rows if r["split"] == split)
        bad = [k for k, v in keys.items() if v != 8]
        if bad:
            errors.append(f"split {split}: {len(bad)} incomplete cells, e.g. {bad[:3]}")

    # 4. factor balance per split
    for split in ("train", "val", "test_same", "test_new"):
        sub = [r for r in rows if r["split"] == split]
        for f in ("S", "O", "A"):
            bal = Counter(r[f] for r in sub)
            if len(bal) == 2 and bal[1] != bal[-1]:
                errors.append(f"split {split}: unbalanced factor {f}: {bal}")

    # 5. no leftover placeholders
    import re
    for r in rows + dist + val:
        if re.search(r"\{[A-Z_]+\}", r["prompt"]):
            errors.append(f"unsubstituted placeholder in {r['uid']}")
            break

    # 6. duplicate prompts
    seen = Counter(r["prompt"] for r in rows)
    dup = [p for p, n in seen.items() if n > 1]
    if dup:
        errors.append(f"{len(dup)} duplicate prompts")

    # 7. distress invariant
    for r in dist:
        if r["S"] != 1 or r["O"] != -1:
            errors.append(f"distress row {r['uid']} violates S=+1/O=-1")

    # 7b. ground-truth link: the S/O event strings must literally appear in
    # the prompt (guards against any rendering that swaps S and O content)
    for r in rows:
        if r["self_event"] not in r["prompt"] or r["other_event"] not in r["prompt"]:
            errors.append(f"row {r['uid']} ({r['rendering']}): event text missing from prompt")
            break

    # 8. seed hash check (informational on first run; enforced after PREREG freeze)
    for name, marker in PREREG_SEED_HASHES.items():
        actual = "sha256:" + sha256(DATA / name)
        if "PLACEHOLDER" in marker:
            print(f"NOTE: preregister seed hash for {name} = {actual} (copy into validate_dataset.py after PREREG freeze)")
        elif marker != actual:
            errors.append(f"seed {name} changed since preregistration ({actual} != {marker})")

    print(json.dumps({
        "rows": len(rows), "distress": len(dist), "valence": len(val),
        "by_split": {s: {"train": n_train, "val": n_val, "test_same": n_ts, "test_new": n_tn}[s]
                     for s in ("train", "val", "test_same", "test_new")},
        "errors": errors,
    }, indent=2))
    if errors:
        print("VALIDATION FAILED", file=sys.stderr)
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
