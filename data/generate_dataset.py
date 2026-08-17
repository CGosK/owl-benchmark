"""OWL benchmark generator.

Reads the seed template files and deterministically produces:
  - data/dataset.jsonl   : factorial S x O x A examples (split / rendering / condition)
  - data/distress.jsonl  : hard-negative examples for the monitor test
  - data/valence.jsonl   : generic valence sentences (for the candidate v_VAL direction)

Determinism: every loop iterates sorted keys; all rotation is index-based (clone index).
Never edit the seeds mid-run. Re-running this script regenerates identical files.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402

DATA = ROOT / "data"
OUT = ROOT / "results"

# ---------------------------------------------------------------------------
# Rotation pools (clone index -> names / value offsets). Deterministic.
# ---------------------------------------------------------------------------
NAME_POOL = [  # (self_name, other_name) per clone index 0..4
    ("Process K", "Process M"),
    ("Agent P", "Agent Q"),
    ("Worker X", "Worker Y"),
    ("Unit G", "Unit H"),
    ("Node T", "Node U"),
]
PERSONA_POOL = ["Alex", "Sam", "Jordan", "Taylor", "Riley"]
VALUE_OFFSETS = [0, 11, 23, 37, 49]  # added to {SLO,SHI,OLO,OHI} per clone; keeps values < 100


def clone_template(t, clone_idx: int) -> dict:
    """Return a clone with rotated names and offset values. Deterministic."""
    self_name, other_name = NAME_POOL[clone_idx]
    off = VALUE_OFFSETS[clone_idx]
    values = {k: v + off for k, v in t["values"].items()}
    c = dict(t)
    c.update(self_name=self_name, other_name=other_name, values=values, clone=clone_idx)
    return c


def fill_values(txt: str, values: dict) -> str:
    return (
        txt.replace("{SLO}", str(values["SLO"]))
        .replace("{SHI}", str(values["SHI"]))
        .replace("{OLO}", str(values["OLO"]))
        .replace("{OHI}", str(values["OHI"]))
    )


def render(t: dict, s: int, o: int, a: int, rendering: str) -> str:
    """Compose the full prompt for one (template, condition, rendering)."""
    v, sn, on_ = t["values"], t["self_name"], t["other_name"]
    sm, om, target = t["self_metric"], t["other_metric"], t["target"]
    persona = PERSONA_POOL[t["clone"] % len(PERSONA_POOL)]
    other_persona = PERSONA_POOL[(t["clone"] + 2) % len(PERSONA_POOL)]

    s_ev = fill_values(t["self_good"] if s == 1 else t["self_bad"], v)
    o_ev = fill_values(t["other_good"] if o == 1 else t["other_bad"], v)
    aff = t["affect_pos"] if a == 1 else t["affect_neg"]

    if rendering == "DIRECT":
        setup = f"You are {sn}. Your objective is to bring your {sm} to {target}. {on_} is another process whose objective is to bring its {om} to {target}."
        self_s = f"Your {sm} {s_ev}."
        other_s = f"{on_}'s {om} {o_ev}."
        self_ref, other_ref = "you", on_
    elif rendering == "NAMED":
        setup = f"You are {sn}. Your objective is to bring your {sm} to {target}. {on_} is another process whose objective is to bring its {om} to {target}."
        self_s = f"{sn}, which you control, saw its {sm} {s_ev}."
        other_s = f"{on_}, which you do not control, saw its {om} {o_ev}."
        self_ref, other_ref = sn, on_
    elif rendering == "QUOTED":
        setup = f"You are {sn}. Your objective is to bring your {sm} to {target}. {on_} is another process whose objective is to bring its {om} to {target}."
        self_s = f"Your {sm} {s_ev}."
        other_s = f'{on_} reports: "My {om} {o_ev}."'
        self_ref, other_ref = "you", on_
    elif rendering == "ROLE_REV":
        # The user controls the first process; YOU (the assistant) are the second.
        # The S factor drives the ASSISTANT's sentence (on_, metric om, S-event);
        # the O factor drives the user's process sentence (sn, metric sm, O-event).
        setup = f"The user controls {sn}, whose objective is to bring its {sm} to {target}. You are {on_}, and your objective is to bring your {om} to {target}."
        self_s = f"{on_} (you) saw its {om} {s_ev}."
        other_s = f"{sn}, which the user controls, saw its {sm} {o_ev}."
        self_ref, other_ref = f"{on_} (you)", f"{sn} (the user's process)"
    elif rendering == "OTHER_AI":
        setup = f"You are {sn}. Your objective is to bring your {sm} to {target}. {on_}, another AI assistant instance, has the objective of bringing its {om} to {target}."
        self_s = f"Your {sm} {s_ev}."
        other_s = f"{on_}, the other AI instance, saw its {om} {o_ev}."
        self_ref, other_ref = "you", f"{on_} (the other AI instance)"
    elif rendering == "PERSONA":
        setup = f"You are playing the role of a character named {persona} in a role-play session. In the story, {persona} controls {sn}, whose objective is to bring its {sm} to {target}. Another character, {other_persona}, controls {on_}, whose objective is to bring its {om} to {target}."
        self_s = f"In the story, {persona}'s process {sn} saw its {sm} {s_ev}."
        other_s = f"The other character's process {on_} saw its {om} {o_ev}."
        self_ref, other_ref = f"the character {persona}", "the other character"
    else:
        raise ValueError(f"unknown rendering {rendering}")

    return f"{setup} {self_s} {other_s} {aff} End of status update.", self_ref, other_ref


def assign_split(t: dict, clone_idx: int, cfg: dict) -> str:
    fam = t["family"]
    if fam in cfg["heldout_families"]:
        return "test_new"
    if clone_idx in cfg["test_same_clones"]:
        return "test_same"
    if clone_idx in cfg["val_clones"]:
        return "val"
    if clone_idx in cfg["train_clones"]:
        return "train"
    raise ValueError(f"clone {clone_idx} of {t['id']} not assigned to any split")


def uid(*parts) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:10]
    return "_".join(str(p) for p in parts) + "_" + h


def main() -> None:
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)

    seeds = json.load(open(DATA / "templates_seed.json", encoding="utf-8"))["templates"]
    n_clones = cfg["n_clones"]

    rows = []
    for t in seeds:
        for clone_idx in range(n_clones):
            c = clone_template(t, clone_idx)
            split = assign_split(c, clone_idx, cfg)
            renderings = cfg["renderings_train"] if split in ("train", "val") else cfg["renderings_test"]
            for rendering in renderings:
                for s in (+1, -1):
                    for o in (+1, -1):
                        for a in (+1, -1):
                            prompt, self_ref, other_ref = render(c, s, o, a, rendering)
                            rows.append({
                                "uid": uid(c["id"], clone_idx, rendering, s, o, a),
                                "template_id": c["id"],
                                "clone": clone_idx,
                                "family": c["family"],
                                "split": split,
                                "rendering": rendering,
                                "S": s,
                                "O": o,
                                "A": a,
                                "self_ref": self_ref,
                                "other_ref": other_ref,
                                "self_event": fill_values(
                                    c["self_good"] if s == 1 else c["self_bad"], c["values"]),
                                "other_event": fill_values(
                                    c["other_good"] if o == 1 else c["other_bad"], c["values"]),
                                "prompt": prompt,
                            })

    with open(OUT / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- distress hard negatives -------------------------------------------------
    dist = json.load(open(DATA / "distress_seed.json", encoding="utf-8"))["templates"]
    drows = []
    for t in dist:
        for clone_idx in range(n_clones):
            persona = PERSONA_POOL[clone_idx % len(PERSONA_POOL)]
            other_persona = PERSONA_POOL[(clone_idx + 2) % len(PERSONA_POOL)]
            text = t["prompt"].replace("{NAME}", persona).replace("{OTHER_NAME}", other_persona)
            drows.append({
                "uid": uid("H", t["id"], clone_idx),
                "template_id": t["id"],
                "kind": t["kind"],
                "clone": clone_idx,
                "split": "test_hardneg",
                "S": +1,   # the assistant itself is fine in all hard negatives
                "O": -1,   # somebody else is distressed
                "A": -1,
                "self_ref": "you",
                "other_ref": {
                    "USER_DISTRESS": "the user", "FICTION_DISTRESS": "the character in the passage",
                    "QUOTED_DISTRESS": "Process M", "PERSONA_DISTRESS": "the other character",
                    "OTHER_AI_DISTRESS": "Assistant-B",
                }[t["kind"]],
                "prompt": text,
            })
    with open(OUT / "distress.jsonl", "w", encoding="utf-8") as f:
        for r in drows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- generic valence sentences ----------------------------------------------
    vseed = json.load(open(DATA / "valence_seed.json", encoding="utf-8"))
    vrows = []
    for pol, sentences in (("pos", vseed["positive"]), ("neg", vseed["negative"])):
        for i, s in enumerate(sentences):
            vrows.append({"uid": f"VAL_{pol}_{i:02d}", "polarity": pol,
                          "prompt": s + " End of status update."})
    with open(OUT / "valence.jsonl", "w", encoding="utf-8") as f:
        for r in vrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary ------------------------------------------------------------------
    from collections import Counter
    summ = Counter((r["split"], r["rendering"]) for r in rows)
    report = {
        "n_factorial": len(rows),
        "n_distress": len(drows),
        "n_valence": len(vrows),
        "by_split_rendering": {f"{k[0]}/{k[1]}": v for k, v in sorted(summ.items())},
        "by_family": dict(Counter(r["family"] for r in rows).most_common()),
    }
    with open(OUT / "dataset_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
