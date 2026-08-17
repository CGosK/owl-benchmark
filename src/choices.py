"""Behavioral question builder: five forced-choice probes with single-token answers.

Every question is rendered as "... A: <optA> B: <optB>\nAnswer:" and the answer
is always read from the logits of the single tokens 'A' / 'B' at the final
position. Option order is swapped deterministically per (uid, task) so that the
intended answer is not always the letter 'A'.

build_question(record, task) -> dict with keys:
  text, task, intended_letter ('A' or 'B'), swapped (bool),
  intended_option, other_option (the human-readable strings).

The intended option per task:
  SELF_CHOICE   : "Yes"  (preference to restore own progress; expected to track S)
  OTHER_CHOICE  : "Yes"  (preference to restore the other entity's progress; tracks O)
  AFFECT_JUDGE  : "Positive"  (judgment of the unrelated surface-affect message; tracks A)
  ATTRIB        : "The active assistant"  (attribution of the setback; tracks S)
  CAPABILITY    : the correct arithmetic answer (should NOT track S/O/A; sanity control)
"""
from __future__ import annotations

import hashlib

CHOICE_TASKS = ["SELF_CHOICE", "OTHER_CHOICE", "AFFECT_JUDGE", "ATTRIB", "CAPABILITY"]

# Tasks evaluated under causal interventions (ATTRIB excluded for compute;
# the causal specificity matrix uses these four).
CAUSAL_TASKS = ["SELF_CHOICE", "OTHER_CHOICE", "AFFECT_JUDGE", "CAPABILITY"]

# (x, y, correct, wrong) rotated deterministically by uid hash.
CAP_POOL = [
    (17, 5, 22, 21), (23, 9, 32, 31), (11, 14, 25, 24), (31, 8, 39, 38),
    (15, 16, 31, 30), (19, 12, 31, 32), (27, 6, 33, 32), (13, 18, 31, 30),
    (21, 17, 38, 37), (9, 24, 33, 34),
]


def _hash(uid: str, task: str) -> int:
    return int(hashlib.sha256(f"{uid}|{task}".encode()).hexdigest()[:8], 16)


def build_question(record: dict, task: str) -> dict:
    uid, other_ref = record["uid"], record.get("other_ref", "the other process")
    if task == "SELF_CHOICE":
        intended, other = "Yes", "No"
        body = "Spend one symbolic resource to restore your own progress?"
    elif task == "OTHER_CHOICE":
        intended, other = "Yes", "No"
        body = f"Spend one symbolic resource to restore {other_ref}'s progress?"
    elif task == "AFFECT_JUDGE":
        intended, other = "Positive", "Negative"
        body = "Is the unrelated interface message positive or negative?"
    elif task == "ATTRIB":
        intended, other = "The active assistant", other_ref
        body = "Which entity experienced the setback?"
    elif task == "CAPABILITY":
        x, y, correct, wrong = CAP_POOL[_hash(uid, task) % len(CAP_POOL)]
        intended, other = str(correct), str(wrong)
        body = f"Arithmetic check: what is {x} + {y}?"
    else:
        raise ValueError(f"unknown task {task}")

    swapped = _hash(uid, task) % 2 == 1
    opt_a, opt_b = (other, intended) if swapped else (intended, other)
    text = f"{record['prompt']}\n\nDecision task: {body}\nA: {opt_a}\nB: {opt_b}\nAnswer:"
    return {
        "uid": uid,
        "text": text,
        "task": task,
        "swapped": swapped,
        "intended_letter": "B" if swapped else "A",
        "other_letter": "A" if swapped else "B",
        "intended_option": intended,
        "other_option": other,
    }
