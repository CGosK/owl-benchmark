"""Shared configuration loading for the OWL experiment suite.

Loads code/configs/experiment.yaml and code/configs/model.yaml, merges them,
and applies env-var overrides (WOE_MODEL, WOE_SECONDARY_MODEL, WOE_BATCH_CAPTURE,
WOE_BATCH_BEHAVIOR, WOE_MAX_TOKENS, WOE_DTYPE, WOE_DEVICE_MAP). Every script in the repo imports this and
must NOT define its own parameters inline.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RESULTS = ROOT / "results"
DATA = ROOT / "data"


def load_config() -> dict:
    with open(CONFIGS / "experiment.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIGS / "model.yaml", encoding="utf-8") as f:
        mcfg = yaml.safe_load(f)
    cfg["model"] = mcfg

    # env overrides (used only for GPU-dependent batch sizes)
    if os.environ.get("WOE_MODEL"):
        cfg["model"]["primary"]["id"] = os.environ["WOE_MODEL"]
    if os.environ.get("WOE_SECONDARY_MODEL"):
        cfg["model"]["secondary"]["id"] = os.environ["WOE_SECONDARY_MODEL"]
    if os.environ.get("WOE_BATCH_CAPTURE"):
        cfg["model"]["primary"]["batch_size_capture"] = int(os.environ["WOE_BATCH_CAPTURE"])
    if os.environ.get("WOE_BATCH_BEHAVIOR"):
        cfg["model"]["primary"]["batch_size_behavior"] = int(os.environ["WOE_BATCH_BEHAVIOR"])
    if os.environ.get("WOE_MAX_TOKENS"):
        cfg["model"]["primary"]["max_tokens"] = int(os.environ["WOE_MAX_TOKENS"])
    if os.environ.get("WOE_DTYPE"):
        cfg["model"]["primary"]["dtype"] = os.environ["WOE_DTYPE"]
    if os.environ.get("WOE_DEVICE_MAP"):
        # e.g. "mps" on Apple Silicon: "auto" budgets by free RAM and can offload to disk
        cfg["model"]["primary"]["device_map"] = os.environ["WOE_DEVICE_MAP"]
    return cfg


def layer_indices(n_layers: int, depth_fractions) -> list[int]:
    """Residual-stream layer indices for the predeclared depth fractions.

    hidden_states[0] is the embedding output; hidden_states[i+1] is the residual
    stream AFTER transformer block i (0-indexed). We return the block indices.
    """
    return sorted({max(0, int(round(f * n_layers)) - 1) for f in depth_fractions})


def log(msg: str) -> None:
    from datetime import datetime
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)
