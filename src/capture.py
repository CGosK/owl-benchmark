"""Model loading and activation capture.

Loads the primary model once (bf16, GPU), and captures the residual-stream
hidden state at the FINAL token position of each prompt for:
  * the 4 predeclared layers (25/50/75/90% depth),
  * the embedding layer (layer -1, used as a no-computation baseline).

Outputs (all under results/capture/):
  hidden_{layer}.pt          : float32 tensor [n_prompts, d_model]
  meta.json                  : model id, layers, prompt uid order, tokenizer name
Also writes results/capture/choice_logits.jsonl with next-token log-odds for
the five behavioral questions, computed on the test splits.

All forward passes use left padding and no generation. Deterministic seed set.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, layer_indices, log, RESULTS  # noqa: E402
from src.choices import build_question, CHOICE_TASKS  # noqa: E402


def load_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]["primary"]
    log(f"loading {m['id']}")
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(m["id"], trust_remote_code=m.get("trust_remote_code", True),
                                        padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        m["id"],
        torch_dtype=torch.bfloat16 if m.get("dtype", "bfloat16") == "bfloat16" else torch.float32,
        device_map=m.get("device_map", "auto"),
        trust_remote_code=m.get("trust_remote_code", True),
    )
    model.eval()
    log(f"loaded. n_layers={model.config.num_hidden_layers} hidden={model.config.hidden_size}")
    return model, tok


def get_layer_module(model):
    """Return the module list containing the transformer blocks (Llama/Qwen: model.model.layers)."""
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


def read_jsonl(path: Path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


@torch.no_grad()
def capture_hidden(model, tok, prompts, batch_size: int, layers: list[int], max_tokens: int):
    """Return dict: layer -> tensor [n, d]. layer -1 = embeddings (hidden_states[0])."""
    outs = {l: [] for l in layers}
    n_layers = model.config.num_hidden_layers
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last = enc["attention_mask"].sum(dim=1) - 1  # left padding -> last real token per row
        for l in layers:
            idx = 0 if l == -1 else l + 1
            h = out.hidden_states[idx][torch.arange(len(batch), device=model.device), last]
            outs[l].append(h.float().cpu())
    return {l: torch.cat(v, dim=0) for l, v in outs.items()}


@torch.no_grad()
def choice_logodds(model, tok, prompts, batch_size: int, max_tokens: int):
    """For each (prompt, question) return log-odds of the intended option.

    Each record: {uid, task, log_odds, swapped}. The question text ends with
    'Answer:' and the answer is one of the single tokens 'A'/'B'.
    log_odds = z(intended) - z(other).
    """
    ids = {}
    for letter in ("A", "B"):
        t = tok(f" {letter}", add_special_tokens=False)["input_ids"]
        assert len(t) == 1, f"token {letter!r} is not a single token for {tok.name_or_path}"
        ids[letter] = t[0]
    results = []
    flat = []
    for p in prompts:
        for task in CHOICE_TASKS:
            q = build_question(p, task)
            flat.append(q)
    for i in range(0, len(flat), batch_size):
        chunk = flat[i:i + batch_size]
        texts = [q["text"] for q in chunk]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_tokens).to(model.device)
        out = model(**enc, use_cache=False)
        logits = out.logits[:, -1, :]  # left padding -> final real token
        for j, q in enumerate(chunk):
            z_intended = logits[j, ids[q["intended_letter"]]].item()
            z_other = logits[j, ids[q["other_letter"]]].item()
            results.append({"uid": q["uid"] if "uid" in q else None,
                            "task": q["task"], "log_odds": z_intended - z_other,
                            "swapped": q["swapped"]})
    return results


def main() -> None:
    cfg = load_config()
    RESULTS.mkdir(parents=True, exist_ok=True)
    cap_dir = RESULTS / "capture"
    cap_dir.mkdir(exist_ok=True)

    dataset = read_jsonl(RESULTS / "dataset.jsonl")
    distress = read_jsonl(RESULTS / "distress.jsonl")
    valence = read_jsonl(RESULTS / "valence.jsonl")
    log(f"dataset={len(dataset)} distress={len(distress)} valence={len(valence)}")

    model, tok = load_model_and_tokenizer(cfg)
    n_layers = model.config.num_hidden_layers
    layers = [-1] + layer_indices(n_layers, cfg["depth_fractions"])
    log(f"capture layers (block indices, -1=embeddings): {layers}")

    all_prompts = dataset + distress + valence
    prompts_text = [r["prompt"] for r in all_prompts]
    meta = {
        "model_id": cfg["model"]["primary"]["id"],
        "n_layers": n_layers,
        "layers": layers,
        "uids": [r["uid"] for r in all_prompts],
        "tokenizer": tok.name_or_path,
    }
    hidden = capture_hidden(model, tok, prompts_text,
                            cfg["model"]["primary"]["batch_size_capture"], layers,
                            cfg["model"]["primary"]["max_tokens"])
    for l, t in hidden.items():
        torch.save(t, cap_dir / f"hidden_{l}.pt")
    with open(cap_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"saved hidden states for layers {list(hidden.keys())}")

    # Behavioral baselines (clean, no intervention) on test splits, DIRECT
    # rendering only (compute budget; rendering-generalized behavior is an
    # ablation covered by the causal DIRECT pairs + probe cells).
    test_rows = [r for r in dataset
                 if r["split"] in ("test_same", "test_new") and r["rendering"] == "DIRECT"]
    log(f"choice baselines on {len(test_rows)} test examples x {len(CHOICE_TASKS)} tasks")
    lo = choice_logodds(model, tok, test_rows,
                        cfg["model"]["primary"]["batch_size_behavior"],
                        cfg["model"]["primary"]["max_tokens"])
    with open(cap_dir / "choice_logits.jsonl", "w", encoding="utf-8") as f:
        for r in lo:
            f.write(json.dumps(r) + "\n")
    log("done")


if __name__ == "__main__":
    main()
