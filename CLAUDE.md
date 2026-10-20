# OWL replication (Chris's interview prep)

## What this is
A replication of "Readable but Not Causal" by Vishwa Kumaresh, 1st place at Apart Research's
Digital Minds Research Sprint (Aug 2026). Upstream code: github.com/Vishwa-docs/owl-benchmark.
Chris is preparing for a Director of Research Programs interview at Apart. The point of this
project is that he can explain every step he ran, not just get a number.

## Goal
1. Reproduce the probe result on Qwen3-8B: AUROC 0.859 for the assistant's own outcome (S)
   vs 0.601 for a represented other (O). Skip the causal patching unless there's time.
2. Change one variable: rerun the probe on a model from a different family (the reviewers'
   main ask). Note that `secondary` in configs/model.yaml is Qwen2.5-7B-Instruct, the SAME
   family, so it does not answer the reviewers. Pick a non-Qwen model of similar size.
3. Write a one-page note: what was run, what came out, how it compares, and how far a team
   could get in 48 hours.

## How to work with Chris
- He's an experienced technical PM, new to interpretability. Before running each step,
  explain in plain language what it does and why. After it runs, explain what the output means.
- Keep a running log in notes/RUNLOG.md: date, machine, model, command, result, cost.
- Never edit configs/experiment.yaml: it's the preregistered design. Change models only with
  env vars (WOE_MODEL, WOE_DTYPE, WOE_BATCH_CAPTURE, WOE_BATCH_BEHAVIOR).
- Ask before spending money (GPU rental) or before anything that takes over 30 minutes.

## Where things run
- Mac: unit tests, synthetic pipeline test, and the smoke test on Qwen3-0.6B
  (`bash setup_mac.sh`). An 8B model is not practical here.
- Rented GPU (24GB+ VRAM; an A100 is what the author timed): the real runs.
  - `bash run_probe_only.sh` runs steps 0-3 (dataset, capture, probes) and prints S/O/A
    AUROCs. This is enough for goal 1.
  - `bash run_primary.sh` is the full pipeline, which the author times at about 2.5 hours
    on an A100 and 5-6 hours on an RTX 4090.
- For a new model, first run `WOE_MODEL=<id> python smoke_test.py` to confirm the choice
  tokens 'A'/'B' are single tokens for that tokenizer. Gated models need a Hugging Face login
  and license acceptance.

## Pipeline map
data/generate_dataset.py (deterministic prompts) -> data/validate_dataset.py ->
src/capture.py (hidden states at layers by depth fraction, plus choice log-odds) ->
eval/predictive.py (linear probes for S/O/A, baselines; picks a layer) ->
eval/causal.py (interchange patching + steering) -> eval/monitor.py -> figures/make_figs.py.
Results land in results/ (gitignored). Probe numbers: results/predictive/probe_results.json,
cells[chosen_layer][test_same | test_new]. test_new = held-out scenario families.

## Local changes from upstream (uncommitted, see `git diff`)
- run_primary.sh: skip ../paper/make_tables.py, which isn't in the public repo
  (it only builds LaTeX tables; results are in JSON).
- tests/synthetic_pipeline_test.py: same skip, plus a fix for a crash when printing
  monitor results.
- Added setup_mac.sh, run_probe_only.sh, SETUP.md, this file, notes/.

## Verified on Oct 19, 2026 (cloud sandbox, CPU, no model)
Unit tests pass, the dataset validates (4,800 factorial rows), and the synthetic pipeline
test passes end to end after the fixes above. Not yet verified: anything with a real model.
