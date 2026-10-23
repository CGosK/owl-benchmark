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
  env vars (WOE_MODEL, WOE_DTYPE, WOE_BATCH_CAPTURE, WOE_BATCH_BEHAVIOR, WOE_DEVICE_MAP).
- Ask before spending money (GPU rental) or before anything that takes over 30 minutes.

## Where things run
- Mac (M1 Max, 32 GB): unit tests, synthetic pipeline test, the Qwen3-0.6B smoke test
  (`bash setup_mac.sh`), and the probe runs for 7-8B models. Set
  `WOE_DEVICE_MAP=mps HF_DEACTIVATE_ASYNC_LOAD=1`: without the first, part of the model is
  offloaded to disk (5x slower); without the second, transformers 5.x segfaults loading onto
  MPS. Capture takes about 30 min per model. Qwen3-8B's A/B choice step outgrows 32 GB and
  swaps; it is only needed for the causal step, so stop capture.py once hidden states are saved.
- Rented GPU (24GB+ VRAM; an A100 is what the author timed): the causal step and the full
  pipeline. `bash run_primary.sh` is the full pipeline, which the author times at about
  2.5 hours on an A100 and 5-6 hours on an RTX 4090.
- `bash run_probe_only.sh` runs steps 0-3 (dataset, capture, probes) and prints S/O/A AUROCs.
- For a new model, `smoke_test.py` does NOT test it: it overwrites WOE_MODEL with the 0.6B
  smoke model. Check instead that ' A' and ' B' are single tokens for the new tokenizer and
  that the model loads on MPS (the scratch pilot did this). Gated models need a Hugging Face
  login and license acceptance.

## Pipeline map
data/generate_dataset.py (deterministic prompts) -> data/validate_dataset.py ->
src/capture.py (hidden states at layers by depth fraction, plus choice log-odds) ->
eval/predictive.py (linear probes for S/O/A, baselines; picks a layer) ->
eval/causal.py (interchange patching + steering) -> eval/monitor.py -> figures/make_figs.py.
Results land in results/ (gitignored). Probe numbers: results/predictive/probe_results.json,
cells[chosen_layer][test_same | test_new]. test_new = held-out scenario families.

## Changes from upstream (committed on the fork, CGosK/owl-benchmark)
- run_primary.sh, tests/synthetic_pipeline_test.py: skip ../paper/make_tables.py, which
  isn't in the public repo; fix a crash when printing monitor results.
- src/config.py, src/capture.py: opt-in WOE_DEVICE_MAP (unset = upstream behaviour).
- src/capture.py: BUG FIX (Oct 23). Upstream took `attention_mask.sum() - 1` as the last
  token under LEFT padding, which only holds for unpadded rows: 62-69% of prompts were read
  at the wrong token, almost all 1-4 tokens early (inside "End of status update."), and the
  offset leaks the labels. Now
  reads the true last token. Results from the old index are in results/qwen3-8b and
  results/mistral-7b-instruct-v0.3 ("paper pipeline"); fixed ones in results/*-fixed.
- eval/gap_bootstrap.py: paired cluster bootstrap for O and the S - O gap.
- Added setup_mac.sh, run_probe_only.sh, SETUP.md, this file, notes/.

## Status (Oct 23, 2026)
Goals 1 and 2 done on the Mac for Qwen3-8B and Mistral-7B-Instruct; see notes/RUNLOG.md.
Write-up: "Whose Outcome, Which Model" (claude.ai artifact), filed in Apart Prep. Causal
step and monitor not run.
