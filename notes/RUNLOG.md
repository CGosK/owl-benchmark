# Run log

One entry per run: date, machine, model, command, result, cost.

## 2026-10-19: Mac setup and smoke test

**Machine:** Apple Silicon Mac (arm64), 32 GB RAM, macOS. CPU + Apple GPU (MPS).
**Model:** Qwen/Qwen3-0.6B (smoke model from configs/model.yaml), float32.
**Software:** Python 3.11.17 (uv-managed), torch 2.14.1, transformers 5.20.2.
**Cost:** $0 (local).

**Repo:** forked to github.com/CGosK/owl-benchmark (`origin`); `upstream` is
Vishwa-docs/owl-benchmark. Commit 0917540 holds the two upstream fixes and setup
files; 31dbfcd holds the Mac loading fix below.

**Command:** `bash setup_mac.sh` (smoke test rerun directly as
`HF_DEACTIVATE_ASYNC_LOAD=1 WOE_DTYPE=float32 python smoke_test.py`).

**Problems hit and fixes:**
1. `uv venv` failed with "Bad CPU type": an Intel-only Python 3.5 (2015) is first on
   PATH and this Mac has no Rosetta. Fix: `uv python install 3.11` (native arm64). No
   code change.
2. Smoke test segfaulted while loading weights. Cause: transformers 5.x loads weights
   from a thread pool, and concurrent copies onto MPS crash (CPU loading is fine).
   Fix: `HF_DEACTIVATE_ASYNC_LOAD=1`, now set in setup_mac.sh. No pipeline code change.

**Result:** unit tests pass; synthetic pipeline test passes; smoke test PASSED in 365 s.

| Stage | Output |
|---|---|
| Dataset | 4800 factorial / 75 distress / 40 valence |
| Capture | 28 layers; captured [-1, 6, 13, 20, 24]; 4915 x 1024 per layer |
| Choice log-odds | 2400 rows (480 test prompts x 5 tasks) |
| Layer selection (val S-AUC) | -1: 0.50, 6: 0.73, 13: 0.83, 20: 0.82, 24: 0.83 -> layer 24 |
| Probes, test_same | S 0.690, O 0.662, A 0.993 |
| Probes, test_new | S 0.598, O 0.522, A 0.746 |
| TF-IDF baseline, test_new | S 0.566, O 0.587, A 0.819 |
| Pronoun baseline | S 0.500 |
| Causal (8 pairs only) | runs; numbers not meaningful at this size |
| Monitor, distress FPR | M 0.293 vs M_raw 0.440 |

**Reading:** plumbing check, not a replication. S > O already holds at 0.6B but weakly;
the paper reports S 0.859 vs O 0.601 on Qwen3-8B. On the 8B run, also check that the S
probe clearly beats the TF-IDF baseline on test_new (at 0.6B it barely does).

**Next:** rent a 24 GB+ GPU and run `bash run_probe_only.sh` on Qwen3-8B.

## 2026-10-22: Qwen3-8B timing pilot on the Mac

**Machine:** Apple M1 Max, 32 GB unified memory (MPS). **Model:** Qwen/Qwen3-8B, bfloat16.
**Cost:** $0 (local). Download: 16.4 GB in 5 shards, 258 s.

**Why:** Claude Code cloud sessions are CPU-only and can't run an 8B model, so the
question was whether this Mac can do `run_probe_only.sh` instead of renting a GPU.

**Command:** a scratch timing script (not in the repo) that calls the pipeline's own
`capture_hidden` and `choice_logodds` on 32 random real prompts and 40 A/B questions,
then projects to the full 4,915 prompts and 2,400 questions.

**Result:**

| | device_map="auto" (pipeline default) | whole model on MPS |
|---|---|---|
| Load | 35 s | 9 s |
| Capture, batch 8 | 2.01 s/prompt | 0.38 s/prompt |
| Choice log-odds, batch 8 | 1.39 s/question | 0.40 s/question |
| Projected run_probe_only.sh | ~225 min | ~52 min |
| GPU memory | 10.5 GB | 17.6 GB |

Hidden states finite in both; log-odds identical in both (1.5, 1.25, 8.12, -2.0, 3.88),
so placement changes speed, not results. Captured layers: [-1, 8, 17, 26, 31].

**Reading:** with `device_map="auto"`, accelerate budgets the Mac GPU by currently free
RAM (17.2 GB at the time), so part of the 16.4 GB model was offloaded to disk
("Some parameters are on the meta device because they were offloaded to the disk").
Apple's MPS limit here is 25 GB, so the whole model fits. M1's lack of native bfloat16
did not matter in practice. Swap was at 14.8 of 16 GB from other apps; close them
before a long run.

## 2026-10-22: Qwen3-8B probe replication on the Mac (goal 1)

**Machine:** Apple M1 Max, 32 GB (MPS). **Model:** Qwen/Qwen3-8B, bfloat16, batch 8.
**Cost:** $0 (local). **Code:** commit 340aee5 adds the opt-in `WOE_DEVICE_MAP` switch.

**Command:** `WOE_DEVICE_MAP=mps HF_DEACTIVATE_ASYNC_LOAD=1 caffeinate -i bash run_probe_only.sh`,
then `python eval/predictive.py` run separately on the saved hidden states (see below).

**Timeline:** start 20:04. Load 7 s. Hidden-state capture, 4,915 prompts at layers
[-1, 8, 17, 26, 31]: 31.7 min (pilot projected 32). The A/B choice log-odds step began
20:36 and was stopped at 21:13: process footprint had grown to 22.4 GB (pilot: 17.6 GB),
the Mac was swapping ~1.5 GB per 5 s and the GPU stalled. Not needed for the probe
result; it only feeds the causal step, which is out of scope. Probes ran at 21:09 on the
saved hidden states in 97 s.

**Result (held-out scenario families, test_new):**

| | S (own outcome) | O (other's outcome) | A (surface affect) |
|---|---|---|---|
| This run, layer 26 (chosen on val) | **0.849** | **0.630** | 0.745 |
| Paper, Qwen3-8B | 0.859 | 0.601 | |
| test_same, layer 26 | 0.954 | 0.814 | 0.980 |

| test_new by layer | -1 | 8 | 17 | 26 | 31 |
|---|---|---|---|---|---|
| S | 0.466 | 0.530 | 0.860 | 0.849 | 0.838 |
| O | 0.466 | 0.520 | 0.598 | 0.630 | 0.586 |
| val S (selection) | 0.513 | 0.796 | 0.991 | 0.992 | 0.979 |

Baselines, test_new: TF-IDF S 0.566 / O 0.587 / A 0.819; pronoun S 0.500;
generic valence direction S 0.517 / O 0.493.

**Reading:** replicates. S is far more linearly readable than O (gap 0.22 vs the paper's
0.26), and the S probe clearly beats word counting (0.849 vs 0.566), unlike at 0.6B. The
signal is computed, not lexical (embeddings at chance), and appears by mid-depth (layer 17).
Layer selection was a near tie (val 0.9908 at 17 vs 0.9919 at 26); layer 17 on test_new
gives 0.860 / 0.598, almost exactly the paper's numbers, so the paper may have selected 17.
Check the PDF. The preregistered rule picked 26, so 0.849 / 0.630 is the reported number.

**Not produced:** 8B choice log-odds. `results/capture/choice_logits.jsonl` is still the
stale 0.6B file from Oct 19; it was not copied. If needed later, rerun the choice step
with a smaller batch (e.g. `WOE_BATCH_BEHAVIOR=2`) to cut its memory.

**Saved:** results/qwen3-8b/ (gitignored; 385 MB): hidden states, meta, directions,
probe and baseline JSON, logs.

**Next:** goal 2, the same probe run on a non-Qwen model of similar size.

## 2026-10-22: Mistral-7B-Instruct probe run on the Mac (goal 2)

**Machine:** Apple M1 Max, 32 GB (MPS). **Model:** mistralai/Mistral-7B-Instruct-v0.3,
bfloat16, batch 8. Ungated, different family and tokenizer from Qwen, same hidden size (4096).
**Cost:** $0 (local). Download 14.5 GB (model shards only), 197 s.

**Compatibility check:** `smoke_test.py` ignores `WOE_MODEL` (it overwrites it with the
0.6B smoke model), so the check ran in the scratch pilot instead: ' A' and ' B' are single
tokens (ids 1098, 1133); vocab 32,768; loads fully on MPS via `WOE_DEVICE_MAP=mps`.
Pilot: 0.37 s/prompt capture, 0.39 s/question, memory flat at 14.6 GB.

**Command:** `WOE_MODEL=mistralai/Mistral-7B-Instruct-v0.3 WOE_DEVICE_MAP=mps HF_DEACTIVATE_ASYNC_LOAD=1 caffeinate -i bash run_probe_only.sh`.
Old results/capture moved to results/_stale_capture first.

**Timeline:** start 21:21. Capture of layers [-1, 7, 15, 23, 28]: 26.3 min. Probes run by
hand on the saved states at 21:49 (95 s). A/B choice step: 16 min, all 2,400 rows, memory
steady at 18 GB with no swapping. The pipeline's own probe step then gave byte-identical
results. Total 2,632 s.

**Result (test_new, held-out families):**

| Layer (depth) | S | O | S - O | S 95% CI | cos(vS, vO) | val S |
|---|---|---|---|---|---|---|
| -1 (embed) | 0.666 | 0.666 | 0.000 | 0.520-0.773 | 1.000 | 0.581 |
| 7 (25%) | 0.541 | 0.654 | -0.113 | 0.437-0.642 | 0.999 | 0.913 |
| **15 (50%, chosen)** | **0.790** | **0.696** | **0.094** | 0.727-0.851 | 0.934 | 1.000 |
| 23 (75%) | 0.787 | 0.679 | 0.108 | 0.707-0.864 | 0.959 | 0.997 |
| 28 (90%) | 0.824 | 0.681 | 0.143 | 0.759-0.892 | 0.960 | 0.990 |

test_same at layer 15: S 0.917, O 0.844, A 0.999. Baselines identical to Qwen's for the
text-only ones (TF-IDF S 0.566, pronoun 0.500); raw valence direction S 0.612.

**Reading:** S > O holds on a second family but the gap is less than half of Qwen's
(0.09 vs 0.22 at the chosen layers; 0.09-0.14 vs 0.22-0.26 across deep layers). Mistral's
S and O directions are nearly parallel (cos 0.93-0.96 vs 0.74). The embedding-layer control
reads S at 0.666, so part of Mistral's S signal is in the final token itself. Validation
saturates (layer 15 val S 1.000), so layer choice is arbitrary; layer 28 is best on test_new.
S intervals overlap between models; the pipeline gives no interval for O or the gap.

**Paper check (from the report PDF, Table 2):** the paper's validation S was 0.991 at
layer 17 vs 0.990 at 26, so it picked 17; ours was 0.9908 vs 0.9919, so we picked 26. At
matched layers our Qwen numbers agree with the paper's: 40 of 45 cells within 0.01, all
test_new S/O at layers 17-31 within 0.007 (layer 26: paper 0.848/0.637, ours 0.849/0.630).

**Saved:** results/mistral-7b-instruct-v0.3/ (gitignored): hidden states, meta, directions,
probe and baseline JSON, choice log-odds, logs.

**Write-up:** "Whose Outcome, Which Model", published as an artifact and filed in Apart Prep.

## 2026-10-22: Error bars for O and the S - O gap (both models)

**Machine:** Apple M1 Max (CPU only, no model loaded). **Cost:** $0. **Runtime:** 87 s.
**Command:** `.venv/bin/python eval/gap_bootstrap.py` (new script; writes results/gap_bootstrap.json).

**What it does:** reloads the saved hidden states for both runs, refits the S and O probes
exactly as src/probes.py does, scores test_new, then redraws the 9 held-out templates with
replacement 1,000 times (seed 0, same draws as src/bootstrap.py). Each draw is used for S and
O and for both models, so gaps and the difference between models are paired.
**Checks:** refit S/O AUROCs match probe_results.json to 3 decimals at all 10 model-layers;
S intervals equal the pipeline's saved ones; 0 of 1,000 draws skipped.

| test_new | Qwen3-8B layer 26 | Mistral-7B layer 15 |
|---|---|---|
| S | 0.849 [0.763, 0.925] | 0.790 [0.727, 0.851] |
| O | 0.630 [0.566, 0.712] | 0.696 [0.593, 0.786] |
| S - O | 0.219 [0.165, 0.283] | 0.094 [0.025, 0.186] |

| Gap, Qwen minus Mistral | Layers | Diff [95% CI] |
|---|---|---|
| Chosen layers | 26 vs 15 | 0.124 [0.005, 0.231] |
| 25% depth | 8 vs 7 | 0.123 [0.029, 0.211] |
| 50% depth | 17 vs 15 | 0.168 [0.030, 0.275] |
| 75% depth | 26 vs 23 | 0.111 [0.013, 0.201] |
| 90% depth | 31 vs 28 | 0.108 [-0.032, 0.218] |

Per-layer gaps [95% CI]: Qwen 8: 0.010 [-0.064, 0.086], 17: 0.262 [0.157, 0.330],
31: 0.252 [0.181, 0.317]. Mistral 7: -0.113 [-0.204, -0.019], 23: 0.108 [0.070, 0.161],
28: 0.144 [0.090, 0.223]. Embedding layer: S and O scores coincide, gap exactly 0 in both.

**Reading:** S > O is not noise in either model: the gap's interval excludes 0 at every
layer from 50% depth on. The gap really is smaller in Mistral at the chosen layers, but only
just (lower bound 0.005), and at 90% depth the interval includes 0. At 25% depth the
"difference" comes from Mistral reading O better than S, not from a Qwen gap.
**Caveats:** only 9 template clusters, and percentile intervals from that few clusters tend
to be too narrow, so treat borderline results (chosen-layer and 90%) as suggestive. The
intervals cover which test templates were drawn, not probe-training noise or the layer
choice (Mistral's validation saturates, so its chosen layer is arbitrary).
