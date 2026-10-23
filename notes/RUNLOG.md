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
