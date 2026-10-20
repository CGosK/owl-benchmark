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
