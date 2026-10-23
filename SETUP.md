# Running this replication in Claude Code

Everything below happens on your Mac except step 4, which needs a rented GPU.
Use your personal Claude account and GitHub identity here, not a client's.

## 1. One-time tools (Terminal)

    brew install gh uv        # skip any you already have
    gh auth login             # sign in to your personal GitHub
    git config user.email     # check it's your personal address; set it in this repo if not:
                              #   git -C ~/projects/owl-benchmark config user.email "you@example.com"

## 2. Start Claude Code in the project

    cd ~/projects/owl-benchmark
    claude

## 3. Fork, commit, set up: paste this into Claude Code

> Read CLAUDE.md and SETUP.md. Then do the following one step at a time, explaining each
> step in plain language before you run it and what the output means after:
> 1. Fork Vishwa-docs/owl-benchmark to my GitHub with `gh repo fork --remote`, so `origin`
>    is my fork and `upstream` is the original. Show me `git remote -v`.
> 2. Show me `git diff`, then commit the local changes (two fixes plus the setup files)
>    with a message saying what changed and why, and push to my fork.
> 3. Run `bash setup_mac.sh`. If anything fails, diagnose it with me before changing code.
> 4. When the smoke test passes, walk me through what each stage did, using its output.
> 5. Start notes/RUNLOG.md with today's entry, then commit and push it.

Done when: your fork is on GitHub with your changes, and the smoke test passed.

## 4. The real run

**On the Mac (what was actually done, Oct 22-23).** A 32 GB Apple Silicon Mac runs the
probe-only replication for a 7-8B model in about 35 minutes, at no cost:

    WOE_DEVICE_MAP=mps HF_DEACTIVATE_ASYNC_LOAD=1 caffeinate -i bash run_probe_only.sh

Qwen3-8B's A/B choice step (inside capture.py, after hidden states are saved) outgrows
32 GB; stop it once the log says "saved hidden states" and run `python eval/predictive.py`.
See notes/RUNLOG.md for the full record.

**On a rented GPU** (needed for the causal step and the full pipeline). Rent one GPU with at least 24 GB of memory from any cloud GPU provider (the author timed
an A100; an RTX 4090 works but is slower), using a PyTorch image. Then in Claude Code:

> I've rented a GPU. The SSH command is: <paste it>. Help me run the probe-only
> replication there: clone my fork, install requirements, run the smoke test, then
> `bash run_probe_only.sh` and save the log. Copy results back into
> results/qwen3-8b/ on my Mac, add the numbers to notes/RUNLOG.md, then remind me to
> shut the GPU down.

Or by hand, on the GPU machine:

    git clone https://github.com/<your-username>/owl-benchmark.git && cd owl-benchmark
    pip install -r requirements.txt
    python smoke_test.py
    mkdir -p results && bash run_probe_only.sh 2>&1 | tee results/probe_run.log

And on your Mac, to copy the results back:

    scp -r <gpu-host>:owl-benchmark/results ~/projects/owl-benchmark/results/qwen3-8b

Shut the GPU down when you're finished; most providers bill until you do.

## 5. Change one variable

> Pick an ungated, non-Qwen instruct model of about 7-8B that loads with transformers.
> Confirm ' A' and ' B' are single tokens for its tokenizer and that it loads on the Mac
> GPU (smoke_test.py can't do this: it overwrites WOE_MODEL with the 0.6B smoke model),
> then run run_probe_only.sh with WOE_MODEL set. Save results under
> results/<model-name>/ and log it in notes/RUNLOG.md.

Done Oct 22 with mistralai/Mistral-7B-Instruct-v0.3.

## 6. Write it up

> Draft my one-page replication note from notes/RUNLOG.md and the results JSON: what I
> ran, my numbers vs the paper's, what changed with the second model, and how far a team
> could get in 48 hours. Keep it plain and short; I'll edit it.

Paste the final version into the Replication note section of Apart Prep.

Note: results/ is gitignored, so model outputs stay on your machine. Commit RUNLOG.md
and the write-up, not the raw results.
