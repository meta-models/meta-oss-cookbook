# Data Analyst: post-train Muse Glimmer with a verifiable reward

Teach Muse Glimmer to reliably compute a number from a csv or log file **and write
it to the file it was asked for** -- using reinforcement learning with a
deterministic, machine-checkable reward. No LLM judge, no labelled dataset, no
external services.

## Recipe banner

| | |
|---|---|
| Precision | bf16 |
| Model server | vLLM (TitanRL's in-process generator) |
| Offline? | Yes, after the one-time checkpoint download |
| Hardware verified on | 8x H100 95GB (single node) |
| Max VRAM observed | ~88 GB/GPU on the training GPUs (peak, post-optimizer-allocation) |
| Requires | [TorchTitan](https://github.com/pytorch/torchtitan) **pinned at `8108e201a`** + TitanRL deps (Monarch, TorchStore, vLLM, FlashAttention-3) |

> **Measured result.** Over 10 GRPO steps on the public 30B checkpoint, mean
> rollout reward went from **0.20 to 0.88** -- the model learns to run a working
> command and write the answer file. Zero OOMs, zero collective timeouts.
> ([W&B run](https://wandb.ai/a-shamsoshoara-m/titan_rl/runs/bo0fjdsh).) That is
> a short run showing a clear trend, not a convergence study; a longer curve and
> a before/after benchmark comparison are still to come, and this page will not
> claim numbers it has not measured.
>
> **You must pin TorchTitan to `8108e201a`.** On current `main` this recipe --
> and upstream's own `rl_grpo_muse_glimmer_30b_search_r1` -- runs out of memory
> during weight sync. See [Pinning TorchTitan](#pinning-torchtitan).

## The problem this solves

Muse Glimmer scores ~65% overall on PinchBench (a 147-task agentic benchmark) but
only **37% on csv analysis** and **39% on log analysis**. A deep-dive on all 36
failures found something useful: **none of them were analysis errors.** The model
consistently worked out the right answer. What it failed at was the agentic
mechanics around it:

1. It emitted Python with literal `\n` escapes inside a shell heredoc, so the
   shell handed Python invalid syntax and the script never ran.
2. On the `SyntaxError`, it re-sent the *same* broken command instead of trying
   another approach.
3. It ran out of turns without ever writing the output file -- so even runs where
   the correct number appeared in scratch output scored zero.

That is a behavior problem, and behavior is what RL shapes. The reward here is
exactly "did the right number end up in the right file", which is checkable in
microseconds with no judge.

## Quickstart

```bash
# from a TorchTitan checkout with the TitanRL deps installed (see Setup below)
python -m torchtitan.experiments.rl.train \
  --module glimmer_data_analyst \
  --config rl_grpo_muse_glimmer_30b_data_analyst_smoke \
  --dump-folder outputs/rl/glimmer_data_analyst_smoke
```

## Setup

**1. Get TorchTitan and the RL dependencies.**

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git checkout 8108e201a       # required -- see "Pinning TorchTitan" below
pip install uv
uv venv --python 3.12 glimmer-rl
source glimmer-rl/bin/activate

uv pip install -r torchtitan/experiments/rl/requirements.txt
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu130
```

**2. Install torch, torchvision, vllm and torchcomms pinned to ONE nightly date.**
This matters more than it looks -- see [Troubleshooting](#troubleshooting).

```bash
DATE=20260909   # pick a date where all four wheels exist for your platform
uv pip install \
  torch==2.15.0.dev${DATE}+cu130 \
  torchvision==0.30.0.dev${DATE}+cu130 \
  torchcomms==0.3.0.dev${DATE}+cu130 \
  vllm==1.0.0.dev${DATE}+cu130 \
  --extra-index-url https://download.pytorch.org/whl/nightly/cu130 \
  --index-strategy unsafe-best-match
```

**3. Download the open-weights checkpoint** (~56 GB, one time):

```bash
python scripts/download_hf_assets.py \
  --repo_id meta-models/Muse-Glimmer-30B \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --all
```

**4. Point `PYTHONPATH` at the checkout** (Monarch-spawned workers import from it):

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
wandb login          # metrics are on by default; --metrics.no-enable-wandb to skip
```

**If your machine has no InfiniBand**, also:

```bash
export USE_TORCHCOMMS_RDMA=0
```

TorchStore moves trainer->generator weights over RDMA by default and will hang
without it.

## What just happened

Each rollout is one generated task. The model is dropped into a scratch directory
containing a `data.csv` or `access.log`, given a question with a single numeric
answer, and handed exactly one tool: `bash`.

```
prompt   "Count how many lines have level WARN. Write the count to
          answer.json as JSON: {"answer": <number>}."

turn 0   bash: ls -la                                  -> sees access.log
turn 1   bash: head -n 20 access.log                   -> inspects the format
turn 2   bash: grep -c ' WARN ' access.log > /tmp/n && python3 -c "..."
                                                       -> writes answer.json
done     env checks answer.json against the golden value -> reward
```

Four moving parts, one file each:

| File | Role |
|---|---|
| `data.py` | Generates the tasks. Four templates (csv sum, csv group-by-max, log count, log top-status), each with the golden answer computed at generation time. Endless, seeded, and **fully offline** -- there is no dataset to download. Train and validation draw from disjoint index ranges. |
| `env.py` | The `bash` tool and a private scratch directory per rollout. Grades the workspace after every command. |
| `rubric.py` | Turns those checks into a reward. |
| `config_registry.py` | GRPO/DAPO hyperparameters and the GPU split. |

**Why `bash` and not a clean `run_python(code=...)` tool.** The bug being fixed
lives in the shell boundary -- broken heredoc quoting. A tool that took code as a
JSON string would route around the bug rather than teach the model to get it
right, and the fix wouldn't transfer to a real terminal-using agent.

**The reward is graded, not pass/fail:**

| Outcome | Reward |
|---|---|
| Correct number in `answer.json` | **1.0** |
| Well-formed file, wrong number | 0.25 |
| File written but malformed | 0.15 |
| At least one command ran successfully | 0.05 |
| Nothing | 0.0 |
| *(minus)* each repeat of an already-failed command | −0.05 |

The tiers are not decoration. GRPO learns from *differences between rollouts of
the same prompt*: if every attempt scores 0 because none fully succeeds, the group
has no variance, the batch is discarded, and the model never gets a gradient. In
early testing a bare 0/1 reward produced 12-18 consecutive discarded batches. The
tiers make a partially-competent attempt outrank a flailing one -- while keeping
correctness strictly dominant, so partial credit can never beat a right answer.

## Make it yours

**Swap in your own task.** Add a template to `data.py`. It needs to return a
`GlimmerDataAnalystSample`: the prompt, the input filename and contents, and the
golden value. Everything else -- env, reward, config -- works unchanged. This is
the cheapest way to point the recipe at a task you care about.

**Change what gets rewarded.** The tiers in `rubric.py` are config fields. Set
`file_exists_credit=0.0` and `file_parses_credit=0.0` for a strict 0/1 reward, or
raise `repeated_failing_command_penalty` to punish thrashing harder.

**Tune the rollout budget** in `rollouter.py`. If you widen `max_num_turns` or the
generator's `max_tokens`, raise `max_rollout_tokens` and the trainer's
`max_context_length` to match -- a rollout that outgrows its budget is killed as
`truncated_prompt_too_long`, and that silently looks like the model failing.

## Pinning TorchTitan

This recipe pins TorchTitan to commit `8108e201a`. That is not superstition --
on commits after it, 30B Muse Glimmer RL does not fit on a single 8x95 GB node.

TorchTitan [#4535](https://github.com/pytorch/torchtitan/pull/4535) (2026-09-13)
made the fused gate-up projection the default and added a `state_dict` save hook
that unflattens the fused `w13` parameter back into `w1`/`w3`. RL calls
`state_dict()` on **every weight sync**, so at 30B that materializes a full-size
extra copy of the projection every step. The trainer peaks at ~97 GB of a 95 GiB
card and the next collective either OOMs or hangs until NCCL's watchdog aborts
the run.

This is not specific to this recipe. Upstream's own
`rl_grpo_muse_glimmer_30b_search_r1` -- documented as running 100 steps on
8x H100 -- fails the same way on current `main`, and runs clean at `8108e201a`:

| Config | Commit | Result |
|---|---|---|
| stock `..._search_r1` | current `main` | OOM in `_split_w13_on_save` |
| stock `..._search_r1` | `8108e201a` | 3/3 steps, trainer peak ~88 GB |
| this recipe | current `main` | hang at step 2 |
| this recipe | `8108e201a` | 10/10 steps, reward 0.20 -> 0.88 |

`8108e201a` is the commit immediately before #4535, so it still includes
[#4145](https://github.com/pytorch/torchtitan/pull/4145)'s Muse Glimmer renderer
and everything else that landed in between.

If you are reading this after the issue has been fixed or gated upstream, drop
the pin and use `main`.

### Sizing, if you are adapting this

Adam's fp32 `m`/`v` are allocated on the **first** `optimizer.step()`, so per-GPU
memory jumps sharply between step 1 and step 2 -- a run that looks comfortable
during step 1 can still die right after. Size for the post-step-1 number.

The trainer's FSDP degree must divide the projection dimension 19968
(= 2^9 x 3 x 13), so usable degrees are {1, 2, 3, 4, 6}. With the generator
holding 2 GPUs, this recipe uses FSDP=3 x TP=2 across the remaining 6.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| OOM in `_split_w13_on_save`, or a silent hang at step 2 ending in a NCCL watchdog abort | TorchTitan #4535's state-dict hook doubles the gate-up projection on every weight sync | Pin to `8108e201a` (see [Pinning TorchTitan](#pinning-torchtitan)) |
| `vllm==0.2.5` tries to build from source, fails on CUDA version mismatch | Dependency resolution picked an old PyPI vLLM instead of the nightly wheel | Pin all four torch-family packages to the same explicit `.dev<date>+cu130` version, as in Setup step 2 |
| Weight sync hangs; `CtranIb: Found 0 InfiniBand device(s)` | TorchStore defaults to an RDMA transport | `export USE_TORCHCOMMS_RDMA=0` before launching |
| `Cannot unflatten unevenly sharded tensor` at startup | FSDP degree does not divide the projection dim 19968 | Use an FSDP degree in {1, 2, 3, 4, 6} |
| `num_tokens_per_microbatch_per_dp_rank must be divisible by max_context_length` | The two are set independently | Make the microbatch budget a multiple of the context length |
| Most rollouts end `truncated_prompt_too_long` | Per-turn generation budget x turns exceeds the rollout token budget | Shrink `sampling.max_tokens` / `max_num_turns`, or raise `max_rollout_tokens` and the context length together |
| Repeated `Consecutive untrainable batches` warnings | Every rollout in the group scored identically, so GRPO has no signal | Use a graded reward (see above); check whether rollouts are being truncated before they can succeed |
| `ModuleNotFoundError` for `grain`, `datasets`, `tyro`, `tensorboard`, `torch_remat` | Transitive TorchTitan deps not pulled by the RL requirements file | `uv pip install grain datasets tyro tensorboard torch-remat spmd_types torchdata` |

## Next steps

- [`../../agentic-fundamentals/`](../../agentic-fundamentals/) -- the tool-use loop
  and ATEM tool-call format this recipe trains against.
- [TitanRL docs](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl)
  -- the framework: rollouters, environments, rubrics, and the async controller.
- [`search_r1`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl/examples/search_r1)
  -- the multi-turn retrieval recipe this one is modeled on, if your task needs an
  external tool service rather than a local sandbox.
