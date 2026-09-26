# Post-training

Everything else in this cookbook is about *running* Muse Glimmer. This section
is about *changing* it: taking the open-weights checkpoint and training it
further on your own task, so it gets measurably better at something it is
currently bad at.

There are two training objectives. Use [`sft/`](sft/) when you have target
responses, and use [`rl/`](rl/) when you have a machine-checkable reward. Each
objective contains framework-specific recipes so additional trainers can be
added without mixing their setup, configs, or results.

## Paths

| Objective | Available framework | Recipe |
|---|---|---|
| Supervised fine-tuning | Axolotl | [`sft/axolotl/`](sft/axolotl/) |
| Reinforcement learning | TitanRL | [`rl/titanrl/data-analyst/`](rl/titanrl/data-analyst/) |

The objective indexes compare frameworks and reserve named locations for future
implementations such as LLaMA-Factory or Unsloth. Framework directories own
their configs, tests, assets, and measurements.

## Before you start: choose the training signal

Start with **SFT** when you have examples of the response you want. It is the
direct route for teaching a new output format, tone, domain pattern, or visual
task. The [`sft/axolotl/`](sft/axolotl/) recipe starts with QLoRA, then shows
when to pay the additional memory cost for LoRA or full-parameter tuning.

Use **RL** when the model can already attempt the task and success is cheap to
check, but the behavior is unreliable. RL post-training shapes behavior rather
than supplying missing knowledge. It is a good fit when:

- The model already *can* do the task but does it unreliably (wrong format,
  gives up after an error, never produces the required output).
- Success is **cheaply and objectively checkable** -- a file exists, a number
  matches, a test passes. If you need a human or an LLM to judge each attempt,
  expect noise and reward hacking.
- You have a **measured baseline**, so "better" is a number rather than a
  vibe.

[TitanRL](rl/titanrl/) uses the same TorchTitan model definition in the
trainer and its vLLM
generator. That avoids separate training and serving model implementations that
can silently disagree.

RL is a poor fit for teaching facts the model does not know. Start with SFT for
that job, then add RL only if the remaining failure has an objective reward.

## Hardware reality check

Post-training a 30B dense model is substantially heavier than quantized
inference. The SFT choices trade memory for how much of the model can change:

| SFT method | What stays in memory |
|---|---|
| QLoRA | 4-bit frozen base plus trainable adapters |
| LoRA | bf16 frozen base plus trainable adapters |
| Full FSDP2 | Sharded weights, gradients, and optimizer state |

QLoRA is the default and lowest-memory experiment. Try LoRA when you want an
unquantized frozen base. Use full FSDP2 only when adapters are a measured
capacity bottleneck.

Measured on one 8x H100 node with 512x512 CoSyn examples and a global batch
of 8, bounded smoke runs produced:

| SFT method | GPUs | Peak observed device memory | Trainer result |
|---|---:|---:|---|
| QLoRA | 1 | 26,995 MiB | 2 steps in 25.44 seconds |
| QLoRA DDP | 8 | 27,211 MiB per GPU | 2 steps in 14.89 seconds |
| LoRA FSDP2 | 8 | 32,966 MiB on rank 0 | 2 steps in 30.68 seconds |
| Full FSDP2 | 8 | 57,608 MiB on rank 0 | 3 steps including resume |

These are pipeline checks, not quality benchmarks. Separate controlled runs in
the SFT recipe trained CoSyn QLoRA, bf16 LoRA FSDP2, and full FSDP2 for 64
steps; trained Tulu QLoRA for 64 steps; and trained true multi-image Docmatix
QLoRA for 32 steps. Final validation losses were 0.8655, 0.8968, 0.8717,
0.7973, and 0.7187, respectively. Full FSDP2 peaked at 97,340 MiB per GPU
during final checkpoint consolidation. The [Axolotl SFT results](sft/axolotl/#controlled-quality-runs)
include W&B links, held-out base-versus-tuned metrics, counterfactuals, plots,
and exact plot inputs.

RL additionally keeps a model in the vLLM generator. For full-parameter bf16
training, whether SFT or RL, the parameter-state arithmetic starts here:

Concretely, per training GPU, with the model sharded N ways:

| | Per-GPU at 30B |
|---|---|
| bf16 weights | ~60 GB / N |
| bf16 gradients | ~60 GB / N |
| fp32 Adam state (m + v) | ~240 GB / N |

Adam's state is allocated on the **first** `optimizer.step()`, so memory jumps
sharply between step 1 and step 2 -- a run that looks comfortable during step 1
can still fail immediately after. Size for the post-step-1 number, not what you
see at startup.

Those figures exclude activations, temporary buffers, and the generator used by
RL.

Each recipe's banner states only hardware it was actually verified on.

## Next steps

- [`../recipes/`](../recipes/) -- the inference-side recipes, if you want to
  ship an agent with the stock model first.
- [`../agentic-fundamentals/`](../agentic-fundamentals/) -- the tool-use loop
  and ATEM tool-call format that the TitanRL data-analyst recipe builds on.
- [`rl/titanrl/data-analyst/`](rl/titanrl/data-analyst/) -- the measured RL
  recipe, if you want to train behavior with a deterministic reward.
