# Post-training

Everything else in this cookbook is about *running* Muse Glimmer. This section is
about *changing* it: taking the open-weights checkpoint and training it further on
your own task, so it gets measurably better at something it is currently bad at.

The tool is [**TitanRL**](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl),
the reinforcement-learning stack inside TorchTitan. It runs the same TorchTitan
model definition in the trainer and inside vLLM, so there is one model to reason
about rather than a training copy and a serving copy that can silently disagree.

## Recipes

| Recipe | What it teaches |
|---|---|
| [`data-analyst/`](data-analyst/) | Post-train Muse Glimmer on agentic csv/log analysis with a deterministic, verifiable reward -- no LLM judge, no external dataset. |

## Before you start: is RL the right tool?

RL post-training shapes **behavior**, not knowledge. It is a good fit when:

- The model already *can* do the task but does it unreliably (wrong format,
  gives up after an error, never produces the required output).
- Success is **cheaply and objectively checkable** -- a file exists, a number
  matches, a test passes. If you need a human or an LLM to judge each attempt,
  expect noise and reward hacking.
- You have a **measured baseline**, so "better" is a number rather than a
  vibe.

It is a poor fit for teaching the model facts it does not know. That fights the
30B parameter budget, and fine-tuning on documents is a different (cheaper) tool.

## Hardware reality check

Post-training a 30B dense model is substantially heavier than serving one.
Serving Muse Glimmer quantized fits a single 24-32 GB GPU; RL post-training at
bf16 additionally holds gradients and optimizer state for every parameter, plus a
second copy of the model inside the vLLM generator.

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

Each recipe's banner states the hardware it was actually verified on, and the
TorchTitan commit it was verified against. RL support in TorchTitan moves fast:
a change that landed on 2026-09-13 ([#4535](https://github.com/pytorch/torchtitan/pull/4535))
made 30B Muse Glimmer RL stop fitting on a single 8x95 GB node, including
upstream's own reference config. Recipes here pin a known-good commit and say
why.

## Next steps

- [`../recipes/`](../recipes/) -- the inference-side recipes, if you want to
  ship an agent with the stock model first.
- [`../agentic-fundamentals/`](../agentic-fundamentals/) -- the tool-use loop and
  the ATEM tool-call format, which the data-analyst recipe builds on.
