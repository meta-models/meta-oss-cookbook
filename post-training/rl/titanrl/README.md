# TitanRL

TitanRL is TorchTitan's reinforcement-learning stack. It uses the same model
definition in the trainer and vLLM generator, reducing the risk that training
and serving implementations disagree.

## Recipes

| Recipe | What it teaches |
|---|---|
| [`data-analyst/`](data-analyst/) | Deterministic CSV and log analysis with GRPO |

The data-analyst recipe includes a measured baseline, reward design, training
configuration, rollout analysis, and reproducible implementation files.

## Related paths

- [`../`](../) compares RL frameworks.
- [`../../sft/`](../../sft/) contains supervised fine-tuning frameworks.
- [TitanRL upstream](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl)
