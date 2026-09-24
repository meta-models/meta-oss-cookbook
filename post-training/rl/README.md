# Reinforcement learning

Reinforcement learning (RL) improves behavior using a reward signal rather than
a reference response. Use it when Muse Glimmer can already attempt the task and
success is cheap and objective to verify.

## Frameworks

| Framework | Status | Recipes |
|---|---|---|
| TitanRL | Available and measured | [`titanrl/`](titanrl/) |
| Unsloth | Planned | Add under `unsloth/` when implemented |

TitanRL currently provides the data-analyst recipe, which trains tool-using
behavior against deterministic CSV and log-analysis checks.

## Before choosing RL

RL is a good fit when:

- The model already can do the task but behaves unreliably.
- A file, number, schema, or test can determine success.
- A measured baseline exists.

If the model lacks the target knowledge or response pattern, start with
[`../sft/`](../sft/) and add RL only for the remaining measurable behavior.

Framework directories own framework-specific setup and shared guidance.
Task-specific recipes live below the framework, such as
`titanrl/data-analyst/`. Add future frameworks by name instead of using an
`other-methods/` directory.
