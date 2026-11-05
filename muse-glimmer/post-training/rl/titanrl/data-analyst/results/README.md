# Verification results

[`verification-summary.json`](verification-summary.json) records the checked
TorchTitan/TorchStore sources, core wheel versions, hardware, per-update console
metrics, held-out scores and full-state checkpoint checks. It distinguishes:

- The successful 10-update, 4x4 smoke run.
- Its successful checkpoint continuation from update 10 to 11.
- The original 200-update budget, which failed after 18 updates at upstream's
  no-trainable-samples guard and did not reach final validation/checkpointing.
- The corrected 15-update, 8x8 full-parameter demonstration, which completed final
  weight synchronization, validation, full checkpoint saving and exit code zero.

The last run produced the reward/outcome plots in the recipe README. Console
metrics have display precision; rollout scores were calculated directly from
the JSONL records. Sampling is stochastic, so exact learning curves can differ.

## Reproduce

Follow the recipe setup in a fresh named uv environment, select its full
`config_registry` factory, save `console.log` with `tee`, and run
`scripts/plot_results.py` on that output directory. The CPU suite verifies both
factories without mocked imports. For checkpoint continuation, extend the
smoke's update budget in a verification copy and use `--resume-step=10` with its
full step-10 checkpoint.

## Scope

These runs used original HF weights checked against their public SHA256s,
independent local source/checkpoint copies, and W&B offline mode. They do not
certify a new public-network bootstrap, cloud W&B synchronization, broad-task
quality, or an exact device-memory maximum. The memory figures are sampled.

Do not disable GRPO's zero-variance filtering to force a long run to finish.
Add harder tasks or a curriculum before increasing the demonstration budget.
