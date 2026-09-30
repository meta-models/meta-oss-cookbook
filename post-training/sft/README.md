# Supervised fine-tuning

Supervised fine-tuning (SFT) teaches Muse Glimmer from demonstrations: a prompt
and the response the model should learn to produce. Choose a framework below,
then follow that framework's setup, data, training, and evaluation workflow.

## Frameworks

| Framework | Status | Recipe |
|---|---|---|
| Axolotl | Available and measured | [`axolotl/`](axolotl/) |
| LLaMA-Factory | Planned | Add under `llama-factory/` when implemented |
| Unsloth | Planned | Add under `unsloth/` when implemented |

The Axolotl recipe covers text-only, single-image, and true multi-image data. It
includes QLoRA, bf16 LoRA with FSDP2, and full-parameter bf16 FSDP2 on one 8x
H100 node, with W&B histories, held-out evaluation, and reproducible plots.

## Choose a framework

Start with Axolotl when you need the measured Muse Glimmer paths already in this
cookbook. Add another framework when it offers a concrete advantage such as a
different hardware target, training backend, memory profile, or user workflow.
Keep dataset manifests and held-out IDs compatible when comparing frameworks so
the result measures the framework rather than a data change.

Framework directories own their configs, scripts, tests, assets, and measured
results. Do not add an `other-methods/` catch-all. Give each implemented
framework a named directory and an index entry here.

## Related paths

- [`../`](../) chooses between SFT and RL.
- [`../rl/`](../rl/) contains reinforcement-learning frameworks and recipes.
