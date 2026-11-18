# Supervised fine-tuning

Supervised fine-tuning (SFT) teaches Muse Glimmer from demonstrations: a prompt
and the response the model should learn to produce. Choose a framework below,
then follow that framework's setup, data, training, and evaluation workflow.

## Frameworks

| Framework | Status | Recipe |
|---|---|---|
| Axolotl | Available and measured | [`axolotl/`](axolotl/) |
| TRL | Available and measured | [`trl/`](trl/) |
| Unsloth | Available and measured | [`unsloth/`](unsloth/) |

The Axolotl recipe covers text-only, single-image, and true multi-image data. It
includes QLoRA, bf16 LoRA with FSDP2, and full-parameter bf16 FSDP2 on one 8x
H100 node, with W&B histories, held-out evaluation, and reproducible plots.

The TRL recipe trains QLoRA adapters with `SFTTrainer` on the same CoSyn circuit
data as the Axolotl QLoRA run, on one 8x H100 node. It scores the base model and
the adapter on all 821 held-out questions in the same run.

The Unsloth recipe runs the same task as the TRL recipe on a single GPU:
Unsloth loads the 4-bit model and builds the batches, and TRL's `SFTTrainer`
runs the training loop. It also scores the base model and the adapter on all
821 held-out questions in the same run.

## Choose a framework

Start with Axolotl for text-only or multi-image data, bf16 LoRA, or
full-parameter training. Use TRL for QLoRA across the GPUs of one node with
standard Hugging Face components, and Unsloth for QLoRA on a single GPU. The TRL
and Unsloth recipes share their data, prompt and held-out questions, so their
scores compare directly. Add another framework when it offers a concrete
advantage such as a different hardware target, training backend, memory
profile, or user workflow. Keep dataset manifests and held-out IDs compatible
when comparing frameworks so the result measures the framework rather than a
data change.

Framework directories own their configs, scripts, tests, assets, and measured
results. Do not add an `other-methods/` catch-all. Give each implemented
framework a named directory and an index entry here.

## Related paths

- [`../`](../) chooses between SFT and RL.
- [`../rl/`](../rl/) contains reinforcement-learning frameworks and recipes.
