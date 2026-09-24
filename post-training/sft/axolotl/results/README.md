# Measured result inputs

This directory contains the exact compact inputs used to render the figures in
the SFT README. It does not contain model weights, adapters, prepared images, or
raw generated responses.

- `histories/*.json` are `wandb.Api().run(...).scan_history()` exports from the
  five substantive GPU runs. The final row also records the peak per-device
  memory sampled by `nvidia-smi` every two seconds, in
  `sampled_peak_gpu_memory_gib`.
- `evaluations/*.json` contain aggregate metrics, fixed-ID hashes, generation
  settings, and empty-response counts. Raw prompts and generations remain local
  with the prepared datasets.

The measured runs are:

| File label | W&B run |
|---|---|
| `cosyn-qlora` | <https://wandb.ai/a-shamsoshoara-m/muse-glimmer-sft-quality/runs/o3disfur> |
| `cosyn-lora-fsdp2` | <https://wandb.ai/a-shamsoshoara-m/muse-glimmer-sft-quality/runs/3y6r7c5h> |
| `cosyn-full-fsdp2` | <https://wandb.ai/a-shamsoshoara-m/muse-glimmer-sft-quality/runs/hx05hx9u> |
| `tulu-qlora` | <https://wandb.ai/a-shamsoshoara-m/muse-glimmer-sft-quality/runs/i8uuy1o0> |
| `docmatix-qlora` | <https://wandb.ai/a-shamsoshoara-m/muse-glimmer-sft-quality/runs/2w8rql9f> |

The CoSyn methods used identical data, seed, sequence limit, global batch,
evaluation cadence, and step count. Tulu and Docmatix test the text-only and
true multi-image QLoRA paths, so their losses and throughput should not be used
as direct method comparisons with CoSyn.

## Reproduce the plots

Run from `post-training/sft/axolotl/` after installing the environment in the main
README:

```bash
python scripts/plot_results.py \
  --history 'CoSyn QLoRA=results/histories/cosyn-qlora.json' \
  --history 'CoSyn bf16 LoRA FSDP2=results/histories/cosyn-lora-fsdp2.json' \
  --history 'CoSyn full FSDP2=results/histories/cosyn-full-fsdp2.json' \
  --evaluation 'Base bf16=results/evaluations/cosyn-base.json' \
  --evaluation 'Base NF4=results/evaluations/cosyn-base-nf4.json' \
  --evaluation 'QLoRA=results/evaluations/cosyn-qlora.json' \
  --evaluation 'bf16 LoRA FSDP2=results/evaluations/cosyn-lora.json' \
  --evaluation 'Full FSDP2=results/evaluations/cosyn-full.json' \
  --output-dir assets/cosyn-methods

python scripts/plot_results.py \
  --history 'CoSyn QLoRA=results/histories/cosyn-qlora.json' \
  --history 'Tulu QLoRA=results/histories/tulu-qlora.json' \
  --history 'Docmatix QLoRA=results/histories/docmatix-qlora.json' \
  --evaluation 'CoSyn base NF4=results/evaluations/cosyn-base-nf4.json' \
  --evaluation 'CoSyn tuned=results/evaluations/cosyn-qlora.json' \
  --evaluation 'Tulu base NF4=results/evaluations/tulu-base-nf4.json' \
  --evaluation 'Tulu tuned=results/evaluations/tulu-qlora.json' \
  --evaluation 'Docmatix base NF4=results/evaluations/docmatix-base-nf4.json' \
  --evaluation 'Docmatix tuned=results/evaluations/docmatix-qlora.json' \
  --evaluation 'Docmatix remove page=results/evaluations/docmatix-qlora-remove.json' \
  --evaluation 'Docmatix reorder pages=results/evaluations/docmatix-qlora-reorder.json' \
  --output-dir assets/qlora-modalities
```

Repeated renders are byte-identical within the pinned recipe environment.

## Reproduce held-out generation

The evaluator sorts examples by a hash of seed and stable ID, then uses greedy
decoding. It records the selected-ID digest and both raw ATEM output and the
first user-directed response. It also adds Muse's `<|eot|>` token to the stop
IDs, allowing private reasoning to finish while preventing repeated answer
turns.

Example QLoRA invocation:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/evaluate.py \
  --examples data/prepared/cosyn-circuit/validation.jsonl \
  --model meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --adapter outputs/muse-glimmer-30b-qlora \
  --load-in-4bit \
  --limit 16 \
  --selection-seed 42 \
  --seed 42 \
  --reasoning-strength low \
  --current-date 2026-09-17 \
  --max-new-tokens 512 \
  --output-json outputs/evaluation/cosyn-qlora.json
```

Use the same IDs, generation settings, and base precision for every comparison.
Pass `--load-in-4bit` for both the untuned NF4 baseline and its QLoRA adapter;
use bf16 for both the untuned baseline and LoRA or full checkpoint. For
Docmatix, repeat the tuned invocation with `--counterfactual remove` and
`--counterfactual reorder`.

> [!CAUTION]
> Evaluation JSON and CSV contain prompts, references, generated answers, and
> raw ATEM output. Keep them in an access-controlled ignored output directory
> when the held-out data is sensitive. Only aggregate summaries are included
> here.
