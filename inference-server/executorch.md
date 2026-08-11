# ExecuTorch

Export Muse Glimmer ahead of time and serve it locally on CUDA or Apple silicon, with vision, tool calling, and DFlash speculative decoding.

> [!NOTE]
> Status: supported upstream — text, vision, tool calling and DFlash. Not re-run for this cookbook, so no VRAM or throughput numbers are published here; the upstream announcement, [Fast on-device agentic AI with ExecuTorch](https://pytorch.org/blog/fast-ondevice-agentic-ai-with-executorch/), explains why this model is exported rather than reimplemented per backend, and publishes PyTorch's own measurements on Apple silicon and NVIDIA. Upstream reference: [`examples/models/muse-glimmer`](https://github.com/pytorch/executorch/tree/main/examples/models/muse-glimmer).

Unlike the other servers here, ExecuTorch is ahead-of-time: the model is exported once into a `.pte` program for a specific backend, then served from that program.

That is the point of it. The other runtimes on this list reimplement a model by hand per backend, which works for a plain text transformer but not for a novel architecture carrying multimodal input and block-diffusion speculative decoding — each backend would need its own rewrite of all three. With ExecuTorch the model and its decoding strategy are written once in PyTorch and `torch.export` lowers the whole graph ahead of time, to Triton on CUDA and to MLX-native or custom Metal kernels on Apple silicon.

| Backend | Host | Artifacts written by a local export |
|---|---|---|
| CUDA | Linux or Windows | `model.pte` plus `aoti_cuda_blob.ptd` |
| MLX | macOS on Apple silicon | self-contained `model.pte` |

Prebuilt exports are also published, under [different filenames](#alternative-download-a-prebuilt-export) — exporting a 30B model yourself is optional.

**CPU export is not supported.** For a CPU-only machine, use [`llama-cpp.md`](llama-cpp.md).

## Install

Build ExecuTorch from source per the [upstream guide](https://github.com/pytorch/executorch), then install the server dependencies:

```bash
pip install -r examples/llm_server/python/requirements.txt
```

### 1. Get the assets

Exports lower directly from the quantized GGUFs — the same files [`llama-cpp.md`](llama-cpp.md) uses — plus tokenizer metadata from the safetensors repo. Run from the ExecuTorch repository root:

```bash
hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir assets/quant \
  --include 'muse-glimmer-30B-kquant-17gb.gguf' \
  --include 'dflash-kquant.gguf' \
  --include 'mmproj-kquant.gguf'

hf download meta-models/Muse-Glimmer-30B \
  tokenizer.json chat_template.jinja config.json processor_config.json \
  --local-dir assets/hf
```

`chat_template.jinja` must stay beside the tokenizer metadata; the serving path renders prompts and tool definitions with it.

```bash
TARGET=assets/quant/muse-glimmer-30B-kquant-17gb.gguf
DRAFT=assets/quant/dflash-kquant.gguf
MMPROJ=assets/quant/mmproj-kquant.gguf
BACKEND=cuda   # mlx on macOS
```

### 2. Export

Target model:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_solo \
  --gguf "$TARGET" --backend "$BACKEND" --output-dir exports/solo
```

DFlash speculative decoding lowers the target and the draft together, into their own export directory:

```bash
python -m executorch.examples.models.muse_glimmer.export.export_dflash \
  --target-gguf "$TARGET" \
  --draft-gguf "$DRAFT" \
  --backend "$BACKEND" \
  --output-dir exports/dflash
```

Both land in one `.pte`: the draft shares the target's token embeddings and output head rather than carrying copies. The block dimension is exported dynamically, so block length is selectable at serve time — but the exported range is backend-specific, `[2, 16]` on MLX against `[2, 4]` on CUDA, where the draft count is also capped at 3.

Add `--mmproj "$MMPROJ"` to either command for vision; that also writes `pos_embed.bin` beside `model.pte`.

A CUDA export autotunes Triton kernels against the GPU it runs on, so export on the same architecture you intend to serve from.

#### Alternative: download a prebuilt export

[`meta-models/Muse-Glimmer-30B-ExecuTorch-PTE`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE) publishes 16 ready-made exports, which skips step 2 entirely. **Step 3 still applies** — a `.pte` is a model program, not a runtime, so you still build `muse_glimmer_worker` from source.

> [!WARNING]
> **That repository is 372 GB.** One export is 18–31 GB, so always download a single directory with `--include` rather than the whole repo.

Directories are named `muse-glimmer-<quantization>-128K-<modality>-<decoding>-<backend>`, and all 16 combinations of the four axes exist:

- **Quantization** — `k-quant-17G` targets 24 GB of VRAM, `k-quant-dynamic` targets 32 GB with less degradation (the [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE) quantifies the tradeoff).
- **Modality** — `text`, or `text-image` for vision.
- **Decoding** — `solo`, or `dflash` for speculative decoding.
- **Backend** — `metal` for Apple silicon, `sm80+ptx` for CUDA on SM80 and newer.

Context length is `128K` for every variant. Sizes, by directory:

| Quantization | Modality | Decoding | `…-metal` | `…-sm80+ptx` |
|---|---|---|---|---|
| `k-quant-17G` | `text` | `solo` | 17.9 GB | 19.8 GB |
| `k-quant-17G` | `text` | `dflash` | 19.6 GB | 27.2 GB |
| `k-quant-17G` | `text-image` | `solo` | 19.4 GB | 21.2 GB |
| `k-quant-17G` | `text-image` | `dflash` | 21.1 GB | 28.6 GB |
| `k-quant-dynamic` | `text` | `solo` | 20.7 GB | 22.6 GB |
| `k-quant-dynamic` | `text` | `dflash` | 22.4 GB | 30.0 GB |
| `k-quant-dynamic` | `text-image` | `solo` | 22.2 GB | 24.0 GB |
| `k-quant-dynamic` | `text-image` | `dflash` | 23.8 GB | 31.5 GB |

Each directory holds `<directory-name>.pte`; `sm80+ptx` variants add `<directory-name>.ptd`, and `text-image` variants add `pos_embed.bin`. On CUDA the weights live in the `.ptd` and the `.pte` is only tens of megabytes, so both files are required. The repository root carries the tokenizer metadata — `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja` — so this one repository covers everything the server needs; the `assets/hf` download in step 1 is for the export path.

```bash
EXPORT_DIR=muse-glimmer-k-quant-17G-128K-text-solo-sm80+ptx

hf download meta-models/Muse-Glimmer-30B-ExecuTorch-PTE \
  --include "$EXPORT_DIR/*" \
  --include tokenizer.json --include tokenizer_config.json --include chat_template.jinja \
  --local-dir exports
```

Upstream documents the same `--include` approach in [Prebuilt PTE artifacts](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/README.md#prebuilt-pte-artifacts), though its example still shows the older underscored directory names.

> [!IMPORTANT]
> A prebuilt directory names its artifacts after itself, so **both** `--model-path` and `--data-path` differ from the export path below — there is no `model.pte` and no `aoti_cuda_blob.ptd` in a download. Upstream's serve snippet and the [announcement](https://pytorch.org/blog/fast-ondevice-agentic-ai-with-executorch/) quickstart both show the export filenames, which fail against downloaded artifacts.

### 3. Build the runner

```bash
(cd examples/models/muse-glimmer && cmake --workflow --preset muse-glimmer-cuda)   # or muse-glimmer-mlx
```

Binaries land in `cmake-out/examples/models/muse-glimmer/`; the server needs `muse_glimmer_worker`.

## Serve

```bash
python -m executorch.examples.models.muse_glimmer.serving.serve \
  --model-path exports/solo/model.pte \
  --data-path exports/solo/aoti_cuda_blob.ptd \
  --tokenizer-path assets/hf/tokenizer.json \
  --hf-tokenizer assets/hf \
  --worker-bin cmake-out/examples/models/muse-glimmer/muse_glimmer_worker \
  --model-id muse-glimmer-30B \
  --tool-parser atem \
  --host 127.0.0.1 --port 8000
```

On MLX, drop `--data-path`. For DFlash, replace `exports/solo` with `exports/dflash` in both artifact paths — the server detects the exported method contract. For vision, add `--pos-embed-path <export-dir>/pos_embed.bin`.

From a [prebuilt export](#alternative-download-a-prebuilt-export), substitute the two artifact flags — the rest of the command is unchanged:

```bash
  --model-path "exports/$EXPORT_DIR/$EXPORT_DIR.pte" \
  --data-path "exports/$EXPORT_DIR/$EXPORT_DIR.ptd" \
  --tokenizer-path exports/tokenizer.json \
  --hf-tokenizer exports \
```

Drop `--data-path` for a `-metal` variant, since it has no `.ptd`. Add `--pos-embed-path "exports/$EXPORT_DIR/pos_embed.bin"` for a `text-image` variant. A `-dflash` variant needs no extra flag.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30B","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32,"temperature":0}'
```

`/health`, `/v1/models` and `/v1/chat/completions` (streaming and non-streaming) are implemented. `--max-context` bounds the context window; prompts over it are rejected with a 400.

## Verify tool calling

`--tool-parser atem` is what makes this work: the Hugging Face template renders the tool definitions, and the server converts Muse Glimmer's `<atem:function_calls>` output into an OpenAI-compatible `tool_calls` array, so a harness needs no Muse Glimmer-specific handling. Reference: [`serving/tool_parsers/atem.py`](https://github.com/pytorch/executorch/blob/main/examples/models/muse-glimmer/serving/tool_parsers/atem.py).

Two OpenAI parameters are rejected with a structured 400 rather than silently ignored, which matters when pointing recipes here:

- `reasoning_effort` — used by [`../recipes/reasoning-effort/`](../recipes/reasoning-effort/). Control reasoning through the chat template instead.
- `tool_choice="required"` — `none`, `auto` and unset are accepted.

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. See [`README.md`](README.md#get-stop-tokens-right).

## Limitations

Runtime limits, not model limits — the [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) covers the latter.

- **No video input.** Text and images only, one image per request.
- **No continuous batching.** One request runs at a time: `--num-runners` must be 1, the exported methods are batch-1, and execution is serialized. Concurrent sessions are isolated from each other, not served in parallel.
- **No cross-session prefix sharing and no checkpointing.** Every session holds its own KV cache, nothing is reused across sessions, and session state is discarded rather than saved.

The upstream [announcement](https://pytorch.org/blog/fast-ondevice-agentic-ai-with-executorch/) lists all three as in progress. If you need concurrent throughput from one box today, use [`vllm.md`](vllm.md).

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Worker fails to load the method | Runner built without the quantized / custom-op kernels | Rebuild with the model's CMake workflow preset rather than a plain ExecuTorch build. |
| Tool calls arrive as plain text | Server started without `--tool-parser atem` | Re-serve with the flag. The default is `none`, which passes model output through unparsed. |
| `400` on a request that works elsewhere | Unsupported OpenAI parameter | See the rejected parameters above. |
| Artifact not found on a downloaded export | Export filenames used against a prebuilt directory | A download has no `model.pte` or `aoti_cuda_blob.ptd`; both artifacts are named after their directory. |

## Next steps

- Compare with the reference path: [`vllm.md`](vllm.md)
- CPU-only machine: [`llama-cpp.md`](llama-cpp.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
