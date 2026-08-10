# ExecuTorch

Export Muse Glimmer ahead of time and serve it locally on CUDA or Apple silicon, with vision, tool calling, and DFlash speculative decoding.

> [!NOTE]
> Status: supported upstream — text, vision, tool calling and DFlash. Not re-run for this cookbook, so no VRAM or throughput numbers are published here. Upstream reference: [`examples/models/muse-glimmer`](https://github.com/pytorch/executorch/tree/main/examples/models/muse-glimmer).

Unlike the other servers here, ExecuTorch is ahead-of-time: the model is exported once into a `.pte` program for a specific backend, then served from that program.

| Backend | Host | Exported artifacts |
|---|---|---|
| CUDA | Linux or Windows | `model.pte` plus `aoti_cuda_blob.ptd` |
| MLX | macOS on Apple silicon | self-contained `model.pte` |

**CPU export is not supported.** For a CPU-only machine, use [`llama-cpp.md`](llama-cpp.md).

## Install

Build ExecuTorch from source per the [upstream guide](https://github.com/pytorch/executorch), then install the server dependencies:

```bash
pip install -r examples/llm_server/python/requirements.txt
```

### 1. Get the assets

Exports lower directly from the quantized GGUFs — the same files [`llama-cpp.md`](llama-cpp.md) uses — plus tokenizer metadata from the safetensors repo. Run from the ExecuTorch repository root:

```bash
hf download meta-models/Muse-Glimmer-30B-GGUF \
  --include '*.gguf' --exclude '*[Bb][Ff]16*.gguf' --local-dir assets/quant

hf download meta-models/Muse-Glimmer-30B \
  tokenizer.json chat_template.jinja config.json processor_config.json \
  --local-dir assets/hf
```

`chat_template.jinja` must stay beside the tokenizer metadata; the serving path renders prompts and tool definitions with it.

```bash
TARGET="$(find assets/quant -name 'muse-glimmer-30B-kquant-17gb.gguf' -print -quit)"
DRAFT="$(find assets/quant -name 'dflash-kquant.gguf' -print -quit)"
MMPROJ="$(find assets/quant -name 'mmproj-kquant.gguf' -print -quit)"
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

Add `--mmproj "$MMPROJ"` to either command for vision; that also writes `pos_embed.bin` beside `model.pte`.

> [!NOTE]
> Prebuilt `.pte` artifacts are published at [`meta-models/Muse-Glimmer-30B-ExecuTorch-PTE`](https://huggingface.co/meta-models/Muse-Glimmer-30B-ExecuTorch-PTE), covering both quantizations across text / text+image, solo / DFlash, and Metal / CUDA. Downloading one skips this export step — **the runner build in step 3 is still required.** Their CUDA blob is named after its own directory rather than `aoti_cuda_blob.ptd`, so `--data-path` below needs adjusting to match. Upstream does not document serving these files yet, so this page stays on the export path.

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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Worker fails to load the method | Runner built without the quantized / custom-op kernels | Rebuild with the model's CMake workflow preset rather than a plain ExecuTorch build. |
| Tool calls arrive as plain text | Server started without `--tool-parser atem` | Re-serve with the flag. The default is `none`, which passes model output through unparsed. |
| `400` on a request that works elsewhere | Unsupported OpenAI parameter | See the rejected parameters above. |

## Next steps

- Compare with the reference path: [`vllm.md`](vllm.md)
- CPU-only machine: [`llama-cpp.md`](llama-cpp.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
