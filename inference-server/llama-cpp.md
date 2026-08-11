# llama.cpp

Portable inference across CPU, Metal, CUDA, ROCm and Vulkan — the practical way to run Muse Glimmer on a single machine.

> [!NOTE]
> Status: verified on Apple silicon (Metal) against upstream llama.cpp release `b10355` (commit `dd1ea52`) — text, vision and tool calling, against ready-to-serve GGUF checkpoints. There is no conversion or quantization step.

Upstream llama.cpp supports Muse Glimmer. Commit [`62bf73d25`](https://github.com/ggml-org/llama.cpp/commit/62bf73d25) ("model: Muse Glimmer Support", PR #26841) adds the `muse-glimmer` architecture, the vision projector, the ATEM tool-call parser and DFlash speculative decoding. No fork is needed.

## Install

### 1. Download the checkpoints

GGUF builds live in [`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF). Fetch only the two files you need — the text model and, for image input, the vision projector:

```bash
pip install -U huggingface_hub
hf download meta-models/Muse-Glimmer-30B-GGUF --local-dir ./muse-glimmer \
  --include "muse-glimmer-30B-kquant-17gb.gguf" \
  --include "mmproj-kquant.gguf"
```

| File | What | Size |
|---|---|---|
| `muse-glimmer-30B-kquant-17gb.gguf` | text model, K-quant — **use this one** | ~17 GB |
| `muse-glimmer-30B-kquant-dynamic.gguf` | text model, dynamic K-quant | ~20 GB |
| `mmproj-kquant.gguf` | vision projector — needed for image input | ~1.4 GB |
| `dflash-kquant.gguf` | speculative-decode draft model, optional | ~1.6 GB |

Full-precision weights are not published as GGUF. For bf16, use the safetensors checkpoint at [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B) with [`vllm.md`](vllm.md).

### 2. Get llama.cpp

Muse Glimmer support first shipped in release [`b10353`](https://github.com/ggml-org/llama.cpp/releases/tag/b10353). **Use `b10353` or newer** — a prebuilt binary and a source build both work.

> [!IMPORTANT]
> `b10344` and older do not register the `muse-glimmer` architecture and refuse to load these checkpoints:
>
> ```
> llama_model_load: error loading model: unknown model architecture: 'muse-glimmer'
> ```
>
> `llama-server --version` prints the build number to check against: `version: 10353 (...)` or higher.

The [releases page](https://github.com/ggml-org/llama.cpp/releases) publishes prebuilt binaries for macOS arm64 (Metal), Linux (CPU, CUDA, ROCm, Vulkan, SYCL) and Windows. If you take one, unpack it and skip to [Serve](#serve) — read `./build/bin/` in the commands below as the directory you unpacked.

To build from source instead — `master` is well past the floor:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

Add `--branch b10353` to pin to the floor, or any later tag to pin to a known build.

Pick your backend:

```bash
# Apple silicon (Metal)
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF

# NVIDIA (CUDA >= 12.4; set the arch for your card, 90 = Hopper, 120 = Blackwell)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF

# CPU only
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
```

```bash
cmake --build build -j"$(sysctl -n hw.ncpu)" --target llama-server llama-cli llama-mtmd-cli   # macOS
cmake --build build -j"$(nproc)"             --target llama-server llama-cli llama-mtmd-cli   # Linux
```

> [!IMPORTANT]
> `-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF` disables the browser Web UI. Keep it off unless you need it: the UI build fetches assets over the network and fails behind restrictive proxies or without Node.js. The HTTP API is unaffected.

If `ccache` errors during the build, add `-DGGML_CCACHE=OFF`.

Run the commands below from the `llama.cpp` clone (or the unpacked release directory), with the `muse-glimmer/` download directory inside it, so both `./build/bin/` and `./muse-glimmer/` resolve. Otherwise pass absolute paths to `-m` and `--mmproj`.

## Serve

Muse Glimmer supports a native context of **131072**:

```bash
./build/bin/llama-server \
  -m ./muse-glimmer/muse-glimmer-30B-kquant-17gb.gguf \
  --mmproj ./muse-glimmer/mmproj-kquant.gguf \
  -ngl 99 -c 131072 -np 1 \
  --host 127.0.0.1 --port 8080 --api-key <your-key> \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"low"}'
```

The server prints the context you actually got:

```
srv load_model: initializing, n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'
```

If it instead reports `exceeds the training context ... - capping`, the GGUF's
`context_length` metadata is stale and the server clamps the slot to it — no serve
flag overrides this. Fix the metadata with the script in the llama.cpp clone
(`muse-glimmer` is the architecture name llama.cpp registers, so it is the key prefix):
`python gguf-py/gguf/scripts/gguf_set_metadata.py <model>.gguf muse-glimmer.context_length 131072`.

| Flag | Why |
|---|---|
| `--jinja` | Applies the Muse Glimmer control-token template embedded in the GGUF. Without it, tool calling and reasoning separation break. |
| `--chat-template-kwargs` | Sets `reasoning_strength` — see below. Template default is `high`. |
| `-np N` | Concurrent slots, and **the context is divided N ways**: `-np 4` with `-c 131072` gives each slot 32768. Long-context agents want `-np 1`. |
| `-ngl 99` | Offload all layers to the GPU. |
| `--api-key` | Guards inference endpoints only; `/health` and `/v1/models` still answer without it. |

Thinking lands in `message.reasoning_content` and `message.content` stays clean — that is the server default, no flag needed. Pass `--reasoning-format none` to keep thinking inline in `message.content` instead.

Drop `--mmproj` for text-only. For speculative decoding add `-md <draft>.gguf --spec-type draft-dflash -ngld 99 --spec-draft-n-max 4`.

> [!NOTE]
> Do not pass `--chat-template-file`. Upstream ships no Muse Glimmer template under `models/templates/`, and none is needed: `--jinja` uses the template embedded in the GGUF, which is byte-for-byte identical to the `chat_template.jinja` published with the safetensors checkpoint. llama.cpp selects the ATEM tool-call parser by detecting that template, so tool calling works from `--jinja` alone.

Smoke test (`--noproxy` bypasses a proxy that would intercept loopback):

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0}'
```

### Controlling reasoning length

The chat template reads `reasoning_strength` and defaults to `high`:

```jinja
{%- set rs = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' -%}
```

Set it server-wide with `--chat-template-kwargs`, or per request:

```json
{"model":"muse-glimmer","messages":[...],"chat_template_kwargs":{"reasoning_strength":"low"}}
```

The model card documents four levels: `low`, `medium`, `high`, `xhigh`. Completion tokens on one agentic decision, same prompt, `temperature=0`:

| `reasoning_strength` | completion tokens |
|---|---|
| default / `high` | 1,811 |
| `medium` | 752 |
| `low` | 256 |
| `xhigh` | Not measured yet |

Lower is not automatically worse: where the environment validates the answer or the caller can retry cheaply, extra reasoning buys little. Reasoning tokens count against `max_tokens` — a request that hits the ceiling mid-thought returns empty `content` with `finish_reason: "length"`.

> [!NOTE]
> `reasoning_effort`, the OpenAI/vLLM spelling used in [`../recipes/reasoning-effort/`](../recipes/reasoning-effort/), is **not** implemented by llama.cpp. Use `chat_template_kwargs.reasoning_strength`.

### Vision

Send a remote URL, a `data:image/...;base64,...` URI, or a local path:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":[
        {"type":"text","text":"What is in this image?"},
        {"type":"image_url","image_url":{"url":"data:image/png;base64,..."}}
      ]}],"temperature":0}'
```

Images are billed as prompt tokens, scaling with resolution.

## Verify tool calling

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":"What is the weather in Paris in celsius? Use the tool."}],
       "tools":[{"type":"function","function":{
         "name":"get_weather","description":"Get current weather for a city.",
         "parameters":{"type":"object","properties":{
           "city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},
           "required":["city"]}}}],
       "tool_choice":"auto","temperature":0}'
```

Returns `finish_reason: "tool_calls"` and a `tool_calls` array with `get_weather(city="Paris", units="celsius")`, reasoning under `reasoning_content`. Parsing is server-side, so any OpenAI-compatible harness needs no Muse Glimmer-specific handling.

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. The template handles this; `--jinja` is what wires it up. See [`README.md`](README.md#get-stop-tokens-right).

## Next steps

- Drive a desktop agent with this endpoint: [`../recipes/computer-use-web/`](../recipes/computer-use-web/)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
- Compare with the multi-GPU path: [`vllm.md`](vllm.md)
