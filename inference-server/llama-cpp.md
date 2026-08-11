# llama.cpp

Portable inference across CPU, Metal, CUDA, ROCm and Vulkan — the practical way to run Muse Glimmer on a single machine.

> [!NOTE]
> Status: verified on Apple silicon (Metal) against upstream llama.cpp release `b10355` (commit `dd1ea52`) — text, vision and tool calling, against ready-to-serve GGUF checkpoints. There is no conversion or quantization step.

Upstream llama.cpp supports Muse Glimmer. Commit [`62bf73d25`](https://github.com/ggml-org/llama.cpp/commit/62bf73d25) ("model: Muse Glimmer Support", PR #26841) adds the `muse-glimmer` architecture, the vision projector, the ATEM tool-call parser and DFlash speculative decoding. No fork is needed.

## Install

### 1. Get llama.cpp

You can install pre-built llama binary with this command, it automatically downloads the one for your platform.

```bash
curl -LsSf https://llama.app/install.sh | sh
```

## Serve

Llama.cpp automatically downloads, caches and serves models with the command `llama serve -hf`. Muse Glimmer supports a native context of **131072**:

```bash
llama serve -hf meta-models/Muse-Glimmer-30B-GGUF --chat-template-kwargs '{"reasoning_strength":"low"}'
```

You can specify which quantization you want to serve by passing quant suffix.

```bash
llama serve -hf meta-models/Muse-Glimmer-30B-GGUF:kquant-17gb
llama serve -hf meta-models/Muse-Glimmer-30B-GGUF:kquant-dynamic
```

You can run llama.cpp with DFlash speculative decoding as follows.

```bash
llama serve \
  -hf meta-models/Muse-Glimmer-30B-GGUF:kquant-17gb \
  --hf-repo-draft meta-models/Muse-Glimmer-30B-GGUF:dflash-kquant \
  --spec-type draft-dflash \
  -ngld 99 \
  --spec-draft-n-max 4
```


| Flag | Why |
|---|---|
| `--jinja` | Applies the Muse Glimmer control-token template embedded in the GGUF. Without it, tool calling and reasoning separation break. |
| `-a muse-glimmer` | The name the API answers to. Without it the alias is the checkpoint path, and the `"model": "muse-glimmer"` every example here sends will not match. |
| `--chat-template-kwargs` | Sets `reasoning_strength` — see below. Template default is `high`. |
| `-np N` | Concurrent slots, and **the context is divided N ways**: `-np 4` with `-c 131072` gives each slot 32768. Long-context agents want `-np 1`. See [Context per slot](#context-per-slot). |
| `-ngl 99` | Offload all layers to the GPU. |
| `--api-key` | Guards inference endpoints only; `/health` and `/v1/models` still answer without it. |


> [!NOTE]
> Do not pass `--chat-template-file`. Upstream ships no Muse Glimmer template under `models/templates/`, and none is needed: `--jinja` uses the template embedded in the GGUF, which is byte-for-byte identical to the `chat_template.jinja` published with the safetensors checkpoint. llama.cpp selects the ATEM tool-call parser by detecting that template, so tool calling works from `--jinja` alone.

Smoke test (`--noproxy` bypasses a proxy that would intercept loopback):

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" -H "Authorization: Bearer <your-key>" \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":"What is 17 * 23?"}],"temperature":0}'
```

### Context per slot

`n_ctx_slot` in the startup log, not `-c`, is what bounds a single generation. Getting this wrong fails quietly: a generation that runs out of room produces no answer and no error, so a batch job or an eval just reports a worse number and gives you nothing to debug.

Muse Glimmer reasons at length, which makes the margin thinner than it looks — the published GGUF run notes for this model record a longest single generation of 31,044 tokens, against a 32,768-token slot. To keep concurrency without shrinking the slot, scale `-c` **with** `-np`:

```bash
  -c 524288 -np 4        # 131072 per slot, still 4-way concurrent
```

KV cache stays affordable at that size: GQA with 2 KV heads, and sliding-window attention on three of every four layers.

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

Reasoning cannot be turned off here. The template opens the thinking channel unconditionally, so `--reasoning off` and `"reasoning_effort": "none"` both leave the output unchanged. `reasoning_strength: low` is how you spend fewer thinking tokens; `--reasoning-budget N` is how you hard-cap them.

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

## Command line, without the server

You can use `llama cli` to interact with the model through CLI.

```bash
llama cli -hf serve -hf meta-models/Muse-Glimmer-30B-GGUF -ngl 99 -c 32768 --jinja -st
```

`-st` / `--single-turn` answers once and exits. Without it `llama-cli` stays interactive and waits on stdin, which reads as a hang.

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
