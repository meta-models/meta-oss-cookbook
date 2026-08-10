# vLLM

Serve Muse Glimmer with native tool calling and stream it to your whole team. This is the reference server for the agentic recipes in this cookbook.

> [!NOTE]
> Status: unverified end to end. The commands below follow the [official vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B), but nobody has attested a full run against this cookbook yet. What *is* confirmed: the `vllm/vllm-openai:muse-glimmer` image is published, and the `muse_glimmer` tool-call and reasoning parsers exist. What is *not* confirmed: that these exact flags serve cleanly on your hardware. If you complete a run, please say so in an issue so we can upgrade this banner.

## Install

Docker is the supported path.

```bash
docker pull vllm/vllm-openai:muse-glimmer
```

`pip install vllm` will **not** serve this model. Muse Glimmer support in vLLM is still an open, unmerged pull request ([vllm-project/vllm#51655](https://github.com/vllm-project/vllm/pull/51655)), so the model code and the `muse_glimmer` parsers are absent from every released wheel. The official recipe sets `pip: false` for exactly this reason. The image is how you get the unreleased code.

The image is ~10.5 GB and publishes `linux/amd64` and `linux/arm64`. The recipe pins it for CUDA 13.0 (`docker_image.nvidia.cu130`). There is no published ROCm image — on ROCm, build from PR #51655.

What the image gives you for Muse Glimmer:

| Capability | How you get it |
|---|---|
| Text model (`MuseGlimmerForCausalLM`) | Automatic. No `trust_remote_code`. |
| Tool-call parser | `--tool-call-parser muse_glimmer` |
| Reasoning parser | `--reasoning-parser muse_glimmer` |
| Chat template | Ships with the checkpoint as `chat_template.jinja`. Don't pass `--chat-template`. |
| Model config | Automatic (`muse-glimmer` / `muse_glimmer_text` / `muse_glimmer_vision`). |

Parser names use **underscores** (`muse_glimmer`). vLLM matches them literally, so the hyphenated form fails. `--served-model-name muse-glimmer` is a label you choose and stays hyphenated.

Vision (multimodal) is a fast-follow. Text and tool calling work today.

Full tool calling needs a checkpoint converted with the current HF Muse Glimmer converter, whose tokenizer registers the framing tokens (`<|eom|>`, `<|eot|>`, `<|start|>`, `<|message|>`) as real single tokens. Legacy exports still work for text and NLL, but won't produce clean token-level framing.

## Serve

The image sets `ENTRYPOINT ["vllm", "serve"]`. Everything after the image name is appended to that, so pass the model and flags directly — **do not** retype `vllm serve`, and don't reach for `python -m vllm.entrypoints.openai.api_server`.

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:muse-glimmer \
  meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
```

Two things to know about that command. The flags from `--served-model-name` down are the recipe's own argument list for a single card. The `docker run` wrapper around them — `--gpus all`, `--ipc=host`, the port, the cache mount — is standard vLLM boilerplate rather than anything the recipe specifies; adjust it to your host. Mounting `~/.cache/huggingface` is what stops the container re-downloading ~60 GB of weights on every start.

The recipe itself mounts pre-downloaded weights at `/model` instead of passing a Hub id:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v /path/to/Muse-Glimmer-30B:/model \
  vllm/vllm-openai:muse-glimmer \
  /model \
  --served-model-name muse-glimmer \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
```

Multi-GPU — raise `--tensor-parallel-size` to cover the model:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:muse-glimmer \
  meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
```

Tighter memory, at a cost in throughput and context:

```bash
docker run --rm --gpus all --ipc=host \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:muse-glimmer \
  meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --gpu-memory-utilization 0.9 --max-model-len 8300 --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --generation-config auto
```

> [!NOTE]
> Budget **72 GB of VRAM** to serve, which is what the official recipe asks for. The bf16 weights are only ~60 GB; the rest is KV cache, activations and CUDA overhead. Sizing to the weights alone is the usual way to OOM shortly after startup.

## Sampling

Serve with `--generation-config auto`, as every command above does, and vLLM picks up the checkpoint's published sampling settings: **temperature 1.0, top_p 0.95, top_k 64**.

> [!WARNING]
> Don't run this model greedy. Passing `"temperature": 0` in a request overrides the published settings and the recipe explicitly advises against it. If you set sampling per-request, carry all three values rather than a bare temperature.

## Verify tool calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "muse-glimmer",
  "messages": [{"role":"user","content":"What is the weather in Paris in celsius? Use the tool."}],
  "tools": [{"type":"function","function":{
    "name":"get_weather","description":"Get current weather for a city.",
    "parameters":{"type":"object","properties":{
      "city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},
      "required":["city"]}}}],
  "tool_choice":"auto"
}'
```

You get back `get_weather(city="Paris", units="celsius")`, with the reasoning surfaced under `message.reasoning` (`delta.reasoning` when streaming).

> [!IMPORTANT]
> Muse Glimmer emits **one tool call per message**. Several calls arrive as consecutive assistant messages, not as several entries in a single `tool_calls` array. Harnesses that read only the first element of the first message, or that assume one assistant message means one round of calls, will silently drop work.

## Stop tokens

`eos_token_id = [200001, 200008]` = `<|end_of_text|>` + `<|eot|>`.

> [!WARNING]
> Don't include `<|eom|>` (200007). It's end-of-*message*: the turn continues, and it separates the reasoning block and each non-final parallel tool call. Stopping on it drops single-turn parallel tool calling to near zero. Ship this in `generation_config.json`; `--generation-config auto` picks it up.

Token IDs above come from the converted checkpoint. Confirm against your checkpoint's `tokenizer_config.json`: older exports may differ.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ValueError: model architectures ... are not supported` | Running a released vLLM wheel, not the image | The model code isn't in any wheel. Use `vllm/vllm-openai:muse-glimmer`. |
| `invalid tool call parser` | Hyphenated parser name | Use underscores: `muse_glimmer`. |
| Tool calls returned as plain text | Parser not enabled | Pass `--tool-call-parser muse_glimmer`. The chat template comes from the checkpoint. |
| Only the first tool call runs | Harness expects a `tool_calls` array | One call per message — read consecutive assistant messages. |
| Model runs forever | Wrong stop tokens | See above — never stop on `<\|eom\|>`. |
| `to=self` reasoning leaking into content | No reasoning parser | Use `--reasoning-parser muse_glimmer`; it routes reasoning to `message.reasoning`. |
| Flat, repetitive answers | Running greedy | Don't send `"temperature": 0`. Use the published settings. |
| OOM shortly after startup | Sized to the ~60 GB weights, not the 72 GB serving footprint | Lower `--max-model-len`, raise `--tensor-parallel-size`, or use a quant. |

## Next steps

- Learn the loop this endpoint drives: [`../agentic-fundamentals/`](../agentic-fundamentals/)
- Ship a flagship agent against it: [`../recipes/`](../recipes/)

## Appendix: NLL parity vs the HF reference

Teacher-forced per-token NLL (via vLLM `prompt_logprobs`) against the HF reference:

| Dataset | Docs | MATCH (<1e-3) | CLOSE (<0.1) | MISMATCH | max mean-NLL Δ |
|---|---|---|---|---|---|
| prose | 20 | 11 | 9 | 0 | 0.0042 |
| code | 20 | 16 | 4 | 0 | 0.0050 |

`CLOSE` deltas are expected under bf16 and different kernels.
