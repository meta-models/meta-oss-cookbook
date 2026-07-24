# vLLM

Serve Muse Glimmer with native tool calling and stream it to your whole team. This is the server every recipe in this cookbook is verified against.

> [!NOTE]
> Status: verified. vLLM ships a native `MuseGlimmerForCausalLM` (no `trust_remote_code`), a tool-call parser, a reasoning parser, and the Muse Glimmer chat template.

## Install

```bash
pip install vllm
```

What vLLM provides for Muse Glimmer:

| Area | File | Flag |
|---|---|---|
| Text model | `vllm/model_executor/models/muse_glimmer.py` | (auto) |
| Tool-call parser | `vllm/tool_parsers/muse_glimmer_tool_parser.py` | `--tool-call-parser muse-glimmer` |
| Reasoning parser | `vllm/reasoning/muse_glimmer_reasoning_parser.py` | `--reasoning-parser muse-glimmer` |
| Chat template | `examples/tool_chat_template_muse_glimmer.jinja` | `--chat-template …` |
| Config (native) | `vllm/transformers_utils/configs/muse_glimmer.py` | (auto: `muse-glimmer` / `muse_glimmer_text` / `muse_glimmer_vision`) |

Vision (multimodal) is a fast-follow. Text and tool calling work today.

Full tool calling needs a checkpoint converted with the current HF Muse Glimmer converter, whose tokenizer registers the framing tokens (`<|eom|>`, `<|eot|>`, `<|start|>`, `<|message|>`) as real single tokens. Legacy exports still work for text and NLL, but won't produce clean token-level framing.

## Serve

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --enable-auto-tool-choice \
  --tool-call-parser muse-glimmer \
  --reasoning-parser muse-glimmer \
  --chat-template examples/tool_chat_template_muse_glimmer.jinja \
  --generation-config auto
```

Single 80 GB card, tighter memory:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B --served-model-name muse-glimmer \
  --gpu-memory-utilization 0.9 --max-model-len 8300 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser muse-glimmer --reasoning-parser muse-glimmer \
  --chat-template examples/tool_chat_template_muse_glimmer.jinja --generation-config auto
```

Multi-GPU with tensor parallelism (bf16 needs ~60 GB, so use enough cards to cover it — e.g. 2× 40 GB or 4× 24 GB):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B --served-model-name muse-glimmer \
  --tensor-parallel-size 2 \
  --enable-auto-tool-choice --tool-call-parser muse-glimmer --reasoning-parser muse-glimmer \
  --chat-template examples/tool_chat_template_muse_glimmer.jinja --generation-config auto
```

## Verify tool calling

```bash
curl http://localhost:8000/v1/chat/completions -d '{
  "model": "muse-glimmer",
  "messages": [{"role":"user","content":"What is the weather in Paris in celsius? Use the tool."}],
  "tools": [{"type":"function","function":{
    "name":"get_weather","description":"Get current weather for a city.",
    "parameters":{"type":"object","properties":{
      "city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},
      "required":["city"]}}}],
  "tool_choice":"auto","temperature":0.0
}'
```

You get a `tool_calls` array with `get_weather(city="Paris", units="celsius")`, and the reasoning surfaced under `reasoning_content`.

## Stop tokens

`eos_token_id = [200001, 200008]` = `<|end_of_text|>` + `<|eot|>`.

> [!WARNING]
> Don't include `<|eom|>` (200007). It's end-of-*message*: the turn continues, and it separates the reasoning block and each non-final parallel tool call. Stopping on it drops single-turn parallel tool calling to near zero. Ship this in `generation_config.json`; `--generation-config auto` picks it up.

Token IDs above come from the converted checkpoint. Confirm against your checkpoint's `tokenizer_config.json`: older exports may differ.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Tool calls returned as plain text | Parser or template missing | Pass both `--tool-call-parser muse-glimmer` and the Muse Glimmer chat template. |
| Model runs forever | Wrong stop tokens | See above — never stop on `<\|eom\|>`. |
| `to=self` reasoning leaking into content | No reasoning parser | Use `--reasoning-parser muse-glimmer`; it routes reasoning to `reasoning_content`. |
| OOM | bf16 needs ~60 GB | Lower `--max-model-len`, add `--tensor-parallel-size`, or use a quant. |

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
