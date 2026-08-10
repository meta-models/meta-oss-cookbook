# Quickstart

Get Muse Glimmer running with one command per server — pick the one that matches your hardware and go.

Every path here runs fully offline after the initial weight download. No API keys.

## Pick your server

| Server | Best for | Precision | Guide |
|---|---|---|---|
| vLLM | Serving to a team, tool calling | bf16 / fp8 | [jump](#serve-with-vllm) |
| HF Transformers | Learning the loop, pure Python | bf16 | [jump](#run-with-hugging-face-transformers) |
| Ollama | Single command, no build step | Q4_K_M | [jump](#run-with-ollama) |
| LM Studio | GUI, Apple Silicon (MLX) | Q4 / MLX | [jump](#run-with-lm-studio) |

> [!NOTE]
> The dense Muse Glimmer text model is ~29B params.
> - bf16 lands around 59–60 GB: fits a single 80 GB card, or two cards with tensor parallelism.
> - Q4_K_M / INT4 lands around 16–17 GB.
>
> The agentic recipes in this cookbook target bf16 on vLLM. Quantized paths work, but we don't check each recipe across quant schemes.

## Serve with vLLM

> [!NOTE]
> Status: unverified end to end. This is the path to use for agentic tool calling, and it follows the official vLLM recipe, but no full run has been attested for this cookbook. Details in [`../inference-server/vllm.md`](../inference-server/vllm.md).

vLLM has a native Muse Glimmer model (`MuseGlimmerForCausalLM`, no `trust_remote_code`) plus a tool-call parser and reasoning parser. They ship in a dedicated image — `pip install vllm` will not serve this model, because Muse Glimmer support is still an unmerged upstream PR and isn't in any released wheel.

```bash
docker pull vllm/vllm-openai:muse-glimmer
```

The image's entrypoint is already `vllm serve`, so everything after the image name is appended as arguments:

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

Budget 72 GB of VRAM to serve — the bf16 weights are ~60 GB, and KV cache and activations need the rest.

Test tool calling against the OpenAI-compatible endpoint:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "muse-glimmer",
  "messages": [{"role": "user", "content": "What is the weather in Paris in celsius? Use the tool."}],
  "tools": [{"type":"function","function":{
    "name":"get_weather",
    "description":"Get current weather for a city.",
    "parameters":{"type":"object","properties":{
      "city":{"type":"string"},
      "units":{"type":"string","enum":["celsius","fahrenheit"]}},
      "required":["city"]}}}],
  "tool_choice": "auto"
}'
```

You get back `get_weather(city="Paris", units="celsius")`. Note that Muse Glimmer emits one tool call per message — several calls arrive as consecutive assistant messages, not as several entries in one `tool_calls` array.

> [!WARNING]
> Stop tokens matter. Set `eos_token_id = [<|end_of_text|>, <|eot|>]`. Don't add `<|eom|>` as a stop token: it's end-of-*message* (the turn continues), and stopping on it drops parallel tool calling to near zero. Details in [`../inference-server/vllm.md`](../inference-server/vllm.md).

## Run with Hugging Face Transformers

> [!NOTE]
> Status: tested. This is the path for learning the agent loop with no server.

```bash
pip install "transformers>=5.15" accelerate torch
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-models/Muse-Glimmer-30B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "In one sentence, what is Muse Glimmer good at?"}]
enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                              return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**enc, max_new_tokens=128, do_sample=False,
                     eos_token_id=[tok.convert_tokens_to_ids("<|eot|>"),
                                   tok.convert_tokens_to_ids("<|end_of_text|>")])
print(tok.decode(out[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True))
```

To turn this into an agent that calls tools and self-corrects, go to [`../agentic-fundamentals/`](../agentic-fundamentals/). It ships a complete, runnable agent loop built on this exact API.

## Run with Ollama

> [!NOTE]
> Status: pending Ollama-verified GGUF publish. The command shape below is the target.

Install Ollama from [ollama.com/download](https://ollama.com/download), then pull and run. Swap in the published tag when available.

```bash
ollama run muse-glimmer
```

Ollama exposes an OpenAI-compatible endpoint on `:11434`. Call it from another terminal:

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "muse-glimmer",
  "messages": [{"role": "user", "content": "In one sentence, what is Muse Glimmer good at?"}]
}'
```

For tool calling, use the vLLM path above: Ollama's default templates may not emit Muse Glimmer's tool-block framing.

## Run with LM Studio

> [!NOTE]
> Status: pending LM Studio / MLX quant publish. See [`../inference-server/lm-studio.md`](../inference-server/lm-studio.md).

1. Install LM Studio from [lmstudio.ai](https://lmstudio.ai).
2. Search the model catalog for Muse Glimmer and download the recommended build (MLX on Apple Silicon, GGUF Q4_K_M elsewhere).
3. Load it and open the Chat tab, or start the Local Server (OpenAI-compatible, `:1234`) to call it from code.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Model never stops generating | Missing or wrong stop tokens | Set `eos_token_id=[<\|end_of_text\|>, <\|eot\|>]`. Don't include `<\|eom\|>`. |
| Tool calls come back as plain text | Server isn't parsing the tool block | Use vLLM with `--tool-call-parser muse_glimmer`, or parse it yourself ([agentic-fundamentals](../agentic-fundamentals/)). |
| OOM on a 24 GB card | Running bf16 (~60 GB) | Add tensor parallelism, or use a quantized build (Ollama, LM Studio). |
| `trust_remote_code` prompt | Checkpoint ships custom modeling code | Use the native path: transformers ≥ 5.15, or serve with the `vllm/vllm-openai:muse-glimmer` image. |

## Next steps

- Understand what just happened: [`../agentic-fundamentals/`](../agentic-fundamentals/)
- Ship a flagship agent: [`../recipes/`](../recipes/)
- Go deeper on your server: [`../inference-server/`](../inference-server/)
