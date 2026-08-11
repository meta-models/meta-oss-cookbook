---
vendor: Together AI
contact: Mourya Vangala Srinivasa (msrinivasa@together.ai)
updated: 2026-08-11
---

# Together AI

Together AI serves Muse Glimmer as a managed API. There are no weights to download, no runtime to install, and no GPU to own — you send requests to `https://api.together.xyz/v1/chat/completions` with a model string and an API key.

**This page requires an API key.** That is the trade the rest of this cookbook doesn't make; see [`README.md`](README.md) for why the folder exists.

Everything under "What Together publishes" is taken from Together's model page, [together.ai/models/muse-glimmer](https://www.together.ai/models/muse-glimmer), read on 2026-08-11. Anything Together measured but doesn't publish there is marked as theirs.

## The model string

```
meta-models/Muse-Glimmer-30B
```

That string is the whole integration. It is what Together lists as the endpoint on the model page, and it's what appears in all three of their published examples.

## What Together publishes

| | |
|---|---|
| Endpoint | `meta-models/Muse-Glimmer-30B` |
| Provider | Meta |
| Type | Chat, Vision |
| Deployment | Serverless, Dedicated |
| Parameters | 30B |
| Context length | 128K+ |
| Input modalities | Text, Image |
| Output modalities | Text |
| Input price | $0.35 / 1M tokens ($0.04 / 1M cached) |
| Output price | $1.50 / 1M tokens |
| Released | August 10, 2026 |
| License | Apache 2.0 |
| Availability | 99.9% SLA, serverless and dedicated |

Tool calling is supported through the model's chat template, which is what makes this endpoint usable for the agent loops in [`../agentic-fundamentals/`](../agentic-fundamentals/).

## Get a key

```bash
export TOGETHER_API_KEY=...   # https://api.together.xyz/settings/api-keys
```

Check that it works and that the model is visible to your account:

```bash
curl -s https://api.together.xyz/v1/models \
  -H "Authorization: Bearer $TOGETHER_API_KEY" | grep -o 'meta-models/Muse-Glimmer-30B'
```

## Call it

Together publishes three clients. The curl form is the one to reach for when you want to see the wire format:

```bash
curl -X POST "https://api.together.xyz/v1/chat/completions" \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Glimmer-30B",
    "messages": [
      {"role": "user", "content": "What are some fun things to do in New York?"}
    ]
  }'
```

Python, using Together's official SDK (`pip install together`; it reads `TOGETHER_API_KEY` from the environment):

```python
from together import Together

client = Together()
response = client.chat.completions.create(
    model="meta-models/Muse-Glimmer-30B",
    messages=[{"role": "user", "content": "What are some fun things to do in New York?"}],
)
print(response.choices[0].message.content)
```

TypeScript, using `together-ai`:

```javascript
import Together from 'together-ai';

const together = new Together();
const completion = await together.chat.completions.create({
  model: 'meta-models/Muse-Glimmer-30B',
  messages: [{ role: 'user', content: 'What are some fun things to do in New York?' }],
});
console.log(completion.choices[0].message.content);
```

The path, the bearer header and the `{model, messages}` body are the OpenAI chat-completions wire format, so an HTTP client you already have pointed at `https://api.together.xyz/v1` will speak to it.

## Prove tool calling works

A response with text in it only proves the endpoint is up. The bar in this cookbook is a tool-calling round trip — expect `tool_calls` in the response, not prose:

```bash
curl -s https://api.together.xyz/v1/chat/completions \
  -H "Authorization: Bearer $TOGETHER_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "meta-models/Muse-Glimmer-30B",
    "messages": [{"role": "user", "content": "Weather in Paris, Tokyo and Cairo?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                     "required": ["city"]}}}]
  }'
```

A working call returns three parallel `tool_calls` entries — one per city — in a single assistant turn, with reasoning available separately on the response object.

You can also drive the same prompt from Together's playground: [api.together.ai/playground/meta-models/Muse-Glimmer-30B](https://api.together.ai/playground/meta-models/Muse-Glimmer-30B).

## Behavior worth knowing

**Sampling.** Defaults come from the checkpoint: `temperature 0.95`, `top_p 1.0`. Greedy decoding loops on this model, so don't set `temperature 0`.

> [!WARNING]
> Stop tokens: `eos_token_id = [<\|end_of_text\|>, <\|eot\|>]`. Never add `<\|eom\|>` to a `stop` parameter — it marks end-of-*message*, the turn continues after it, and stopping there reduces parallel tool calling to near zero. Background: [`../inference-server/README.md`](../inference-server/README.md).

**Weights.** Nothing here downloads them, but if you want the model card or want to run the same model locally instead, it's the public repo the rest of this cookbook uses: [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) on HuggingFace.

## What Together measured

Not on the model page — these are Together's own numbers, reported by Mourya Vangala Srinivasa on 2026-08-10. They are not reproduced here, and they describe Together's serving stack rather than anything you configure.

| Metric | Value |
|---|---|
| TTFT p50 | 307 ms |
| Output tokens/s per request | 105 |

Measured on a dedicated 2×H100-80GB tensor-parallel replica under a sustained 0.4 QPS long-context replay: mean input 58,290 tokens (p99 ≈ 127K), mean output 244 tokens, 1–2 requests in flight. That is long-prompt agentic traffic, not a short-prompt microbenchmark — TTFT at 1K-token inputs is substantially lower.

Together also reports serving the model quantized to fp8 (MLP-only blockwise e4m3, 128×128, with dynamic activations; attention, embeddings and the vision tower stay bf16), chosen after checking quality against a bf16 reference on the same hardware — GPQA-Diamond 83.7 ± 0.9 over 6 runs, against 83.5 on the model card. Precision is not something you select on this endpoint; it is noted because a quantized serving path is worth knowing about when you compare a hosted result to a local bf16 one.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `401` on every request | `TOGETHER_API_KEY` unset, or the key is from a different account | Re-export the key from https://api.together.xyz/settings/api-keys and re-run the `/v1/models` check above |
| Output repeats or loops | `temperature 0` / greedy decoding | Use the checkpoint defaults (`temperature 0.95`, `top_p 1.0`) |
| Multi-tool prompts produce one call per turn instead of parallel calls | `<\|eom\|>` passed as a stop token, or a client that flattens `tools` into the prompt | Send `tools` as a top-level request field and leave `stop` unset |
| Long requests end at 32,768 output tokens | Together caps generation per request | Split the task; the cap guards runaway reasoning loops |
| `503 Service unavailable` minutes after provisioning a dedicated endpoint | Replica still warming (Together reports ~10–15 min for graph compile and fp8 kernel warmup) | Retry with backoff until the first `200`. Serverless is not affected |

## Support

- Model page: https://www.together.ai/models/muse-glimmer
- Docs: https://docs.together.ai
- Pricing: https://www.together.ai/pricing
- Issues: https://www.together.ai/support
- Maintainer: Mourya Vangala Srinivasa (msrinivasa@together.ai)
