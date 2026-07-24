# Ollama

One command to a local Muse Glimmer, from a prebuilt quantized GGUF.

> [!NOTE]
> Status: pending verified content. This page is seeded with the shared section order so a verified quickstart can drop straight in. For tool calling today, use [`vllm.md`](vllm.md).

## Install

Install Ollama from [ollama.com/download](https://ollama.com/download).

## Serve

```bash
ollama run muse-glimmer     # exact tag TBD
```

Ollama exposes an OpenAI-compatible endpoint on `:11434/v1`.

## Verify tool calling

Not yet verified. Ollama's default templates may not emit Muse Glimmer's tool-block framing — until that's confirmed, point agentic recipes at vLLM ([`vllm.md`](vllm.md)).

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. See [`README.md`](README.md#get-stop-tokens-right).

## Troubleshooting

Filled in once a verified run lands.

## Next steps

- Serve with the verified path instead: [`vllm.md`](vllm.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
