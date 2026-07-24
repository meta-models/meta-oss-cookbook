# SGLang

High-throughput serving for many concurrent local users.

> [!NOTE]
> Status: pending verified content. This page is seeded with the shared section order so a verified quickstart can drop straight in. For tool calling today, use [`vllm.md`](vllm.md).

## Install

Install SGLang per the [upstream instructions](https://docs.sglang.ai/).

## Serve

```bash
python -m sglang.launch_server --model-path meta-models/Muse-Glimmer-30B --port 30000
```

## Verify tool calling

Not yet verified. Until it is, point agentic recipes at vLLM ([`vllm.md`](vllm.md)).

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. See [`README.md`](README.md#get-stop-tokens-right).

## Troubleshooting

Filled in once a verified run lands.

## Next steps

- Serve with the verified path instead: [`vllm.md`](vllm.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
