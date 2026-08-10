# SGLang

High-throughput serving for many concurrent local users.

> [!NOTE]
> Status: pending verified content. Muse Glimmer support is not in an SGLang release — it lives on the upstream `muse-glimmer` branch ([PR #34262](https://github.com/sgl-project/sglang/pull/34262)), and nothing below has been verified against a run in this cookbook. For tool calling today, use [`vllm.md`](vllm.md).

## Install

Install SGLang per the [upstream instructions](https://docs.sglang.ai/).

Muse Glimmer is not in a released version, so SGLang's own Muse Glimmer guide builds the branch:

```bash
git clone -b muse-glimmer https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[all]"
```

## Serve

```bash
python -m sglang.launch_server \
  --model-path meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --port 30000
```

SGLang exposes an OpenAI-compatible endpoint on `:30000/v1` — point a harness at `http://localhost:30000/v1` with the model name you served.

`--served-model-name` is not optional here. SGLang defaults the served name to `--model-path` verbatim, so without it the model answers to `meta-models/Muse-Glimmer-30B`, and every recipe in this cookbook sends `"model": "muse-glimmer"` and won't match.

## Verify tool calling

Not yet verified. SGLang does ship a Muse Glimmer tool-call parser — `--tool-call-parser muse` resolves to `MuseGlimmerDetector` — but only on the `muse-glimmer` branch, not in `main` or any release, and no tool-calling round has been run against it for this cookbook. Until one lands, point agentic recipes at vLLM ([`vllm.md`](vllm.md)).

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. See [`README.md`](README.md#get-stop-tokens-right).

There is no serve flag for this. SGLang builds its stop set at load time by unioning `eos_token_id` from the checkpoint's `config.json` and its `generation_config.json`, so the checkpoint is where you set it — as the README says. Because it is a union, nothing subtracts from that set afterwards: if either file lists `<|eom|>`, SGLang stops on it and the fix is the checkpoint metadata. A request can add ids via `stop_token_ids` on `/v1/chat/completions`, but cannot remove them.

## Troubleshooting

Filled in once a verified run lands.

## Next steps

- Serve with the verified path instead: [`vllm.md`](vllm.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
