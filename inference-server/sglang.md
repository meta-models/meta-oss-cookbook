# SGLang

High-throughput serving for many concurrent local users.

> [!NOTE]
> Status: not verified in this cookbook. SGLang publishes its own [Muse Glimmer cookbook page](https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer) with a matrix of verified hardware/checkpoint combinations, and the commands below are taken from it. That page also confirms Muse Glimmer support is **not in an SGLang release** — it lives on the upstream `muse-glimmer` branch ([PR #34262](https://github.com/sgl-project/sglang/pull/34262)). Nothing here has been run for this cookbook. For a path verified here, see [`vllm.md`](vllm.md) or [`llama-cpp.md`](llama-cpp.md).

## Install

Muse Glimmer is not in a released version, so SGLang's own guide builds the branch:

```bash
git clone -b muse-glimmer https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[all]"
```

Or take the prebuilt image:

```bash
docker pull lmsysorg/sglang:dev-muse-glimmer
```

Both are from the official page's *Install SGLang* panel. See the [upstream install guide](https://docs.sglang.io/docs/get-started/install) for platform variations.

## Serve

SGLang publishes four checkpoint formats, and which one you use is decided by your GPU rather than by preference:

| Format | Checkpoint | Modality | Verified hardware (per SGLang's matrix) |
|---|---|---|---|
| BF16 | `meta-models/Muse-Glimmer-30B` | text + image | B200, H200, RTX PRO 6000 (96 GB), DGX Spark |
| GGUF Q4_K_M | `meta-models/Muse-Glimmer-30B-GGUF` | text only | RTX 5090 (32 GB) |
| NVFP4 + MXFP8 | `RadixArk/Muse-Glimmer-NVFP4` | text only | B200, RTX 5090, RTX PRO 6000, DGX Spark |
| MLX Q4 variants | `RadixArk/Muse-Glimmer-*-MLX` | text only | Apple silicon, 48 GB+ unified memory |

The BF16 checkpoint is the only one that takes images. NVFP4 is ready to serve with no conversion step. Hardware and VRAM labels above are SGLang's, from the command panel on the official page.

### BF16 — the multimodal path

```bash
sglang serve \
  --model-path meta-models/Muse-Glimmer-30B \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

`python -m sglang.launch_server` with the same flags is the equivalent long form; both entrypoints exist on the branch.

### GGUF Q4_K_M — 32 GB cards

This is the one combination SGLang publishes for a 32 GB card. `--model-path` points at the **file**, not the repo:

```bash
sglang serve \
  --model-path meta-models/Muse-Glimmer-30B-GGUF/muse-glimmer-30B-kquant-17gb.gguf \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

That is the same ~17 GB file [`llama-cpp.md`](llama-cpp.md) serves. Two limits come with it, both from the official page: the path is **text only** — SGLang has no `mmproj` support, so the vision GGUF is unusable — and SGLang describes the path as not optimized and warns at startup.

### NVFP4

```bash
sglang serve \
  --model-path RadixArk/Muse-Glimmer-NVFP4 \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

SGLang's matrix offers NVFP4 on B200, RTX 5090, RTX PRO 6000 and DGX Spark, but not on H200.

### Flags

| Flag | Why |
|---|---|
| `--reasoning-parser muse` | Splits the thinking channel into `message.reasoning_content`. |
| `--tool-call-parser muse` | Resolves to `MuseGlimmerDetector`; emits structured `message.tool_calls`. |
| `--served-model-name muse-glimmer` | See below. |
| `--mem-fraction-static` | Fraction of VRAM reserved for weights + KV cache. SGLang tunes this per box — see the table. |
| `--kv-cache-dtype fp8_e4m3` | Paired with NVFP4 in every published NVFP4 command. |
| `--language-model-only` | BF16 only. Skips building and loading the vision tower, freeing memory for KV cache. Image requests are then rejected. |

`--served-model-name` is not optional here, and it is the one flag below that SGLang's page does *not* use. SGLang defaults the served name to `--model-path` verbatim (`server_args.py` sets `served_model_name = model_path` when it is unset), so without it the model answers to `meta-models/Muse-Glimmer-30B` — or, for GGUF, to the whole file path. Every recipe in this cookbook sends `"model": "muse-glimmer"` and would not match. SGLang's own examples send the full checkpoint path instead; either is fine as long as client and server agree.

SGLang exposes an OpenAI-compatible endpoint on `:30000/v1` — point a harness at `http://localhost:30000/v1` with the model name you served.

### Memory and parallelism

SGLang's verified cells set `--mem-fraction-static` per box:

| Hardware | BF16 | NVFP4 |
|---|---|---|
| B200 / H200 / RTX PRO 6000 | 0.85 | 0.85 |
| RTX 5090 | — (GGUF: 0.85) | 0.90 |
| DGX Spark | 0.75 | 0.40 |

Lower it if the server dies during startup with an out-of-memory error; raise it for more KV cache headroom. Turning on DFlash speculative decoding needs room for the draft model, and SGLang drops the fraction further on DGX Spark to make room (0.65 BF16, 0.38 NVFP4).

Every verified combination on the official page is **single node with no tensor parallelism** — no published command passes `--tp-size`. The page's Playground exposes TP 1 and 2 as unverified options. Muse Glimmer at 30B fits one card in each of these formats, so treat multi-GPU as untested territory rather than the default.

### Speculative decoding (DFlash)

The draft model `meta-models/Muse-Glimmer-30B-assistant` serves as published, with no conversion step. Add:

```bash
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path meta-models/Muse-Glimmer-30B-assistant \
  --speculative-dflash-block-size 5
```

Two target-specific additions, both called out by SGLang:

- **GGUF target** also needs `--speculative-draft-load-format auto`. Without it the draft inherits the `gguf` load format from the target and the loader rejects the draft directory.
- **NVFP4 target** also takes `--speculative-draft-model-quantization fp8`.

DFlash is not available on the MLX backend.

### Apple silicon

Apple silicon uses MLX repacks, not the GGUF files — the MLX backend has no GGUF path. Serve `RadixArk/Muse-Glimmer-q4km-gs128-MLX` (or the `q4` / `q4k-dynamic` siblings) with `SGLANG_USE_MLX=1` and `SGLANG_MLX_CACHE_LIMIT_GB=8`, adding `--trust-remote-code --disable-radix-cache` — SGLang notes radix cache must stay off because the sliding-window layers use windowed KV storage. Text only. SGLang publishes accuracy and throughput comparisons for these artifacts on its page; they are not reproduced here. On a Mac, [`llama-cpp.md`](llama-cpp.md) is the path verified in this cookbook.

## Verify tool calling

Not verified here. SGLang does ship the parser — `--tool-call-parser muse` resolves to `MuseGlimmerDetector`, and the official page documents tool calling as supported and passes the flag in every published command — but it exists only on the `muse-glimmer` branch, not in `main` or any release, and no tool-calling round has been run against it for this cookbook.

> [!NOTE]
> SGLang's page says the `muse` reasoning and tool-call parsers are "enabled by default". On the branch as it stands, `tool_call_parser` defaults to `None` in `server_args.py` and there is no Muse Glimmer rule in the chat-template auto-detection table. Pass both flags explicitly, as every command on SGLang's own page does.

Once a server is up, this is the check:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"muse-glimmer","messages":[{"role":"user","content":"What is the weather in Paris in celsius? Use the tool."}],
       "tools":[{"type":"function","function":{
         "name":"get_weather","description":"Get current weather for a city.",
         "parameters":{"type":"object","properties":{
           "city":{"type":"string"},"units":{"type":"string","enum":["celsius","fahrenheit"]}},
           "required":["city"]}}}],
       "tool_choice":"auto","temperature":0}'
```

A working parser returns a `tool_calls` array with `get_weather(city="Paris", units="celsius")` and the thinking under `reasoning_content`. Until that is confirmed here, point agentic recipes at vLLM ([`vllm.md`](vllm.md)).

## Stop tokens

Muse Glimmer needs `eos_token_id = [<|end_of_text|>, <|eot|>]`. Never stop on `<|eom|>`. See [`README.md`](README.md#get-stop-tokens-right).

There is no serve flag for this. SGLang builds its stop set at load time by unioning `eos_token_id` from the checkpoint's `config.json` and its `generation_config.json`, so the checkpoint is where you set it — as the README says. Because it is a union, nothing subtracts from that set afterwards: if either file lists `<|eom|>`, SGLang stops on it and the fix is the checkpoint metadata. A request can add ids via `stop_token_ids` on `/v1/chat/completions`, but cannot remove them.

## Troubleshooting

**Requests 404 or report an unknown model.** The served name does not match what the client sends. Check `/v1/models`, and see the `--served-model-name` note above.

**Image requests are rejected.** Either you passed `--language-model-only`, or you are on GGUF, NVFP4 or MLX — all three are text only. Images need the BF16 checkpoint without that flag.

**The draft model fails to load with a GGUF target.** Add `--speculative-draft-load-format auto`; the draft otherwise inherits the target's `gguf` load format.

**Out of memory at startup.** Lower `--mem-fraction-static`. SGLang's own values are in the table above; DGX Spark runs far lower than a discrete GPU, and turning on DFlash needs another step down.

**A startup warning on the GGUF path.** Expected. SGLang describes its GGUF path as not optimized and warns about it.

## Next steps

- Serve with a path verified here instead: [`vllm.md`](vllm.md) or [`llama-cpp.md`](llama-cpp.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
