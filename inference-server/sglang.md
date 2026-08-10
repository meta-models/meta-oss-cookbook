# SGLang

High-throughput serving for many concurrent local users.

> [!NOTE]
> Status: not verified in this cookbook. SGLang publishes its own [Muse Glimmer cookbook page](https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer) with a matrix of verified hardware/checkpoint combinations, and the commands below are taken from it and from a platform write-up supplied by the SGLang team (Liangsheng Yin, dated 2026-08-10). That page also confirms Muse Glimmer support is **not in an SGLang release** — it lives on the upstream `muse-glimmer` branch ([PR #34262](https://github.com/sgl-project/sglang/pull/34262)). Nothing here has been run for this cookbook. For a path verified here, see [`vllm.md`](vllm.md) or [`llama-cpp.md`](llama-cpp.md).

Every "verified" claim on this page is SGLang's own, on SGLang's hardware. Attribution and dates are in the [platform matrix](#platforms-sglang-has-verified) below.

## Install

Muse Glimmer is not in a released version, so SGLang's own guide builds the branch:

```bash
git clone -b muse-glimmer https://github.com/sgl-project/sglang.git
cd sglang
uv pip install -e "python[all]"
```

> [!WARNING]
> `uv pip install sglang` — the plain released package — does **not** work. [PR #34262](https://github.com/sgl-project/sglang/pull/34262) is still open, so no PyPI release contains the `muse-glimmer` architecture and the server cannot load these checkpoints. Build the branch above, or take the Docker image below. SGLang's own page carries the same warning.

Or take the prebuilt image:

```bash
docker pull lmsysorg/sglang:dev-muse-glimmer
```

Multi-arch — the same tag resolves to amd64 or arm64, so it also covers aarch64 boxes like DGX Spark. Verified present on Docker Hub on 2026-08-10, alongside explicit `dev-muse-glimmer-amd64` and `dev-muse-glimmer-arm64` tags if you need to pin one.

> [!NOTE]
> SGLang images are published under `lmsysorg/sglang`. `vllm/vllm-openai:muse-glimmer` is a real image but it is **vLLM's** server, not SGLang's — see [`vllm.md`](vllm.md). The two get confused because the same configurator page offers both.

Both are from the official page's *Install SGLang* panel. See the [upstream install guide](https://docs.sglang.io/docs/get-started/install) for platform variations.

## Serve

SGLang publishes four checkpoint formats, and which one you use is decided by your GPU rather than by preference:

| Format | Checkpoint | Modality | Verified hardware (per SGLang's matrix) |
|---|---|---|---|
| BF16 | `meta-models/Muse-Glimmer-30B` | text + image | B300, B200, H200, RTX PRO 6000 (96 GB), DGX Spark |
| GGUF Q4_K_M | `meta-models/Muse-Glimmer-30B-GGUF` | text only | B300, RTX 5090 (32 GB) |
| NVFP4 + MXFP8 | `RadixArk/Muse-Glimmer-NVFP4` | text only | B300, B200, RTX 5090, RTX PRO 6000, DGX Spark |
| MLX Q4 variants | `RadixArk/Muse-Glimmer-q4km-gs128-MLX`, `-q4-MLX`, `-q4k-dynamic-MLX` | text only | Apple silicon, 48 GB+ unified memory |

The BF16 checkpoint is the only one that takes images. NVFP4 is ready to serve with no conversion step. Hardware and VRAM labels above are SGLang's, from the command panel on the official page.

### Platforms SGLang has verified

Single accelerator in every row — see [Memory and parallelism](#memory-and-parallelism). Attribution is SGLang's, from the platform write-up their team supplied:

| Platform | Class | Accelerator | Memory | Precisions verified | Verified by |
|---|---|---|---|---|---|
| NVIDIA B300 | server | B300 SXM6 × 1 | 288 GB HBM3e | bf16, nvfp4, q4_k_m | Jimmy Shong, 2026-08-10 |
| NVIDIA B200 | server | B200 × 1 | 192 GB HBM3e | bf16, nvfp4 | Zijie Xia, 2026-08-09 |
| NVIDIA H200 | server | H200 SXM × 1 | 141 GB HBM3e | bf16 | Zijie Xia, 2026-08-09 |
| NVIDIA RTX PRO 6000 | workstation | RTX PRO 6000 × 1 | 96 GB GDDR7 | bf16, nvfp4 | Brayden Zhong, 2026-08-09 |
| NVIDIA DGX Spark | workstation | GB10 × 1 | 128 GB unified | bf16, nvfp4 | Jimmy Shong, 2026-08-09 |
| NVIDIA RTX 5090 | consumer | RTX 5090 × 1 | 32 GB GDDR7 | q4_k_m, nvfp4 | Brayden Zhong, 2026-08-09 |
| Apple Silicon Mac | consumer | M5 Pro (48 GB min) | 64 GB unified | q4_k_m, q4, q4k-dynamic | Alexander Nails and Jimmy Shong, 2026-08-09 |

SGLang publishes throughput and interactivity figures per row on [their page](https://docs.sglang.io/cookbook/autoregressive/Meta/MuseGlimmer). They are not reproduced here: this cookbook does not republish numbers measured on hardware it does not own. Measure your own with:

```bash
python3 -m sglang.bench_serving \
  --backend sglang --host localhost --port 30000 \
  --model meta-models/Muse-Glimmer-30B \
  --dataset-name random --random-input-len 1024 --random-output-len 512 \
  --num-prompts 128 --max-concurrency 64 --flush-cache
```

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

That is the same ~17 GB file [`llama-cpp.md`](llama-cpp.md) serves. Two limits come with it, both from the official page: the path is **text only** — SGLang has no `mmproj` support, so the vision GGUF is unusable — and SGLang describes the path as not optimized and warns at startup. Do not add `--language-model-only` here: the GGUF config is already text-only (`MuseGlimmerForCausalLM`), the flag expects the multimodal class, and the serve is rejected.

### NVFP4

```bash
sglang serve \
  --model-path RadixArk/Muse-Glimmer-NVFP4 \
  --served-model-name muse-glimmer \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --language-model-only \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

SGLang's matrix offers NVFP4 on B300, B200, RTX 5090, RTX PRO 6000 and DGX Spark, but not on H200.

`--language-model-only` is **required** on this path, not optional: the NVFP4 checkpoint ships no vision weights, and without the flag the loader tries to build them and the engine fails to start. SGLang lists that failure as a troubleshooting entry on every NVFP4 platform. On 32 GB cards `--kv-cache-dtype fp8_e4m3` roughly doubles the KV pool (SGLang measured 57k → 115k tokens on a 5090) at no meaningful accuracy cost.

### Flags

| Flag | Why |
|---|---|
| `--reasoning-parser muse` | Splits the thinking channel into `message.reasoning_content`. |
| `--tool-call-parser muse` | Resolves to `MuseGlimmerDetector`; emits structured `message.tool_calls`. |
| `--served-model-name muse-glimmer` | See below. |
| `--mem-fraction-static` | Fraction of VRAM reserved for weights + KV cache. SGLang tunes this per box — see the table. |
| `--kv-cache-dtype fp8_e4m3` | Paired with NVFP4 in every published NVFP4 command. |
| `--language-model-only` | Skips building and loading the vision tower. **Required** on NVFP4, which has no vision weights to build. Optional on BF16, where it frees memory for KV cache and makes image requests fail. Rejected on GGUF, which is already text-only. |

`--served-model-name` is not optional here, and it is the one flag below that SGLang's page does *not* use. SGLang defaults the served name to `--model-path` verbatim (`server_args.py` sets `served_model_name = model_path` when it is unset), so without it the model answers to `meta-models/Muse-Glimmer-30B` — or, for GGUF, to the whole file path. Every recipe in this cookbook sends `"model": "muse-glimmer"` and would not match. SGLang's own examples send the full checkpoint path instead; either is fine as long as client and server agree.

SGLang exposes an OpenAI-compatible endpoint on `:30000/v1` — point a harness at `http://localhost:30000/v1` with the model name you served.

### Memory and parallelism

SGLang's verified cells set `--mem-fraction-static` per box:

| Hardware | BF16 | NVFP4 |
|---|---|---|
| B300 / B200 / H200 / RTX PRO 6000 | 0.85 | 0.85 |
| RTX 5090 | — (GGUF: 0.85) | 0.90 |
| DGX Spark | 0.75 | 0.40 |
| Apple silicon (MLX) | — (MLX: 0.85) | — |

Lower it if the server dies during startup with an out-of-memory error; raise it for more KV cache headroom. Turning on DFlash speculative decoding needs room for the draft model, and SGLang drops the fraction further on DGX Spark to make room (0.65 BF16, 0.38 NVFP4).

DGX Spark is the one box where a too-high fraction takes down more than the server. Its 128 GB is unified and shared with the OS and your clients, so discrete-GPU values around 0.85 OOM the whole machine during load rather than just failing the process. On a 32 GB 5090, 0.90 is SGLang's measured ceiling for NVFP4 — about 1.1 GB free after CUDA graph capture.

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

There is also a **GGUF draft**, `dflash-kquant.gguf`, shipped in the GGUF repo and verified by SGLang on B300. It has a silent failure mode worth knowing about: pass `--speculative-draft-model-quantization gguf` explicitly, or draft quantization resolves before the target's lazy `gguf` load format and the draft is built unquantized from random init. Nothing errors — the draft loads, speculation runs, and the accept length simply sits at 1.0 with no speedup. On every platform except B300 the verified draft is the native `meta-models/Muse-Glimmer-30B-assistant` export, which needs none of this.

DFlash is not available on the MLX backend.

### Apple silicon

Apple silicon uses MLX repacks, not the GGUF files — the MLX backend has no GGUF path. Serve with `SGLANG_USE_MLX=1`:

```bash
SGLANG_USE_MLX=1 SGLANG_MLX_CACHE_LIMIT_GB=8 \
sglang serve \
  --model-path RadixArk/Muse-Glimmer-q4km-gs128-MLX \
  --served-model-name muse-glimmer \
  --trust-remote-code \
  --reasoning-parser muse \
  --tool-call-parser muse \
  --disable-radix-cache \
  --mem-fraction-static 0.85 \
  --host 0.0.0.0 --port 30000
```

SGLang is explicit that the two extras are requirements, not tuning: the sliding-window layers use windowed KV storage, which needs the radix cache off, and without the 8 GB cache cap the MLX buffer cache ratchets the footprint up by roughly 8 GB under concurrent load — enough to push the artifact off a 48 GB machine.

Three MLX artifacts are drop-in `--model-path` swaps: `q4km-gs128` carries the same Q4_K_M codes as the GGUF served elsewhere and is SGLang's default here, `q4` (MLX-native 4-bit) is the fastest, and `q4k-dynamic` the highest-fidelity. SGLang describes `q4km-gs128` as a lossless repack of the vendor Q4_K_M GGUF — every weight keeps the GGUF's exact quantization code, with group scales re-expressed in MLX affine bf16. Text only, and no speculative decoding on this backend. SGLang publishes accuracy and throughput comparisons for these artifacts on its page; they are not reproduced here. On a Mac, [`llama-cpp.md`](llama-cpp.md) is the path verified in this cookbook.

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

**Image requests are rejected.** Only BF16 takes images, and only when served without `--language-model-only`. GGUF, NVFP4 and MLX are text-only regardless of flags.

**The draft model fails to load with a GGUF target.** Add `--speculative-draft-load-format auto`; the draft otherwise inherits the target's `gguf` load format.

**Speculation runs but nothing gets faster, and accept length is pinned at 1.0.** You are using the GGUF draft without `--speculative-draft-model-quantization gguf`, so it was built unquantized from random init. There is no error for this one.

**The NVFP4 engine fails to start.** Add `--language-model-only`. The checkpoint has no vision weights and the loader tries to build them without it.

**The GGUF serve rejects `--language-model-only`.** Drop it. GGUF is text-only by construction, and the flag expects the multimodal class.

**Out of memory at startup.** Lower `--mem-fraction-static`. SGLang's own values are in the table above; DGX Spark runs far lower than a discrete GPU, and turning on DFlash needs another step down.

**A startup warning on the GGUF path.** Expected. SGLang describes its GGUF path as not optimized and warns about it.

**The box itself runs out of memory during load, not just the server.** DGX Spark, with a discrete-GPU `--mem-fraction-static`. Use the values in the table; the OS and your clients share that memory.

## Support

SGLang's, not ours — nothing on this page was verified in this cookbook.

- Issues: https://github.com/sgl-project/sglang/issues
- Docs: https://docs.sglang.io
- Maintainer: Brayden Zhong (brayden.zhong@radixark.ai)

## Next steps

- Serve with a path verified here instead: [`vllm.md`](vllm.md) or [`llama-cpp.md`](llama-cpp.md)
- Learn the loop: [`../agentic-fundamentals/`](../agentic-fundamentals/)
