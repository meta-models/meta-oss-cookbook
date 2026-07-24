# Inference Server

One page per server. Each is a deeper guide than the [quickstart](../quickstart/): full serve command, tool-parser config, and how to verify tool calling actually works.

| Server | Page | Status |
|---|---|---|
| vLLM | [`vllm.md`](vllm.md) | Verified (native `MuseGlimmerForCausalLM`) |
| Ollama | [`ollama.md`](ollama.md) | Pending verified GGUF |
| LM Studio | [`lm-studio.md`](lm-studio.md) | Partner-authored (LM Studio Bionic) |
| SGLang | [`sglang.md`](sglang.md) | Pending |
| llama.cpp | [`llama-cpp.md`](llama-cpp.md) | Verified (GGUF: text, vision, tool calling) |
| Unsloth | [`unsloth.md`](unsloth.md) | Pending verified GGUF |
| ExecuTorch | [`executorch.md`](executorch.md) | Supported upstream (CUDA / MLX; text, vision, tool calling, DFlash) |

Most pages follow the same section order — **Install → Serve → Verify tool calling → Stop tokens → Next steps** — so you can switch between servers and compare the same step side by side. Some add a Troubleshooting section before Next steps.

Partner-authored pages are the deliberate exception. Where a vendor ships its own app and workflow around the model, that page keeps the vendor's shape rather than being pressed into ours: [`lm-studio.md`](lm-studio.md) leads with the Bionic agent before the local API server, and [`unsloth.md`](unsloth.md) covers fine-tuning alongside serving. Read those as the vendor's own guide. The stop-token rule below applies to every server here regardless.

## Get stop tokens right

Muse Glimmer uses a channel-scoped chat format. Three special tokens end a span, and only two of them are stop tokens:

- `<|eot|>`: end of turn. A real stop token.
- `<|end_of_text|>`: end of text. A real stop token.
- `<|eom|>`: end of message. **Not** a stop token. It separates the reasoning block and each non-final parallel tool call; the turn continues after it.

Set `eos_token_id = [<|end_of_text|>, <|eot|>]`. If a server stops on `<|eom|>`, single-turn parallel tool calling drops to near zero. Ship this in the checkpoint's `generation_config.json`.


## Pointing an agent harness at the endpoint

Every server here exposes an OpenAI-compatible `/v1` endpoint, so any harness that speaks that API works. There are only two things to get right:

1. Point the harness at the endpoint (`http://localhost:8000/v1` for vLLM) with the model name you served.
2. Make sure the harness doesn't override the stop tokens to include `<|eom|>`.

Tool calls are parsed server-side, so the harness receives a standard `tool_calls` array and needs no Muse Glimmer-specific handling.
