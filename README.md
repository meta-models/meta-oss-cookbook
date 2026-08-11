# Muse Glimmer Cookbook

Clone it, run it on the GPU you already own, and ship a working agent — one that plans, calls tools, and self-corrects — in a single sitting, fully offline.

Muse Glimmer is an open-weight model built for local agentic work on a single GPU. This cookbook is how you go from weights to a running agent.

## What you build here

We built Muse Glimmer keeping 2 principles in mind:

- **Local-first**: Inference runs on your own hardware, so nothing leaves your machine. No API keys, no gated access, no hosted dependency — everything works fully offline.
- **Agentic-first**: Designed for the full tool-use loop — the model plans, calls a tool, feeds the result back, and self-corrects, taking multiple steps autonomously until it reaches the goal.

## What you'll have by the end
A local agent running on your own machine that completes a real multi-step task, offline, in a single sitting, starting from a fresh install.

## Pick your path

| You want to | Go to |
|---|---|
| Run the model with a single command | [`quickstart/`](quickstart/) |
| Learn the tool-use loop (chat template, function calling, the agent loop) | [`agentic-fundamentals/`](agentic-fundamentals/) |
| Ship a flagship agent (structured output, reasoning control, a triage pipeline) | [`recipes/`](recipes/) |
| Serve Muse Glimmer (vLLM, Ollama, LM Studio, SGLang, llama.cpp) | [`inference-server/`](inference-server/) |
| Deploy on specific partner hardware, and see which precisions it supports | [`platform/`](platform/) |
| Call a hosted API instead of running the model yourself (needs a provider API key) | [`hosted/`](hosted/) |

## How every recipe is built

Each recipe follows the same contract so you always know what you're getting:

- **Runs end to end, offline**: Any network use is optional and flagged.
- **Recipe banner up top**: Max VRAM we actually observed, precision, and model server before you run anything. We don't publish numbers for hardware we haven't run on.
- **bf16 on vLLM**: Every recipe is verified at bf16 and served with vLLM. Quantized builds work; we just don't verify each recipe across every quant scheme.
- **Copy-paste first**: One command to run. The explanation comes after.
- **Ends with "make it yours"**: The extension hook, plus troubleshooting.

## About the model

| | |
|---|---|
| Family | Llama-derived dense decoder (Gemma2-style text stack with Muse Glimmer deltas). |
| Size | 30B dense decoder. |
| Suggested GPU | Fits a single 24–32 GB GPU when quantized (Q4/INT4, ~16–17 GB); bf16 needs ~60 GB (an 80 GB card, or sharded across several GPUs). |
| Context | 131072 tokens (128K). |
| Modalities | Text and image in, text out, plus tool calling — all supported today, through a dedicated ~1.8B perception encoder. Server support for image input varies; each [`inference-server/`](inference-server/) page states where it stands. Video is not a supported input: the model card processes it as individual frames and is not explicitly optimized for it. |
| Tool format | An XML-ish `<atem:function_calls>` block. See [`agentic-fundamentals/`](agentic-fundamentals/). |
| Chat framing | Channel-scoped `<\|start\|>role<\|message\|>…<\|eot\|>` with a `to=self` reasoning channel. |
| License | See the [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B) on HuggingFace. |

## Status

This cookbook is under active construction. Sections land incrementally, and each folder's README states where it stands.

Contributions welcome. Every recipe follows the shared template in [`assets/RECIPE_TEMPLATE.md`](assets/RECIPE_TEMPLATE.md).

## License

Please refere to the [License](LICENSE)
