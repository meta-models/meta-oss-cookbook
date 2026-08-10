# Unsloth

![Muse Glimmer 30B](../assets/muse-glimmer-30b-wordmark.png)

Run, fine-tune, and serve Muse Glimmer locally via a GUI interface.

Unsloth is an open-source desktop app for running and training Muse Glimmer on local hardware. The app enables OpenAI-compatible API endpoints, agent connections, tool calling, web search, remote access, model exporting, and more.

> [!NOTE]
> Status: pending a Muse Glimmer-specific Unsloth checkpoint, GGUF build, and verified tool-calling run. The Unsloth workflows below are current, but replace the model placeholders only with the official Meta or Unsloth release artifacts.

## Install

### Download Unsloth

**[Download Unsloth](https://unsloth.ai/download)** for macOS, Windows, Linux, or WSL:

- [Download for macOS](https://unsloth.ai/download/mac)
- [Download for Windows](https://unsloth.ai/download/windows)
- [Download for Linux and WSL](https://unsloth.ai/download/linux)

### Manual installation

Use the platform download links at the beginning of this page, then install and launch the app.

You can also install from a terminal. On macOS, Linux, or WSL:

```bash
curl -fsSL https://unsloth.ai/install.sh | sh
```

On Windows PowerShell:

```powershell
irm https://unsloth.ai/install.ps1 | iex
```

Launch it from the installed app shortcut, or run:

```bash
unsloth studio -p 8888
```

See the [Unsloth docs](https://unsloth.ai/docs/) and [platform requirements](https://unsloth.ai/docs/get-started/beginner-start-here/unsloth-requirements).

## Fine-tuning

You can use the Unsloth GUI's 'Train' tab to get started with Muse Glimmer fine-tuning.

For the code-based fine-tuning package, create an isolated environment and install the current release:

```bash
uv venv unsloth_env --python 3.13
source unsloth_env/bin/activate
uv pip install unsloth --torch-backend=auto
```

On Windows, activate the environment with `.\unsloth_env\Scripts\Activate.ps1` in PowerShell. See the [Unsloth installation guide](https://unsloth.ai/docs/get-started/install) for NVIDIA, AMD, Intel, and platform-specific instructions.

## Run and Serve

### Unsloth Desktop

1. Open **Select model** or **Model Hub**, then search for Muse Glimmer and download GGUF, NVFP4 or format of your choice. Select the official Meta checkpoint or Unsloth build. For GGUF, choose a quantization that fits your VRAM, unified memory, and system RAM while leaving room for KV cache.
2. Once the model is downloaded, you can chat with it directly.
3. To serve the model, create an API key under **Settings > API**. Unsloth exposes OpenAI-compatible `/v1/chat/completions` and `/v1/responses` endpoints, plus Anthropic-compatible `/v1/messages`.

Keep localhost binding unless remote access is needed. Keep API keys private and review enabled tools before exposing the server.

For LAN access:

```bash
unsloth studio -H 0.0.0.0 -p 8888
```

For HTTPS tunnel access:

```bash
unsloth studio --secure -p 8888
```

To connect Muse Glimmer to an agentic tool, load Muse Glimmer in Unsloth and run:

```bash
unsloth start claude
```

Change ```claude``` to the [supported agents](https://unsloth.ai/docs/integrations/unsloth-start).

### Unsloth CLI

Once official Muse Glimmer GGUF artifacts are published:

```bash
unsloth run --model unsloth/muse-glimmer-30b-GGUF:UD-Q4_K_XL
```
