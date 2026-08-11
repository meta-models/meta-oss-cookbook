# Computer-Use Web

Muse Glimmer drives a real Mac to shop on a website — it opens Safari, navigates, and clicks its way to a product, using nothing but screenshots and a mouse.

It works the way a person does: look at the screen, decide where to click, click there. No element tree, no accessibility labels, no API — just pixels and coordinates.

## Recipe banner

| | |
|---|---|
| Precision | Quantized GGUF (Q4KM K-quant, ~17 GB on disk) |
| Model server | llama.cpp (upstream, release `b10353` or newer) |
| Offline? | Model runs locally; the task itself browses the web |
| Memory observed | ~25 GB — model + vision projector + 128K KV cache |
| Requires | [metacua](https://github.com/meta-models/meta-model-cookbook/tree/main/03_use_cases/13_macos_cua) · macOS with Safari |

> [!CAUTION]
> `metacua` takes real control of your Mac — it moves the pointer, types, and clicks whatever the model decides, with no confirmation step. Keep the terminal reachable so you can `Ctrl+C`, and don't type in other apps while it runs. Web pages can also prompt-inject the agent, so prefer sites you trust.

## Setup

### 1. Serve Muse Glimmer with vision

Follow [`../../inference-server/llama-cpp.md`](../../inference-server/llama-cpp.md) for the download and the build — you need `muse-glimmer-30B-kquant-17gb.gguf` plus `mmproj-kquant.gguf` for image input, and llama.cpp release [`b10353`](https://github.com/ggml-org/llama.cpp/releases/tag/b10353) or newer. Upstream supports Muse Glimmer, so no fork is needed; `b10344` and older refuse to load these checkpoints with `unknown model architecture: 'muse-glimmer'`. Then:

```bash
./build/bin/llama-server \
  -m ./muse-glimmer/muse-glimmer-30B-kquant-17gb.gguf \
  --mmproj ./muse-glimmer/mmproj-kquant.gguf \
  -a muse-glimmer \
  -ngl 99 -c 131072 -np 1 \
  --host 127.0.0.1 --port 8080 --api-key muse-glimmer \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"high"}'
```

- `-a muse-glimmer` is the name the API answers to, and it has to match the `--model muse-glimmer` set in step 4. Without it the alias is the checkpoint path and the request does not match.
- `-np 1` keeps the whole context in one slot; the agent needs it.
- `--jinja` applies the template embedded in the GGUF. It is what wires up tool calling and the correct stop set — `<|end_of_text|>` and `<|eot|>`, never `<|eom|>`.
- `--chat-template-kwargs '{"reasoning_strength":"high"}'` is where reasoning length is set; the levels are `low`, `medium`, `high` and `xhigh`. **metacua's `--effort` flag has no effect here** — llama.cpp does not read the Responses API's `reasoning.effort` field, so the server flag is the only knob. See [Controlling reasoning length](../../inference-server/llama-cpp.md#controlling-reasoning-length).
- This recipe wants `high`. A misclick navigates somewhere else and costs several steps to undo, so deliberation is worth paying for — unlike tasks whose environment validates each action for free.

### 2. Install metacua

```bash
git clone https://github.com/meta-models/meta-model-cookbook.git
cd meta-model-cookbook
git fetch origin pull/11/head:pr-11 && git switch pr-11
cd 03_use_cases/13_macos_cua/python
python3 -m venv .venv && .venv/bin/python -m pip install -U pip
.venv/bin/pip install -e .
```

[meta-model-cookbook#11](https://github.com/meta-models/meta-model-cookbook/pull/11) is still open, which is why the checkout above moves onto it instead of staying on `main`. On `main` the Python backend returns the screenshot inside the tool result, which llama.cpp rejects with `Output of tool call should be 'Input text'` on the second step of every run. Once it merges, clone `main` and drop the fetch/switch line.

### 3. Grant macOS permissions

metacua posts real mouse and keyboard events, so the **terminal you launch it from** needs both grants — they attach to the terminal, not to metacua:

```bash
.venv/bin/metacua permissions --prompt
```

Approve the dialogs, then **quit and reopen the terminal** — macOS only reads these at process start. Verify:

```bash
.venv/bin/metacua permissions   # both lines must say GRANTED
```

### 4. Point metacua at the local server

```bash
.venv/bin/metacua configure --base-url http://127.0.0.1:8080/v1 --api-key muse-glimmer --model muse-glimmer
```

> [!IMPORTANT]
> `MODEL_API_KEY` in your environment **overrides** this saved key, and metacua will send it to a server that only knows `muse-glimmer`, giving `401 Invalid API Key`. Pass `--api-key muse-glimmer` per run, or `unset MODEL_API_KEY` in that shell.

## Run it

```bash
.venv/bin/metacua agent --api-key muse-glimmer --screenshot-scale 0.5 \
  --goal "Open Safari, go to meta.com, click the Shop button, then find the product called Starfire Kylie Edition. When you can see it on screen, call computer.stop and describe what you found."
```

Watch it work: Spotlight opens Safari, the URL gets typed and submitted, the page is read, and the pointer travels to each target.

- `--screenshot-scale 0.5` halves a Retina capture that is otherwise re-sent every step. Coordinates are unaffected — the action space always uses the display's full logical size.
- Give a **goal with a stop condition**, as above. "Find X" alone leaves the model unsure when it is done.
- Add `--max-steps 60` for longer errands; the default cap is 40.

## The loop

Each step is the same four beats:

1. **See** — screenshot the primary display
2. **Send** — screenshot plus tool definitions go to `POST {base}/responses`
3. **Act** — the model returns a `computer.computer` call, executed with native macOS events
4. **Look again** — screenshot the new state and repeat, until the model calls `computer.stop`

Three things worth knowing:

- **Coordinates are normalized 0-1000**, origin top-left, converted to display pixels before execution. The model never sees your resolution.
- **The pointer really moves.** Unlike accessibility-driven automation, these are synthetic Quartz events, so the cursor travels to each target and the demo is visible on screen.
- **Primary display only.** Coordinates outside it are clamped to the edge, so move the target window to the main screen first.

`type` does not submit — the model must follow it with a `key` action. Traces land in `~/.metacua/traces/`; screenshots are stripped to size placeholders, but goal and message text are kept, so treat that directory as sensitive.

## Next steps

- Serve the endpoint this drives: [`../../inference-server/llama-cpp.md`](../../inference-server/llama-cpp.md)
- The loop underneath: [`../../agentic-fundamentals/`](../../agentic-fundamentals/)
