# Agentic Fundamentals

This is the loop that makes Muse Glimmer an agent. Everything here is a runnable, tested, pure-Python agent loop on the HuggingFace weights — no server, no API keys. Read it top to bottom to learn how Muse Glimmer does tool use, then run `run_agent.py` and watch it work on your own code.

## What's in this folder

| File | What it is |
|---|---|
| [`response_parser.py`](response_parser.py) | Turns a raw Muse Glimmer turn into reasoning / tool calls / final answer. Self-checking: run it directly. |
| [`agent_loop.py`](agent_loop.py) | The reference agent loop and a small `@tool` registry. |
| [`run_agent.py`](run_agent.py) | A worked example: the agent reviews real code on your disk and writes a report. |
| `workspace/` | Sandbox the agent writes files into (gitignored). |

## Run it

```bash
pip install "transformers>=5.15" accelerate torch
# the Muse Glimmer model type is `muse-glimmer`; use a transformers build that registers it

# 1. Parser checks — no model, no GPU:
python response_parser.py

# 2. The full agent loop against your Muse Glimmer checkpoint:
python run_agent.py --model meta-models/Muse-Glimmer-30B
# or: MUSE_GLIMMER_MODEL_PATH=/path/to/checkpoint python run_agent.py
```

## Recipe banner

| | |
|---|---|
| Max VRAM observed | ~60 GB (bf16, `device_map="auto"`, batch 1) |
| Precision | bf16 |
| Model server | None — HF Transformers in-process. This is the learning path. |
| Offline? | Yes, fully. The agent only touches your local filesystem. |

For production latency, serve the same checkpoint with vLLM ([`../inference-server/vllm.md`](../inference-server/vllm.md)). The loop logic is identical.

## The Muse Glimmer chat format

Muse Glimmer trains on a channel-scoped chat format. Every assistant span is addressed to a recipient with `to=<recipient>`:

- `to=self`: the model's private reasoning channel. Ends with `<|eom|>`.
- `to=<tool>.<fn>`: a tool-call channel. Ends with `<|eom|>` (non-final) or `<|eot|>` (final).
- `to=user`: the final answer. Ends with `<|eot|>`.

The framing tokens:

| Token | Meaning | Stop token? |
|---|---|---|
| `<\|start\|>` | Begin a message header (`<\|start\|>role`) | — |
| `<\|message\|>` | Separate the header from the body | — |
| `<\|eom\|>` | End of message: the turn continues | No |
| `<\|eot\|>` | End of turn | Yes |
| `<\|end_of_text\|>` | End of text | Yes |

> [!WARNING]
> The one gotcha that breaks tool calling: `<|eom|>` is not a stop token. It separates the reasoning block and each non-final parallel tool call. Stop generation on `<|eom|>` and multi-tool turns collapse. Correct stop set: `[<|eot|>, <|end_of_text|>]`. `agent_loop.py` sets exactly this.

### The tool-call format

When Muse Glimmer calls a tool, it emits an ATEM block — ATEM is what we named Muse Glimmer's tool-block format. It's XML-ish, and parsed with regexes rather than a strict XML parser:

```
<atem:function_calls>
<atem:invoke name="get_weather">
<atem:parameter name="city">Paris</atem:parameter>
<atem:parameter name="units">celsius</atem:parameter>
</atem:invoke>
</atem:function_calls>
```

Scalar and string parameters are written as-is; lists and objects are JSON. The parser JSON-decodes each parameter value when it can, and keeps the raw string otherwise.

Channel scoping is the subtle part: an `<atem:invoke>` echoed inside a `to=self` reasoning block or a `to=user` answer is not a real call. `MuseGlimmerATEMParser` strips those spans before scanning for invokes, mirroring the vLLM tool parser.

You mostly don't have to care about any of this: `response_parser.py` handles it, and vLLM handles it server-side. It matters when you're debugging why a tool call didn't fire.

## The agent loop

```diagram
  build messages (system + user + tool schemas)
        |
        v
  +----------------------------------------------+
  | apply_chat_template(tools=...)  --> generate  |
  |       stop on <|eot|> / <|end_of_text|>       |
  +----------------+------------------------------+
                   v
        MuseGlimmerATEMParser.parse(raw)
                   |
        +----------+-----------+
        v                      v
  has tool calls?         no tool calls
        |                      |
        v                      v
  execute each tool       return to=user
  append tool results     final answer  (done)
  loop  -------------+
        ^            |
        +------------+   (self-correct on tool errors)
```

Two things the reference loop handles for you:

1. `arguments` must be a dict. When you feed the assistant's tool call back into `apply_chat_template`, `tool_calls[].function.arguments` has to be a mapping, not a JSON string: the Muse Glimmer jinja template raises otherwise. `agent_loop.py` passes the parsed dict.
2. Self-correction comes free. Tools return their errors as JSON strings instead of raising, so a failed call becomes a tool result the model reads and recovers from on the next step.

## The worked example

`run_agent.py` points the agent at a real directory and asks it to review the code:

> Review the Python modules in this project. First list them, then read each one. Write a markdown code review to `code_review.md` containing a table of every module with its line count and a one-line description, and a `Findings` section. Finally, tell me which module carries the most complexity and why.

It gets three local tools — `list_files` and `read_file` (both scoped to the project root) and `write_file` (scoped to `workspace/`) — and nothing else. No network, no fixtures: it reads your actual files.

By default the project root is this cookbook, so it works straight after a clone. Point it anywhere:

```bash
python run_agent.py --root ~/code/my-project
```

A run has this shape — plan, list, read each file, write the artifact, then answer:

```text
=== step 1 ===
[reasoning] I should see which Python modules exist before reading anything.
[tool call] list_files({"pattern": "*.py"})
[tool result] {"root": "...", "count": 3, "files": [{"path": "agentic-fundamentals/agent_loop.py", ...

=== step 2 ===
[tool call] read_file({"path": "agentic-fundamentals/agent_loop.py"})
[tool result] {"path": "agentic-fundamentals/agent_loop.py", "truncated": false, "content": "...

... one read_file per module ...

=== step 5 ===
[reasoning] I have all three modules. Now I can write the review.
[tool call] write_file({"filename": "code_review.md", "content": "# Code Review ...
[tool result] {"ok": true, "path": ".../workspace/code_review.md", "bytes": 1841}

=== step 6 ===
[final] Reviewed 3 modules and wrote workspace/code_review.md. agent_loop.py carries
        the most complexity: it owns generation, the message-history contract, and
        tool dispatch in a single class.
```

The artifact lands in `workspace/code_review.md`. Step count and wording vary run to run — what's stable is the shape: plan, gather with tools, act, answer.

The error paths matter as much as the happy path. Ask for a file that doesn't exist and the tool returns `{"error": "no such file: ...", "hint": "call list_files to see valid paths"}` instead of raising. The model reads that and corrects on the next step. That's the whole self-correction mechanism.

## Make it yours

`run_agent.py` is the template. The three things you'll change first:

- **Add a tool**: use the `@tools.tool(description=..., parameters=...)` decorator. Any Python function becomes a tool; return a string or a JSON-able object. Write the `description` carefully — the model reads it verbatim to decide what to call.
- **Point it at your own code**: `--root ~/code/my-project`. The read tools are confined to that root; writes stay in `workspace/`.
- **Change the task**: `--task "..."` with a goal that needs several tools. Give it more room with `--max-steps`.

To swap the backend for lower latency, point the same loop at a vLLM endpoint. The parser and stop-token rules are identical.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TemplateError: arguments ... must be a dict` | Passed `arguments` as a JSON string | Pass the parsed dict into `apply_chat_template` (the loop does this). |
| Model never stops or repeats | Wrong stop tokens | Use `eos_token_id=[<\|eot\|>, <\|end_of_text\|>]`; never `<\|eom\|>`. |
| Tool calls arrive as plain text | Not parsing the tool block | Use `MuseGlimmerATEMParser` (or vLLM `--tool-call-parser muse_glimmer`). |
| An echoed example call gets executed | Missing channel scoping | Strip `to=self` / `to=user` spans before scanning invokes (the parser does this). |
| Agent stops before writing the report | Hit `--max-steps` | Raise `--max-steps`; one read per file adds up on large projects. |
| OOM loading bf16 | ~60 GB doesn't fit | Use `device_map="auto"` across GPUs, or the vLLM path with tensor parallelism. |

## Next steps

- Ship a flagship agent: [`../recipes/`](../recipes/)
- Serve it for a team: [`../inference-server/vllm.md`](../inference-server/vllm.md)
