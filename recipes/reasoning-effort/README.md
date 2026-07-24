# Reasoning Effort

Muse Glimmer thinks before it answers. This recipe shows you how to control how much, and what each setting costs you in tokens and latency.

Reasoning is the lever people reach for last and should reach for first. On a hard tool-selection or multi-step problem it's often the difference between right and wrong — and on an easy one it's pure latency you're paying for nothing.

## Recipe banner

| | |
|---|---|
| Max VRAM observed | ~60 GB (bf16, batch 1) — the serve itself; this recipe adds no footprint |
| Precision | bf16 |
| Model server | vLLM |
| Offline? | Yes, fully |

## Quickstart

```bash
pip install openai
# with Muse Glimmer already served (see ../../inference-server/vllm.md):
python reasoning_effort.py

# also print the private to=self channel:
python reasoning_effort.py --show-reasoning
```

## What just happened

The same problem ran four times at `low`, `medium`, `high`, and `xhigh`, and you got a table:

```text
[MISS] low      reasoning=    52  completion=    61     1.4s  answer='24'
[ok  ] medium   reasoning=   180  completion=   194     3.1s  answer='60'
[ok  ] high     reasoning=   351  completion=   366     5.6s  answer='60'
[ok  ] xhigh    reasoning=   584  completion=   601     9.2s  answer='60'

--- pick a setting ---
cheapest correct : medium (194 completion tokens)
fastest correct  : medium (3.1s)
```

Numbers vary by run and by problem — the point is the shape. Reasoning tokens climb steeply with effort, the answer stops improving well before the top setting, and everything past that point is latency you're buying for free.

Two things worth knowing:

- **`reasoning_content` is a separate field.** Muse Glimmer reasons in a `to=self` channel; vLLM's `--reasoning-parser muse-glimmer` routes it to `message.reasoning_content` and keeps `message.content` clean. Without that flag, reasoning leaks into your answer.
- **Reasoning tokens are billed as completion tokens.** They count against `max_tokens`. An `xhigh` run that hits the ceiling mid-thought returns an empty answer — the most common way this bites.

## Make it yours

- **Swap in your own problem.** Replace `PROBLEM` and `EXPECTED` with something from your workload. A setting that's right for arithmetic may be wrong for tool selection.
- **Run it more than once.** One sample per setting is a smoke test, not a measurement. Loop each effort 5–10× before you trust the ranking.
- **Raise `max_tokens` before you raise effort.** Truncated reasoning looks exactly like a wrong answer.
- **Then lock it in.** Pass the winning `reasoning_effort` in your production call and stop thinking about it.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Empty `answer`, high reasoning tokens | Hit `max_tokens` mid-thought | Raise `max_tokens`, or lower effort. |
| Reasoning text appears in `content` | Reasoning parser not enabled | Serve with `--reasoning-parser muse-glimmer`. |
| `reasoning_tokens` is `n/a` | Server didn't report the breakdown | Cosmetic — `completion_tokens` still tells you the cost. |
| Every setting is wrong | Problem too hard, or ceiling too low | Raise `max_tokens` first; then reconsider the task. |
| `Connection refused` | Nothing served on the port | Start vLLM ([`../../inference-server/vllm.md`](../../inference-server/vllm.md)). |

## Next steps

- See the `to=self` channel in the raw output: [`../../agentic-fundamentals/`](../../agentic-fundamentals/)
- Apply a tuned effort setting in a real pipeline: [`../alert-triage-copilot/`](../alert-triage-copilot/)
