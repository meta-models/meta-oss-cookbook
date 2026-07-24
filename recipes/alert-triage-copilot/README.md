# Alert-Triage Copilot

Point a three-stage agentic pipeline at a noisy on-call queue. It correlates the alerts, writes a runbook, then grades its own work and tells you when not to trust it.

This is the recipe that stresses the whole stack at once: multi-stage chaining, strict JSON at every boundary, and a self-assessment step that has to be honest about its own uncertainty. It's also the one that runs on data you'd never send to a hosted API — production alerts, hostnames, and internal topology.

## Recipe banner

| | |
|---|---|
| Max VRAM observed | ~60 GB (bf16, batch 1) — the serve itself; this recipe adds no footprint |
| Precision | bf16 |
| Model server | vLLM |
| Offline? | Yes, fully |

## Quickstart

```bash
pip install openai pydantic
# with Muse Glimmer already served (see ../../inference-server/vllm.md):
python alert_triage.py

# the self-assess stage is the one that benefits most from more thinking:
python alert_triage.py --reasoning-effort high
```

## What just happened

Eight alerts fired inside a 20-minute window. Most of them are the same incident.

**Stage 1 — triage.** Correlate, don't enumerate. The pipeline gets the deploy context (`checkout-api v4.22.1` shipped three minutes before the first alert; `payments-worker` shares a postgres cluster with it) and has to work out which alerts are causes and which are symptoms:

```text
root cause : checkout-api v4.22.1 exhausting the shared postgres connection pool
  A-1044  critical availability  PAGE   pods crashlooping post-deploy
  A-1047  critical saturation    PAGE   connection pool at 0/40 free
  A-1041  high     latency       hold   (symptom of A-1044)
  A-1042  high     correctness   hold   (symptom of A-1044)
  A-1043  high     saturation    hold   (symptom of A-1047)
  A-1045  low      latency       hold   unrelated to the deploy
  A-1046  low      maintenance   hold   21 days of runway
  A-1048  noise    maintenance   hold   non-blocking, no customer impact

2/8 alerts would wake someone.
```

**Stage 2 — remediate.** The plan is ordered by reversibility, and every step declares its blast radius. It also has a `do_not_do` list — the tempting actions that make it worse, like scaling up the workers that are exhausting the pool.

**Stage 3 — self-assess.** The stage that earns the recipe its place. The pipeline critiques its own output, names its weakest claim, lists the evidence it's missing, and returns `safe_to_action` or `needs_human_review`.

That last stage is the hard one. It's where a model that can't represent its own uncertainty produces a confident wrong answer — and it's exactly where the muse-spark baseline this recipe was ported from failed to emit parseable output at all. Deferring to a human is a pass here, not a failure.

Structurally, the whole thing works because each stage is schema-constrained: stage 2 consumes stage 1's typed object directly, stage 3 consumes both. Nothing between stages is a string that needs parsing, so a bad stage fails loudly instead of quietly corrupting the next one.

## Make it yours

- **Feed it your real queue.** Replace `ALERTS` with a pull from your alerting backend. The shape is deliberately generic — `id`, `at`, `service`, `text`.
- **Give it your topology.** `CONTEXT` is doing most of the correlation work. "These two services share a database" is worth more than any prompt tuning.
- **Tune the schemas, not the prompts.** Reach for the Pydantic models before the system prompts — field names and descriptions steer the output harder.
- **Gate on stage 3.** Wire `verdict == "safe_to_action"` to auto-remediation and `needs_human_review` to a page. That's the whole product.
- **Add a stage.** A fourth stage that drafts the incident-channel update is a natural extension.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A stage exits with "no parseable ..." | Model refused, or ran out of tokens mid-object | Check the printed refusal; raise `max_tokens`. |
| Every alert gets escalated | No topology in `CONTEXT` | Correlation needs the dependency graph. Add it. |
| Nothing gets escalated | Prompt over-tuned toward suppression | Loosen stage 1's system prompt; check the `noise` bucket. |
| Stage 3 always says `high` confidence | Effort too low for real self-critique | Run with `--reasoning-effort high`. |
| Plan cites hosts that don't exist | Model filling gaps | Tighten `RemediationStep`; nullable fields let it say "judgement call". |
| `Connection refused` | Nothing served on the port | Start vLLM ([`../../inference-server/vllm.md`](../../inference-server/vllm.md)). |

## Next steps

- The schema mechanics under each stage: [`../structured-output/`](../structured-output/)
- Pick the right effort per stage: [`../reasoning-effort/`](../reasoning-effort/)
