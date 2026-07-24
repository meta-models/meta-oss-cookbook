# Structured Output

Turn unstructured text into a typed object you can pass straight into code — schema enforced by the server, parsed on the first try.

This is the plumbing under most useful agents. Before an agent can act on a document, an alert, or a form, something has to turn prose into fields. Doing that with a regex over a hopefully-JSON string is where local pipelines usually break.

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
python structured_output.py
```

## What just happened

The recipe declares a `Contract` Pydantic model, hands the server a real contract as free text, and gets back a populated object:

```text
{
  "title": "Master Services Agreement",
  "parties": [
    {"name": "Northwind Logistics Inc.", "role": "customer"},
    {"name": "Vela Systems GmbH", "role": "vendor"}
  ],
  "effective_date": "2024-03-14",
  "auto_renews": true,
  "annual_value_usd": 148500.0,
  "termination_notice_days": 90,
  "risks": ["Liability capped at 3 months of fees", "..."]
}
```

Three things are doing the work:

1. **The schema is the prompt.** `response_format=Contract` sends the JSON Schema to the server. Field names and `Field(description=...)` are read by the model, so naming a field well does more than any prompt tweak.
2. **The server constrains decoding.** Invalid JSON is not a failure mode you handle — it's a token the model can't emit.
3. **`.parse()` returns an object.** You get `contract.termination_notice_days` as an `int`, not `data["termination_notice_days"]` as a `str` you cast and pray over.

`null` is a first-class answer here. `renewal_date: str | None` lets the model say "not stated" instead of inventing a date — which is what it will do if the only legal output is a string.

## Make it yours

- **Change the schema.** Replace `Contract` with your own model. Start narrow: fewer fields, extracted reliably, beats a wide schema half-filled.
- **Point it at your documents.** Swap `DOCUMENT` for a file read. This runs offline, so the sensitive-document case works without changes.
- **Make every uncertain field optional.** `X | None` is the difference between "not stated" and a confident hallucination.
- **Keep `temperature=0.0`.** Extraction is not a creative task.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `parsed` is `None` | Model refused, or hit the token limit mid-object | Check `completion.choices[0].message.refusal`; raise `max_tokens`. |
| Fields silently wrong | Ambiguous field names | Add `Field(description=...)`. The model reads it. |
| Invented dates / values | Field is required and non-nullable | Make it `str \| None` so "not stated" is representable. |
| Model returns prose, not JSON | `response_format` dropped | Confirm the server accepted the schema; older builds ignore it silently. |
| `Connection refused` | Nothing served on the port | Start vLLM ([`../../inference-server/vllm.md`](../../inference-server/vllm.md)). |

## Next steps

- Use structured output as pipeline glue: [`../alert-triage-copilot/`](../alert-triage-copilot/)
- Trade latency for accuracy on hard extractions: [`../reasoning-effort/`](../reasoning-effort/)
