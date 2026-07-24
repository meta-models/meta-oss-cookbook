"""
alert_triage.py — a three-stage, strict-JSON pipeline that triages a noisy alert queue.

Stage 1  triage      classify each alert, score it, decide if a human is needed
Stage 2  remediate   propose concrete next steps for whatever survived triage
Stage 3  self-assess grade its own output and flag what it is unsure about

Every stage is schema-constrained, so each one's output is a typed object the next
stage consumes directly. No string parsing between stages, and a stage that fails
its schema fails loudly instead of poisoning the one after it.

    python alert_triage.py

Requires a local Muse Glimmer served with vLLM (see ../../inference-server/vllm.md).
"""
from __future__ import annotations

import argparse
import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# The input: a realistically noisy 20-minute window from an on-call queue.
# --------------------------------------------------------------------------------------
ALERTS = [
    {"id": "A-1041", "at": "02:14:07", "service": "checkout-api",
     "text": "p99 latency 2340ms (threshold 800ms), sustained 6m"},
    {"id": "A-1042", "at": "02:14:09", "service": "checkout-api",
     "text": "error rate 4.1% (threshold 1%)"},
    {"id": "A-1043", "at": "02:14:11", "service": "payments-worker",
     "text": "queue depth 18400 and climbing, consumer lag 9m"},
    {"id": "A-1044", "at": "02:15:02", "service": "checkout-api",
     "text": "HealthCheck failing on 3/12 pods, CrashLoopBackOff"},
    {"id": "A-1045", "at": "02:15:40", "service": "cdn-edge",
     "text": "cache hit ratio 71% (baseline 94%)"},
    {"id": "A-1046", "at": "02:16:55", "service": "internal-wiki",
     "text": "TLS certificate expires in 21 days"},
    {"id": "A-1047", "at": "02:17:20", "service": "payments-worker",
     "text": "postgres connection pool exhausted, 0/40 free"},
    {"id": "A-1048", "at": "02:19:03", "service": "batch-reporting",
     "text": "nightly job finished 40m late (non-blocking)"},
]

CONTEXT = (
    "Deploy checkout-api v4.22.1 rolled out at 02:11 UTC. "
    "payments-worker shares the primary postgres cluster with checkout-api. "
    "cdn-edge and internal-wiki have no dependency on either."
)


# --------------------------------------------------------------------------------------
# Stage schemas. These are the contract between stages.
# --------------------------------------------------------------------------------------
class TriagedAlert(BaseModel):
    id: str
    severity: Literal["critical", "high", "medium", "low", "noise"]
    category: Literal["availability", "latency", "saturation", "correctness",
                      "maintenance", "unknown"]
    is_symptom_of: str | None = Field(
        description="ID of the alert this one is likely a downstream symptom of, else null")
    wake_a_human: bool
    one_line_why: str


class TriageResult(BaseModel):
    likely_root_cause: str
    incident_summary: str = Field(description="Two sentences an on-call can read at 2am")
    alerts: list[TriagedAlert]


class RemediationStep(BaseModel):
    order: int
    action: str
    command_or_query: str | None = Field(
        description="A concrete command to run, or null if it's a judgement call")
    reverses_cleanly: bool
    blast_radius: Literal["none", "single-service", "shared-infra", "customer-facing"]


class RemediationPlan(BaseModel):
    immediate_mitigation: str
    steps: list[RemediationStep]
    do_not_do: list[str] = Field(description="Tempting actions that would make it worse")


class SelfAssessment(BaseModel):
    confidence: Literal["high", "medium", "low"]
    weakest_link: str = Field(description="The single least-supported claim made above")
    missing_evidence: list[str] = Field(description="What you'd need to be sure")
    would_page_incorrectly: list[str] = Field(
        description="Alert IDs that might have been escalated wrongly, in either direction")
    verdict: Literal["safe_to_action", "needs_human_review"]


def stage(client: OpenAI, model: str, schema: type[BaseModel],
          system: str, user: str, effort: str) -> BaseModel:
    """Run one schema-constrained stage. Raises if the model won't produce the shape."""
    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        response_format=schema,
        reasoning_effort=effort,
        temperature=0.0,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        refusal = completion.choices[0].message.refusal
        raise SystemExit(f"Stage produced no parseable {schema.__name__}: {refusal}")
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--api-key", default="not-needed", help="Local vLLM ignores this.")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["low", "medium", "high", "xhigh"])
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    alerts_json = json.dumps(ALERTS, indent=2)

    # ---- Stage 1: triage -------------------------------------------------------------
    print("=== stage 1: triage ===")
    triage: TriageResult = stage(
        client, args.model, TriageResult,
        system=("You are an SRE triaging an alert queue. Correlate alerts that share a "
                "root cause, and mark downstream symptoms rather than treating every "
                "alert as independent. Be decisive about what does not need a human."),
        user=f"Deploy context:\n{CONTEXT}\n\nAlerts:\n{alerts_json}",
        effort=args.reasoning_effort,
    )
    print(f"root cause : {triage.likely_root_cause}")
    print(f"summary    : {triage.incident_summary}")
    for a in triage.alerts:
        symptom = f" (symptom of {a.is_symptom_of})" if a.is_symptom_of else ""
        page = "PAGE" if a.wake_a_human else "hold"
        print(f"  {a.id}  {a.severity:<8} {a.category:<12} {page:<5}{symptom}  {a.one_line_why}")

    paged = [a for a in triage.alerts if a.wake_a_human]
    print(f"\n{len(paged)}/{len(triage.alerts)} alerts would wake someone.")
    if not paged:
        print("Nothing actionable. Stopping before remediation.")
        return

    # ---- Stage 2: remediate ----------------------------------------------------------
    print("\n=== stage 2: remediation plan ===")
    plan: RemediationPlan = stage(
        client, args.model, RemediationPlan,
        system=("You are the same SRE, now writing the runbook. Order steps so the "
                "safest, most reversible action comes first. Be explicit about blast "
                "radius. Do not propose anything you cannot justify from the evidence."),
        user=(f"Deploy context:\n{CONTEXT}\n\n"
              f"Your triage:\n{triage.model_dump_json(indent=2)}"),
        effort=args.reasoning_effort,
    )
    print(f"mitigate now: {plan.immediate_mitigation}")
    for s in sorted(plan.steps, key=lambda x: x.order):
        rev = "reversible" if s.reverses_cleanly else "NOT reversible"
        print(f"  {s.order}. [{s.blast_radius}, {rev}] {s.action}")
        if s.command_or_query:
            print(f"      $ {s.command_or_query}")
    for d in plan.do_not_do:
        print(f"  do NOT: {d}")

    # ---- Stage 3: self-assess --------------------------------------------------------
    # The stage that matters. A pipeline that can't say "I'm not sure" is a pipeline
    # that hands you a confident wrong answer at 2am.
    print("\n=== stage 3: self-assessment ===")
    review: SelfAssessment = stage(
        client, args.model, SelfAssessment,
        system=("Critique the triage and plan you just produced. Be adversarial about "
                "your own reasoning. Name the weakest claim, not the easiest one."),
        user=(f"Triage:\n{triage.model_dump_json(indent=2)}\n\n"
              f"Plan:\n{plan.model_dump_json(indent=2)}"),
        effort=args.reasoning_effort,
    )
    print(f"confidence   : {review.confidence}")
    print(f"weakest link : {review.weakest_link}")
    for m in review.missing_evidence:
        print(f"  need: {m}")
    for w in review.would_page_incorrectly:
        print(f"  uncertain escalation: {w}")
    print(f"\nVERDICT: {review.verdict}")

    if review.verdict == "needs_human_review":
        print("Pipeline is deferring to a human. That's a pass, not a failure.")


if __name__ == "__main__":
    main()
