"""
reasoning_effort.py — dial Muse Glimmer's thinking up and down, and see what it costs.

Muse Glimmer reasons in a private `to=self` channel before it answers. `reasoning_effort`
controls how much. This runs the same problem at every setting and prints the
reasoning tokens spent, the latency, and whether the answer was right — so you can
pick a setting with numbers instead of vibes.

    python reasoning_effort.py

Requires a local Muse Glimmer served with vLLM (see ../../inference-server/vllm.md).
"""
from __future__ import annotations

import argparse
import time

from openai import OpenAI

EFFORTS = ["low", "medium", "high", "xhigh"]

# A problem that is easy to state, easy to check, and genuinely needs a few steps.
PROBLEM = (
    "Three services each retry failed calls. A retries up to 3 times, B up to 4, and "
    "C up to 2. A calls B, and B calls C. Each service's retry count applies to every "
    "call it makes. If the very first call from A fails at C every single time, how "
    "many times in total does C get called? Answer with the number only."
)
# A(1 + 3 retries) = 4 attempts, each doing B(1 + 4) = 5, each doing C(1 + 2) = 3.
EXPECTED = "60"


def ask(client: OpenAI, model: str, effort: str) -> dict:
    started = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROBLEM}],
        reasoning_effort=effort,
        temperature=0.0,
    )
    elapsed = time.perf_counter() - started

    message = resp.choices[0].message
    answer = (message.content or "").strip()
    # Muse Glimmer's `to=self` channel is routed here by vLLM's reasoning parser.
    reasoning = getattr(message, "reasoning_content", None) or ""

    usage = resp.usage
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None)

    return {
        "effort": effort,
        "answer": answer,
        "correct": EXPECTED in answer,
        "reasoning_tokens": reasoning_tokens,
        "completion_tokens": usage.completion_tokens,
        "seconds": elapsed,
        "reasoning": reasoning,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--api-key", default="not-needed", help="Local vLLM ignores this.")
    ap.add_argument("--show-reasoning", action="store_true",
                    help="Print the to=self channel for each run.")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"Problem: {PROBLEM}\nExpected: {EXPECTED}\n")
    rows = []
    for effort in EFFORTS:
        row = ask(client, args.model, effort)
        rows.append(row)
        mark = "ok " if row["correct"] else "MISS"
        rt = row["reasoning_tokens"]
        rt_s = str(rt) if rt is not None else "n/a"
        print(f"[{mark}] {effort:<8} reasoning={rt_s:>6}  "
              f"completion={row['completion_tokens']:>6}  "
              f"{row['seconds']:>6.1f}s  answer={row['answer'][:40]!r}")
        if args.show_reasoning and row["reasoning"]:
            print(f"         to=self: {row['reasoning'][:400]}\n")

    print("\n--- pick a setting ---")
    correct = [r for r in rows if r["correct"]]
    if not correct:
        print("Nothing got it right. Raise max_tokens, or the problem is too hard.")
        return
    cheapest = min(correct, key=lambda r: r["completion_tokens"])
    fastest = min(correct, key=lambda r: r["seconds"])
    print(f"cheapest correct : {cheapest['effort']} "
          f"({cheapest['completion_tokens']} completion tokens)")
    print(f"fastest correct  : {fastest['effort']} ({fastest['seconds']:.1f}s)")
    print("Ship the cheapest setting that is still reliably correct on YOUR task.")


if __name__ == "__main__":
    main()
