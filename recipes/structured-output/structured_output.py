"""
structured_output.py — get schema-valid JSON out of Muse Glimmer on the first try.

Free-text in, typed object out. The schema is enforced by the server, so you get a
parsed Pydantic model rather than a string you have to hope is valid JSON.

    python structured_output.py

Requires a local Muse Glimmer served with vLLM (see ../../inference-server/vllm.md).
"""
from __future__ import annotations

import argparse
import json
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field


# --------------------------------------------------------------------------------------
# 1. Describe the shape you want. This IS the contract — the server enforces it.
# --------------------------------------------------------------------------------------
class Party(BaseModel):
    name: str
    role: Literal["vendor", "customer", "partner", "other"]


class Contract(BaseModel):
    title: str = Field(description="Short human-readable name for the agreement")
    parties: list[Party]
    effective_date: str = Field(description="ISO 8601 date, YYYY-MM-DD")
    renewal_date: str | None = Field(description="ISO 8601 date, or null if none stated")
    auto_renews: bool
    annual_value_usd: float | None
    termination_notice_days: int | None
    risks: list[str] = Field(description="Clauses a reviewer should look at, if any")


# --------------------------------------------------------------------------------------
# 2. Unstructured input. Swap this for your own documents.
# --------------------------------------------------------------------------------------
DOCUMENT = """
MASTER SERVICES AGREEMENT

This Agreement is entered into on March 14, 2024 between Northwind Logistics Inc.
("Customer") and Vela Systems GmbH ("Vendor").

Term. The initial term runs twenty-four (24) months from the effective date and
renews automatically for successive twelve (12) month periods unless either party
gives written notice of non-renewal at least ninety (90) days prior to the end of
the then-current term.

Fees. Customer shall pay an annual subscription fee of USD 148,500, invoiced yearly
in advance. Fees may be increased by up to 7% at each renewal.

Liability. Vendor's aggregate liability is capped at fees paid in the preceding
three (3) months, and Vendor disclaims all liability for data loss.
"""

PROMPT = (
    "Extract the structured contract record from the document below. Use null for any "
    "field the document does not state. For `risks`, list clauses a reviewer should "
    "look at more closely.\n\n"
    f"{DOCUMENT}"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="muse-glimmer")
    ap.add_argument("--api-key", default="not-needed", help="Local vLLM ignores this.")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 3. Ask for the schema by name. `.parse()` returns a typed object, not a string.
    completion = client.chat.completions.parse(
        model=args.model,
        messages=[{"role": "user", "content": PROMPT}],
        response_format=Contract,
        temperature=0.0,
    )

    contract = completion.choices[0].message.parsed
    if contract is None:
        raise SystemExit("Model refused or returned no parseable content.")

    # 4. It's a real object — attribute access, not dict spelunking.
    print(json.dumps(contract.model_dump(), indent=2))
    print("\n--- typed access ---")
    print(f"title           : {contract.title}")
    print(f"auto-renews     : {contract.auto_renews}")
    print(f"notice required : {contract.termination_notice_days} days")
    for party in contract.parties:
        print(f"party           : {party.name} ({party.role})")
    for risk in contract.risks:
        print(f"risk            : {risk}")


if __name__ == "__main__":
    main()
