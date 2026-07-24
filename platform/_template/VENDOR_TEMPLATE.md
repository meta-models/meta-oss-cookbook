---
vendor: <Your Org>
contact: <Name (name@example.com)>
updated: <YYYY-MM-DD>
---

# <Your Org>

<!--
  One file per organization. Rename this to <your-org>.md — lowercase, hyphenated —
  and put it directly in platform/. Everything you contribute stays in this one file.

  The four ## sections and their order are fixed, so that pages from different
  partners can be compared line by line. Add one ### block under "Platform details"
  for each platform you support, keeping the five bold headings inside it.

  Delete every angle-bracket placeholder and every HTML comment before sending.
  Run `python platform/validate.py <your-org>.md` to check.

  A merged page that shows the expected level of detail: intel.md, in platform/.
-->

One or two sentences: what your organization builds, and where Muse Glimmer fits. Mention anything true of every platform below — a shared runtime, driver, or quantization toolchain — so it doesn't need repeating.

## Platforms

<!-- The index on platform/README.md is generated from this table. One row per platform. -->

| Platform | Class | Accelerator | Memory | Precisions verified |
|---|---|---|---|---|
| <Platform A> | <workstation> | <chip name> | <128 GB unified> | <bf16, int4> |
| <Platform B> | <consumer> | <chip name> | <32 GB unified> | <q4_k_m> |

Class is one of: `consumer`, `workstation`, `server`, `edge`, `cloud-instance`.

Precision labels: use the artifact's published name, lowercase — `bf16`, `int4`, `nvfp4/mxfp8`, `q4_k_m`. Unfamiliar labels get a warning, not a rejection.

## Performance

**Optional.** Where nothing has been measured, `Not measured yet.` is a valid answer and is preferred over an estimate.

One table for all platforms. The first two columns are fixed; the rest are yours to choose — TTFT and output tok/s are the usual pair. Headline configurations only.

| Platform | Precision | TTFT p50 | Output tok/s |
|---|---|---|---|
| <Platform A> | <bf16> | <ms> | <tok/s> |
| <Platform A> | <int4> | <ms> | <tok/s> |

Measured on <YYYY-MM-DD> with <server + version> and <driver / runtime version>, at <concurrency> concurrent request(s), <input> in / <output> out tokens. Reproduce with:

```bash
# The command behind the numbers above.
```

## Platform details

<!-- One ### block per platform in the table above. Keep the five bold headings. -->

### <Platform A>

**Snapshot**

| | |
|---|---|
| Accelerator | <chip name> × <count> |
| Memory | <128 GB unified> |
| Memory bandwidth | <273 GB/s> |
| Host OS tested | <Ubuntu 24.04> |
| Driver / runtime | <driver 580.x, CUDA 13.0> |
| Model server | <vLLM 0.11.0> |
| Verified by | <Name> on <YYYY-MM-DD> |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| <Driver> | <580.65> | `<nvidia-smi>` |
| <Runtime / toolkit> | <CUDA 13.0> | `<nvcc --version>` |
| <Disk> | <80 GB free for bf16 weights> | |

**Deploy**

```bash
# Install: runtime, server, and any platform-specific packages.
```

```bash
# Get the weights: which artifact for this platform, and where it lands.
```

```bash
# Serve: the full command, with every flag this platform needs. Call out any flag
# that differs from the reference vLLM command in ../inference-server/vllm.md.
```

Say how a reader confirms the deploy works — Muse Glimmer is an agentic model, so "it generates text" isn't a working deploy. Point at the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md) and note anything platform-specific about the response.

> [!WARNING]
> Stop tokens: `eos_token_id = [<\|end_of_text\|>, <\|eot\|>]`. Never stop on `<\|eom\|>` — it marks end-of-*message*, the turn continues, and stopping there reduces parallel tool calling to near zero. Confirm your runtime doesn't override this. Details: [`../inference-server/README.md`](../inference-server/README.md).

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| bf16 | <Verified> | <link to the checkpoint or build> | <~60 GB peak, batch 1, 8K context> | |
| int4 | <Verified> | <link> | <~17 GB peak, batch 1, 8K context> | <quantizer and calibration set used> |
| q4_k_m | <Planned> | | | |

<!--
  Status — exactly one per row:
    Verified            you ran it here, at the versions above, tool calling round-tripped
    Works, not verified it loads and generates; tool calling unchecked
    Planned             roadmap; no numbers, no steps
    Not supported       doesn't work on this platform; a one-line reason is useful

  Use Notes for caveats that would otherwise need a section of their own:
  context ceilings, unsupported modalities, features your runtime hasn't wired up.
-->

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| <…> | <…> | <…> |

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

### <Platform B>

<!-- Repeat the five bold headings for each further platform. -->

**Snapshot**

**Prerequisites**

**Deploy**

**Supported precisions**

**Troubleshooting**

## Support

Where a reader takes a problem with these platforms — your issue tracker, forum, or developer docs. Not a sales contact.

- Issues: <url>
- Docs: <url>
- Maintainer: <Name (name@example.com)>
