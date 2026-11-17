---
vendor: Oracle Cloud Infrastructure
contact: Federico Kamelhar (<contact-email>)
updated: 2026-11-17
---

# Oracle Cloud Infrastructure

Oracle Cloud Infrastructure (OCI) offers Arm-based Ampere A1 compute instances alongside NVIDIA GPU bare-metal and VM shapes. This page covers running Muse Glimmer on OCI instances with upstream llama.cpp built from source, with no OCI-specific runtime, driver or plugin required.

## Platforms

<!-- The index on platform/README.md is generated from this table. One row per platform. -->

| Platform | Class | Accelerator | Memory | Precisions verified |
|---|---|---|---|---|
| VM.Standard.A1.Flex (Ampere Altra, 16 OCPU) | cloud-instance | N/A | 96 GB DDR4 | q4_k_m |

## Performance

| Platform | Precision | TTFT p50 | Output tok/s |
|---|---|---|---|
| VM.Standard.A1.Flex (Ampere Altra, 16 OCPU) | q4_k_m | Not measured yet. | Not measured yet. |

## Platform details

<!-- One ### block per platform in the table above. Keep the five bold headings. -->

### VM.Standard.A1.Flex (Ampere Altra, 16 OCPU)

**Snapshot**

| | |
|---|---|
| Accelerator | N/A (Ampere Altra Arm Neoverse-N1 CPU, 16 OCPU = 16 physical cores) |
| Memory | 96 GB DDR4 |
| Memory bandwidth | Not published per flex shape |
| Host OS tested | Canonical Ubuntu 24.04 (aarch64) |
| Driver / runtime | llama.cpp, built from source at commit TBD |
| Model server | llama-server |
| Verified by | Pending |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| OCI shape | VM.Standard.A1.Flex, 16 OCPU / 96 GB (a 4 OCPU / 24 GB Always Free instance also loads the q4_k_m GGUF) | OCI Console or `oci compute instance get` |
| Disk | 25 GB free for the q4_k_m GGUF, projector and drafter | `df -h` |
| Build tools | `build-essential cmake git curl libcurl4-openssl-dev` | `cmake --version` |

**Deploy**

```bash
sudo apt-get update && sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=ON
cmake --build build -j "$(nproc)" --config Release
```

```bash
./build/bin/llama-server \
  -hf meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M \
  -a muse-glimmer \
  -c 131072 -np 1 \
  -t "$(nproc)" \
  --host 0.0.0.0 --port 8080
```

Confirm the deploy with the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md). Muse Glimmer is an agentic model, so "it generates text" isn't a working deploy.

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m | Planned | `meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M` | | Measurement in progress. |
| bf16 | Not supported | `meta-models/Muse-Glimmer-30B` | | 60 GB of weights fit in RAM but CPU-only bf16 decode is impractical on this shape; use a GPU shape. |

<!--
  Status — exactly one per row:
    Verified            you ran it here, at the versions above, tool calling round-tripped
    Works, not verified it loads and generates; tool calling unchecked
    Planned             roadmap; no numbers, no steps
    Not supported       doesn't work on this platform; a one-line reason is useful
-->

**Troubleshooting**

Nothing reported yet.

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

## Support

Where a reader takes a problem with these platforms — your issue tracker, forum, or developer docs. Not a sales contact.

- Issues: https://github.com/ggml-org/llama.cpp/issues (runtime) and https://github.com/oracle-quickstart/oci-ai-blueprints/issues (OCI deployment)
- Docs: https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm
- Maintainer: Federico Kamelhar (<contact-email>)
