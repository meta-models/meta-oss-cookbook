---
vendor: Intel
contact: Stefanka Kitanovska(stefanka.kitanovska@intel.com)
updated: 2026-08-09
---

# Intel

Intel has a broad AI portfolio, from consumer laptops powered by Intel® Core™ Ultra Series 3 integrated GPUs, to workstations with Intel® Arc™ Pro B70 discrete GPU configurations and Intel® Xeon® processors, ensuring developers and enterprises can deploy Muse Glimmer wherever their workflows demand.

## Platforms

<!-- The index on platform/README.md is generated from this table. One row per platform. -->

| Platform | Class | Accelerator | Memory | Precisions verified |
|---|---|---|---|---|
| Arc Pro GPU (server) | server | Arc Pro B70 x 2 | 32 GB LPDDR5 per accelerator | bf16 |
| Arc Pro GPU (consumer) | consumer | Arc Pro B Series | 32 GB GDDR6 | bf16, int4 |
| Xeon 6 6980P | server | N/A |  | bf16 |
| Intel Core Ultra Series 2 and 3 | consumer | iGPU | 32GB+ | q4_k_m, int4 |

<!-- Class is one of: `consumer`, `workstation`, `server`, `edge`, `cloud-instance`.

Precision labels: use the artifact's published name, lowercase — `bf16`, `int4`, `nvfp4/mxfp8`, `q4_k_m`. Unfamiliar labels get a warning, not a rejection. -->

## Performance

| Platform | Precision | TTFT p50 | Output tok/s |
|---|---|---|---|
| Arc Pro GPU (server) | bf16 | Not measured yet. | Not measured yet. |
| Arc Pro GPU (consumer) | bf16 | Not measured yet. | Not measured yet. |
| Xeon 6 6980P | bf16 | Not measured yet. | Not measured yet. |
| Intel Core Ultra Series 2 and 3 | bf16 | Not measured yet. | Not measured yet. |

## Platform details

<!-- One ### block per platform in the table above. Keep the five bold headings. -->

### Arc Pro GPU (server)

**Snapshot**

| | |
|---|---|
| Accelerator | Arc Pro B70 x 2 |
| Memory | 32 GB GDDR6 per card |
| Memory bandwidth | 608 GB/s |
| Host OS tested | Ubuntu 24.04 |
| Driver / runtime | Level Zero 1.28.2 |
| Model server | vLLM main branch |
| Verified by | Liangliang Ma on 2026-08-09 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| Level Zero | 1.28.2 | `xpu-smi` |
| Disk | 80 GB free for bf16 weights | `df -h` |

**Deploy**

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
docker build -f docker/Dockerfile.xpu -t vllm-xpu-env --shm-size=4g .
docker run -it --rm --network=host --ipc=host --privileged \
                 --device /dev/dri:/dev/dri \
                 -v /dev/dri/by-path:/dev/dri/by-path \
                 --entrypoint bash \
                 vllm-xpu-env
```

```bash
hf download meta-models/Muse-Glimmer-30B --local-dir ./muse-glimmer-bf16
```

```bash
ZE_AFFINITY_MASK=0,1  \
python -m vllm.entrypoints.openai.api_server \
  --model ./muse-glimmer-bf16 \
  --reasoning-parser muse_glimmer \
  --enforce-eager \
  --tensor-parallel-size 2
```

Tensor parallelism across >= 2 accelerators (e.g. `--tensor-parallel-size 4`) also works.

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| bf16 | Verified | `meta-models/Muse-Glimmer-30B` | | |
| fp8 | Planned | TBD | | |

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

Nothing reported yet.

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

### Arc Pro GPU (consumer)

**Snapshot**

| | |
|---|---|
| Accelerator | B70 × 1 |
| Memory | 32 GB  |
| Memory bandwidth | 608 GB/s |
| Host OS tested | Windows 11 Pro |
| Driver / runtime | GPU Driver 32.0.101.8805|
| Model server | llama.cpp, build b10333 |
| Verified by | Gautam Singh on 2026-08-09 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| GPU Driver | 32.0.101.8805| `Intel® Graphics Driver` |
| Disk | 24 GB free for the q4_k_m GGUF | `df -h` |

**Deploy**

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF  

cmake --build build -j 4   --target llama-server llama-cli llama-mtmd-cli llama-bench

```
```bash
llama-server.exe -m muse-glimmer-rl_v1-quant.gguf -mm mmproj-muse-glimmer-rl_v1-bf16.gguf --alias muse-glimmer --no-mmap -ngl 99 --port 8080 --jinja --chat-template-file muse-glimmer.jinja  --reasoning-format deepseek --reasoning auto  --override-kv muse-glimmer.context_length=int:90000
```

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m| Verified | `meta-models/Muse-Glimmer-30B` | | |

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

Nothing reported yet.

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

### Intel Core Ultra Series 2 and 3

**Snapshot**

| | |
|---|---|
| Accelerator | iGPU |
| Memory | 32+ GB  |
| Host OS tested | Windows 11 Pro |
| Driver / runtime | GPU Driver 32.0.101.8805|
| Model server | llama.cpp, build b10333 |
| Verified by | Gautam Singh on 2026-08-09 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| GPU Driver | 32.0.101.8805| `Intel® Graphics Driver` |
| Disk | 32 GB free for the q4_k_m GGUF | `df -h` |

**Deploy**

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp

cmake -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=ON -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF  

cmake --build build -j 4   --target llama-server llama-cli llama-mtmd-cli llama-bench

```
```bash
llama-server.exe -m muse-glimmer-rl_v1-quant.gguf -mm mmproj-muse-glimmer-rl_v1-bf16.gguf --alias muse-glimmer --no-mmap -ngl 99 --port 8080 --jinja --chat-template-file muse-glimmer.jinja  --reasoning-format deepseek --reasoning auto  --override-kv muse-glimmer.context_length=int:90000
```

Confirm the deploy with the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md) — Muse Glimmer is an agentic model, so "it generates text" isn't a working deploy.

> [!WARNING]
> Stop tokens: `eos_token_id = [<\|end_of_text\|>, <\|eot\|>]`. Never stop on `<\|eom\|>` — it marks end-of-*message*, the turn continues, and stopping there reduces parallel tool calling to near zero. Confirm your runtime doesn't override this. Details: [`../inference-server/README.md`](../inference-server/README.md).

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m | Verified | `meta-models/Muse-Glimmer-30B` | | |


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

Nothing reported yet.

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

### Xeon 6 6980P

**Snapshot**

| | |
|---|---|
| Accelerator | N/A |
| Memory | depends on DDR5 mounted, suggest to > 512 GB |
| Memory bandwidth | 1228.8 GB/s |
| Host OS tested | Ubuntu 24.04 |
| Model server | vLLM main branch |
| Verified by | Jiang Li on 2026-08-09 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| Disk | 80 GB free for bf16 weights | `df -h` |

**Deploy**

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env \
        --target vllm-openai . 
docker run -it --rm \
            --privileged \
            --shm-size=4g \
            --entrypoint bash \
            --name vllm-cpu-env \
            vllm-cpu-env 
```

```bash
hf download meta-models/Muse-Glimmer-30B --local-dir ./muse-glimmer-bf16
```

```bash
vllm serve ./muse-glimmer-bf16 \
    -tp <TP_SIZE> \
    --dtype=bfloat16 \
    --trust-remote-code --enable-auto-tool-choice \
    --tool-call-parser muse_glimmer \
    --reasoning-parser muse_glimmer \
    --generation-config auto
```

Confirm the deploy with the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md) — Muse Glimmer is an agentic model, so "it generates text" isn't a working deploy.

> [!WARNING]
> Stop tokens: `eos_token_id = [<\|end_of_text\|>, <\|eot\|>]`. Never stop on `<\|eom\|>` — it marks end-of-*message*, the turn continues, and stopping there reduces parallel tool calling to near zero. Confirm your runtime doesn't override this. Details: [`../inference-server/README.md`](../inference-server/README.md).

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| bf16 | Verified | `meta-models/Muse-Glimmer-30B` | | |
| fp8 | Planned | TBD | | |

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

Nothing reported yet.

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

## Support

Where a reader takes a problem with these platforms — your issue tracker, forum, or developer docs. Not a sales contact.

- Issues: https://github.com/vllm-project/vllm/issues
- Docs: https://docs.vllm.ai/en/stable/getting_started/installation/
- Maintainer: Matrix Yao(matrix.yao@intel.com)
