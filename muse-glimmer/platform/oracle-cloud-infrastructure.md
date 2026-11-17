---
vendor: Oracle Cloud Infrastructure
contact: Federico Kamelhar (github.com/fede-kamel)
updated: 2026-11-17
---

# Oracle Cloud Infrastructure

Oracle Cloud Infrastructure (OCI) offers Arm-based Ampere A1 flexible compute instances alongside NVIDIA GPU VM and bare-metal shapes. This page covers running Muse Glimmer on OCI Ampere A1 instances with upstream llama.cpp on the CPU backend, and on NVIDIA A10 GPU shapes with the upstream llama.cpp CUDA container. No OCI-specific runtime, driver or plugin is involved; the OCI GPU images ship the NVIDIA driver, and the 4 OCPU / 24 GB A1 configuration fits inside OCI's Always Free tier.

## Platforms

<!-- The index on platform/README.md is generated from this table. One row per platform. -->

| Platform | Class | Accelerator | Memory | Precisions verified |
|---|---|---|---|---|
| VM.Standard.A1.Flex, 16 OCPU | cloud-instance | N/A (Ampere Altra, Arm Neoverse-N1 CPU) | 96 GB | q4_k_m |
| VM.Standard.A1.Flex, 4 OCPU (Always Free) | cloud-instance | N/A (Ampere Altra, Arm Neoverse-N1 CPU) | 24 GB | q4_k_m |
| VM.GPU.A10.2 (NVIDIA A10) | cloud-instance | NVIDIA A10 × 2 | 24 GB GDDR6 per GPU | q4_k_m |

## Performance

| Platform | Precision | TTFT p50 | Output tok/s | pp512 tok/s | tg128 tok/s |
|---|---|---|---|---|---|
| VM.Standard.A1.Flex, 16 OCPU | q4_k_m | 4583 ms | 5.8 | 18.5 | 5.7 |
| VM.Standard.A1.Flex, 4 OCPU (Always Free) | q4_k_m | 17159 ms | 2.0 | 4.7 | 2.0 |
| VM.GPU.A10.2 (NVIDIA A10), 1 GPU | q4_k_m | 315 ms | 27.3 | Not measured yet. | Not measured yet. |

A1 rows measured on 2026-11-17 with llama.cpp `1.0.0-dev` (commit `ad9d9e8`, GCC 13.3.0, CPU backend) on Ubuntu 24.04.5 LTS, kernel `7.0.0-1013-oracle`; the A10 row with the upstream `ghcr.io/ggml-org/llama.cpp:server-cuda` image (build 12464, commit `d3c3e242c`) on Oracle Linux 9.8, driver 615.71.09, one of the two A10s, `-ngl 99 -c 131072`. All at 1 concurrent request. TTFT and Output tok/s come from five streamed `/v1/chat/completions` calls with ~120 input / 128 output tokens, `temperature 0`, `reasoning_strength: low`, and a **distinct prompt per run** so the server's prefix cache is never hit (identical prompts report ~1 s TTFT, which is the cache, not the model). pp512 / tg128 are `llama-bench` with 3 repetitions. Model load before the first request took ~35 s on 16 OCPU and ~4 min 20 s on 4 OCPU. The 4 OCPU TTFT figure is the median of runs 2-5 plus a first run of 28 s. Reproduce with:

```bash
./build/bin/llama-bench -m ~/models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf -p 512 -n 128 -t 16 -r 3
```

## Platform details

<!-- One ### block per platform in the table above. Keep the five bold headings. -->

### VM.Standard.A1.Flex, 16 OCPU

**Snapshot**

| | |
|---|---|
| Accelerator | N/A (Ampere Altra, Arm Neoverse-N1; 16 OCPU = 16 physical cores) |
| Memory | 96 GB |
| Memory bandwidth | Not published per flex shape |
| Host OS tested | Ubuntu 24.04.5 LTS (aarch64), kernel 7.0.0-1013-oracle |
| Driver / runtime | llama.cpp 1.0.0-dev, commit ad9d9e8, GCC 13.3.0, CPU backend |
| Model server | llama-server |
| Verified by | Federico Kamelhar on 2026-11-17 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| OCI shape | VM.Standard.A1.Flex with 16 OCPU / 96 GB | OCI Console, or `oci compute instance get` |
| Disk | 25 GB free for the q4_k_m GGUF, vision projector and DFlash drafter | `df -h` |
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
  -t 16 --jinja \
  --host 0.0.0.0 --port 8080
```

`-hf` downloads the checkpoint and its vision projector on first start. The run above used the same files pre-downloaded from `meta-models/Muse-Glimmer-30B-GGUF` and passed as `-m Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj mmproj-Muse-Glimmer-30B-Q4_K_M.gguf`.

Confirm the deploy with the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md). Muse Glimmer is an agentic model, so "it generates text" isn't a working deploy. On this platform the check returned `get_weather(city="Paris", units="celsius")` with `finish_reason: tool_calls`.

**Supported precisions**

One row per precision that's relevant, including the ones that don't work.

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m | Verified | `meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M` (15.6 GiB) | 33.8 GB RSS with `-c 131072 -np 1` | Tool calling round-tripped. |
| bf16 | Not supported | `meta-models/Muse-Glimmer-30B` | | 60 GB of weights fit in RAM, but CPU-only bf16 decode is impractical on this shape. Use a GPU shape. |

<!--
  Status — exactly one per row:
    Verified            you ran it here, at the versions above, tool calling round-tripped
    Works, not verified it loads and generates; tool calling unchecked
    Planned             roadmap; no numbers, no steps
    Not supported       doesn't work on this platform; a one-line reason is useful
-->

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| TTFT looks like ~1 s on repeated requests | Identical prompts hit llama-server's prefix cache | Vary the prompt per request when measuring; cold TTFT for ~120 tokens is ~4.5 s on 16 OCPU. |
| `pip install huggingface_hub` fails with `Cannot uninstall typing_extensions` on Ubuntu 24.04 | The system package is Debian-managed | Install into a venv (`python3 -m venv ~/hfenv`), or let `llama-server -hf` download the files. |

<!-- Escape pipes inside cells: write `<\|eot\|>`, not `<|eot|>`, or the table breaks. -->

### VM.Standard.A1.Flex, 4 OCPU (Always Free)

**Snapshot**

| | |
|---|---|
| Accelerator | N/A (Ampere Altra, Arm Neoverse-N1; 4 OCPU = 4 physical cores) |
| Memory | 24 GB |
| Memory bandwidth | Not published per flex shape |
| Host OS tested | Ubuntu 24.04.5 LTS (aarch64), kernel 7.0.0-1013-oracle |
| Driver / runtime | llama.cpp 1.0.0-dev, commit ad9d9e8, GCC 13.3.0, CPU backend |
| Model server | llama-server |
| Verified by | Federico Kamelhar on 2026-11-17 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| OCI shape | VM.Standard.A1.Flex with 4 OCPU / 24 GB (the Always Free allowance) | OCI Console, or `oci compute instance get` |
| Disk | 25 GB free for the q4_k_m GGUF, vision projector and DFlash drafter | `df -h` |
| Build tools | `build-essential cmake git curl libcurl4-openssl-dev` | `cmake --version` |

**Deploy**

Same build as above. Keep the context small enough for 24 GB of RAM:

```bash
./build/bin/llama-server \
  -hf meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M \
  -a muse-glimmer \
  -c 32768 -np 1 \
  -t 4 --jinja \
  --host 0.0.0.0 --port 8080
```

**Supported precisions**

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m | Verified | `meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M` (15.6 GiB) | 20.7 GB RSS with `-c 32768 -np 1` | Tool calling round-tripped. Model load takes ~4 min on 4 cores. |

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| `/v1/chat/completions` returns HTTP 503 `Loading model` for several minutes after start | Reading the 15.6 GiB checkpoint on 4 cores with cold page cache takes longer than on the 16 OCPU shape | Poll `/health` until it returns `{"status":"ok"}` before sending traffic; don't treat the first 503s as a failure. |

### VM.GPU.A10.2 (NVIDIA A10)

**Snapshot**

| | |
|---|---|
| Accelerator | NVIDIA A10 × 2 (the llama.cpp numbers use one GPU) |
| Memory | 24 GB GDDR6 per GPU, 240 GB host RAM |
| Memory bandwidth | 600 GB/s per GPU |
| Host OS tested | Oracle Linux 9.8 Gen2-GPU image (kernel 5.15.0-325 UEK) |
| Driver / runtime | NVIDIA driver 615.71.09 (preinstalled); Docker 29.10 + nvidia-container-toolkit; llama.cpp `server-cuda` image build 12464 (commit d3c3e242c) |
| Model server | llama-server (container) |
| Verified by | Federico Kamelhar on 2026-11-17 |

**Prerequisites**

| Requirement | Minimum | Check with |
|---|---|---|
| OCI shape | VM.GPU.A10.1 or VM.GPU.A10.2 with the Oracle Linux Gen2-GPU image | `nvidia-smi` |
| Root filesystem | Grow it first: the Gen2-GPU image boots with a ~30 GB root volume regardless of boot-volume size | `sudo /usr/libexec/oci-growfs -y && df -h /` |
| Container runtime | Docker + nvidia-container-toolkit | `sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` |
| Disk | 25 GB free for the q4_k_m GGUF and projector | `df -h` |

**Deploy**

```bash
sudo /usr/libexec/oci-growfs -y
sudo dnf -y install dnf-plugins-core && sudo dnf config-manager --add-repo https://download.docker.com/linux/rhel/docker-ce.repo
sudo dnf -y install docker-ce nvidia-container-toolkit && sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl enable --now docker
```

```bash
sudo docker run -d --name llamacpp --gpus '"device=0"' -p 127.0.0.1:8080:8080 \
  -v $HOME/models/gguf:/models ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf --mmproj /models/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf \
  -a muse-glimmer -ngl 99 -c 131072 -np 1 --jinja --host 0.0.0.0 --port 8080
```

The GGUF files come from `meta-models/Muse-Glimmer-30B-GGUF`. The server answers `/health` about 5 s after start. Confirm the deploy with the tool-calling check in [`../inference-server/vllm.md`](../inference-server/vllm.md); on this platform it returned `get_weather(city="Paris", units="celsius")` with `finish_reason: tool_calls`.

**Supported precisions**

| Precision | Status | Artifact | Memory observed | Notes |
|---|---|---|---|---|
| q4_k_m | Verified | `meta-models/Muse-Glimmer-30B-GGUF:Q4_K_M` (15.6 GiB) | 19.2 GB VRAM on one A10 with `-c 131072 -np 1` | Tool calling round-tripped. |
| fp8 | Planned | `RedHatAI/Muse-Glimmer-30B-FP8-block` via vLLM, TP=2 | | First attempt left no room for KV cache at 32k context with the vision encoder loaded; retry in progress. |
| bf16 | Not supported | `meta-models/Muse-Glimmer-30B` | | 60 GB of weights do not fit two 24 GB A10s. Use BM.GPU.A10.4 (TP=4) or an H100/H200 shape. |

**Troubleshooting**

| Symptom | Cause | Fix |
|---|---|---|
| `docker pull` or the weight download fails with "no space left" soon after boot | The Gen2-GPU image's root LV is ~30 GB | `sudo /usr/libexec/oci-growfs -y` before pulling anything. |
| `permission denied ... docker.sock` in scripts run over `ssh host 'cmd'` | Non-login shells don't pick up the new `docker` group | Use `sudo docker`, or log out and in. |

## Support

Where a reader takes a problem with these platforms — your issue tracker, forum, or developer docs. Not a sales contact.

- Issues: https://github.com/ggml-org/llama.cpp/issues (runtime) and https://github.com/oracle-quickstart/oci-ai-blueprints/issues (OCI deployment)
- Docs: https://docs.oracle.com/en-us/iaas/Content/Compute/References/computeshapes.htm
- Maintainer: Federico Kamelhar (github.com/fede-kamel)
