# Meta OSS Cookbook

**From your first prompt to a working agent, then make the model your own.**

Practical recipes for building with Meta's open-weight **Muse Spark Open** and
**Muse Glimmer** models. Run inference, serve an endpoint, build tool-using
agents, and adapt a model with supervised fine-tuning or reinforcement learning.
Choose a local setup, your own GPU infrastructure, or a hosted provider.

[Choose your model](#choose-your-model) ·
[Pick your path](#pick-your-path) ·
[Explore recipes](#recipes-worth-exploring) ·
[Contribute](#contributing)

## Choose your model

| **Muse Spark Open** | **Muse Glimmer** |
|---|---|
| **Agentic and coding workloads at scale.** A sparse mixture-of-experts model with 816B total parameters and 42B active per token. | **Local-first agents on your own hardware.** A 30B model for tool use, multi-step tasks, and multimodal applications. |
| Text, image, audio, and video input; text output. | Text and image input; text output. Image support depends on the runtime. |
| Self-host on multi-GPU infrastructure, explore the Mac Studio recipe, or use a hosted API. | Start with quantized single-GPU inference or an Apple-silicon runtime; use larger hardware for BF16 and post-training. |
| [**Explore Muse Spark Open →**](muse-spark/) | [**Explore Muse Glimmer →**](muse-glimmer/) |
| [Get the model](muse-spark/quickstart/) · [Hosted providers](muse-spark/hosted/) | [Quickstart](muse-glimmer/quickstart/) · [Hosted providers](muse-glimmer/hosted/) |

> **Choose the setup, not just the model.** Hardware needs depend on checkpoint
> precision, runtime, context length, and whether you are serving or training.
> Check the linked guide for requirements and verification status before you run.

## Pick your path

Both cookbooks take you from setup to an application. Jump straight to the part
you need:

| I want to… | Muse Spark Open | Muse Glimmer |
|---|---|---|
| **Get the model running** | [Download and setup](muse-spark/quickstart/) | [Quickstart](muse-glimmer/quickstart/) |
| **Prompt it and call tools** | [Prompting, reasoning, and multimodal input](muse-spark/prompting/) | [Chat templates, tool calls, and the agent loop](muse-glimmer/agentic-fundamentals/) |
| **Serve on my own hardware** | [Inference runtimes and settings](muse-spark/inference-server/) | [Inference runtimes](muse-glimmer/inference-server/) · [Hardware guides](muse-glimmer/platform/) |
| **Use an API without provisioning GPUs** | [Hosted providers](muse-spark/hosted/) | [Hosted providers](muse-glimmer/hosted/) |
| **Build an end-to-end application** | [Deployment recipes](muse-spark/recipes/) | [Agent recipes](muse-glimmer/recipes/) |
| **Fine-tune or train with rewards** | [SFT and RL](muse-spark/post-training/) | [SFT and RL](muse-glimmer/post-training/) |

## Recipes worth exploring

### Run beyond a single machine

- **[Muse Spark Open on Mac Studio](muse-spark/recipes/muse-spark-open-mac/):**
  Serve the NVFP4 checkpoint across Thunderbolt-linked Mac Studio computers
  with MLX.
- **[Muse Spark Open post-training](muse-spark/post-training/):**
  Explore TorchTitan SFT and RL, Miles, TRL, and hosted training options.

### Build a local agent

- **[Computer use](muse-glimmer/recipes/computer-use-web/):**
  Let Muse Glimmer navigate a website through screenshots and mouse actions.
- **[Voice on macOS](muse-glimmer/recipes/voice-agent-macos/):**
  Build a local conversation stack with speech recognition, Muse Glimmer
  generation, and speech synthesis using ExecuTorch on Apple silicon.
- **[Alert triage](muse-glimmer/recipes/alert-triage-copilot/):**
  Turn alerts into a three-stage structured pipeline that grades its own output.
- **[Sandboxed tool use](muse-glimmer/recipes/sandboxed-agent/):**
  Explore an agent sandbox that blocks attempted credential exfiltration.

### Make the model your own

- **[Structured output](muse-glimmer/recipes/structured-output/)** and
  **[reasoning control](muse-glimmer/recipes/reasoning-effort/):**
  Enforce a response schema and explore the cost of different reasoning budgets.
- **[Supervised fine-tuning with Axolotl](muse-glimmer/post-training/sft/axolotl/):**
  Adapt Muse Glimmer with QLoRA, LoRA, or full-parameter training.
- **[Data-analyst RL with TitanRL](muse-glimmer/post-training/rl/titanrl/data-analyst/):**
  Train tool-using behavior with a machine-checkable reward.

## What to expect

- **Commands you can use, context you can understand.** Setup steps, examples,
  and explanations live together in the model-specific guides.
- **Hardware and runtime details where they matter.** Follow the recipe's
  checkpoint, precision, and server requirements. Measurements apply to the
  documented configuration, not every deployment.
- **A clear distinction between local and hosted.** Local inference can run
  offline once the required artifacts are cached. Hosted APIs require provider
  credentials; setup and training may need downloads or online tracking.
- **Room to adapt.** Use the examples as starting points for your own tools,
  tasks, datasets, and deployments.

The cookbook is growing. Some paths are verified, while others document a
pending or unverified setup. Each model folder and runtime guide states its
current status; a listing here is not a blanket verification claim.

## Contributing

Add a recipe, improve a guide, or share a reproducible fix. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the
[Code of Conduct](CODE_OF_CONDUCT.md). For Muse Glimmer recipes, start with the
[shared recipe template](muse-glimmer/assets/RECIPE_TEMPLATE.md).

## License

See the repository [LICENSE](LICENSE). Model weights have their own terms:
[Muse Spark Open's model license](muse-spark/README.md#license) and
[Muse Glimmer's model card](https://huggingface.co/meta-models/Muse-Glimmer-30B).
