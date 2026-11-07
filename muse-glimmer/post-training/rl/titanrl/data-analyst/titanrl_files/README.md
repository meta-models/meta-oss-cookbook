# Recipe source

Copy these files into a fresh TorchTitan checkout:

```bash
mkdir -p torchtitan/rl/examples/glimmer_data_analyst
cp -R "$COOKBOOK/muse-glimmer/post-training/rl/titanrl/data-analyst/titanrl_files/." \
  torchtitan/rl/examples/glimmer_data_analyst/
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

Use the upstream RL installer in a fresh named `uv` environment first. Then:

```bash
uv pip install pytest
pytest torchtitan/rl/examples/glimmer_data_analyst/tests/ -v
python -m torchtitan.rl.train \
  --module torchtitan.rl.examples.glimmer_data_analyst.config_registry \
  --config rl_grpo_muse_glimmer_30b_data_analyst_smoke \
  --output-dir outputs/rl/glimmer_data_analyst_smoke
```

The module must contain the config factory. The package does not discover
`config_registry.py` automatically. CPU factory tests do not build GPU actors
or replace the full training check.

| File | Responsibility |
|---|---|
| `data.py` | Offline synthetic tasks and deterministic golden answers |
| `env.py` | Bash calls, private task workspace, per-turn artifact grading |
| `rubric.py` | Cumulative partial reward and correctness |
| `config_registry.py` | Full/smoke configs, role models, loader, budgets and recorder |
| `tests/` | Dataset/env/reward and real factory/CLI contracts |

See the [recipe](../) for dependency setup, checkpoint download, W&B offline
mode, inspection, and the bash tool's isolation boundaries.
