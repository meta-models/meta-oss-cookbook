# The recipe source

These are the recipe's TitanRL files. Copy the whole directory into a TorchTitan
checkout as a new RL example:

```bash
cd <torchtitan>
cp -R "$COOKBOOK/post-training/rl/titanrl/data-analyst/titanrl_files" \
  torchtitan/rl/examples/glimmer_data_analyst
```

No registration step is needed: select the example with
`--module torchtitan.rl.examples.glimmer_data_analyst`.

Check it worked:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
uv pip install pytest
pytest torchtitan/rl/examples/glimmer_data_analyst/tests/ -v
# -> 17 passed, no GPU required
```

| File | Role |
|---|---|
| `data.py` | Synthetic csv/log task generator with golden answers; fully offline. |
| `env.py` | The `bash` tool, per-rollout scratch dir, and workspace grading. |
| `rubric.py` | Graded deterministic reward over the env's signals. |
| `config_registry.py` | GRPO/DAPO config, GPU split, and the rollout wiring (dataset + env + rubric + turn/token budget); `..._data_analyst` and `..._smoke`. |
| `tests/` | 17 CPU-only tests. |
