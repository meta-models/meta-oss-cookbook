# The recipe source

These are the recipe's TitanRL files. Copy the whole directory into a TorchTitan
checkout as a new RL example:

```bash
git -C <torchtitan> checkout 8108e201a   # see the recipe's "Pinning TorchTitan"
cp -r titanrl_files \
  <torchtitan>/torchtitan/experiments/rl/examples/glimmer_data_analyst
```

Then register the module so `--module glimmer_data_analyst` resolves, by adding
it to the `_supported_experiments` set in
`<torchtitan>/torchtitan/experiments/__init__.py`:

```python
_supported_experiments = frozenset(
    [
        ...
        "search_r1",
        "glimmer_data_analyst",   # <- add this
    ]
)
```

Check it worked:

```bash
cd <torchtitan>
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
pytest torchtitan/experiments/rl/examples/glimmer_data_analyst/tests/ -v
# -> 17 passed, no GPU required
```

| File | Role |
|---|---|
| `data.py` | Synthetic csv/log task generator with golden answers; fully offline. |
| `env.py` | The `bash` tool, per-rollout scratch dir, and workspace grading. |
| `rubric.py` | Graded deterministic reward over the env's signals. |
| `rollouter.py` | Wires dataset + env + rubric; sets the rollout turn/token budget. |
| `config_registry.py` | GRPO/DAPO config and GPU split (`..._data_analyst` and `..._smoke`). |
| `tests/` | 17 CPU-only tests. |

> These files are verified against TorchTitan at `8108e201a` -- after
> [#4145](https://github.com/pytorch/torchtitan/pull/4145) (which added the Muse
> Glimmer renderer) and before
> [#4535](https://github.com/pytorch/torchtitan/pull/4535) (which breaks 30B RL
> weight sync; see the recipe README). If TitanRL's interfaces have moved since,
> compare against a shipped example such as
> [`search_r1`](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/rl/examples/search_r1).
