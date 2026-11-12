#!/usr/bin/env bash
# Install from the TorchTitan checkout with a fresh named venv active.
set -euo pipefail

recipe_scripts="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f requirements.txt || ! -f torchtitan/rl/requirements.txt || ! -f scripts/install_rl_env.sh ]]; then
  printf 'Run this script from the TorchTitan repository root.\n' >&2
  exit 1
fi

uv pip install -r requirements.txt -r torchtitan/rl/requirements.txt
constraints="$(mktemp "${TMPDIR:-/tmp}/glimmer-rl-datasets-XXXXXX.txt")"
trap 'rm -f "$constraints"' EXIT
python "$recipe_scripts/dataset_constraints.py" "$constraints"
# The upstream constraint file does not include datasets' transitive bounds.
# Carry them into its nightly resolve so vLLM cannot invalidate the base stack.
bash scripts/install_rl_env.sh --skip-base --constraint "$constraints" "$@"
