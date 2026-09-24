# Upstream Compatibility Pins

The native compatibility lock selects ExecuTorch
`20ad5ee43ff53804030899d621590af3daadda53`. This immutable `main` commit
contains every required upstream interface; release readiness remains gated on
final artifacts and clean-machine end-to-end validation.

## Capability status

- ExecuTorch PR [#22063](https://github.com/pytorch/executorch/pull/22063):
  base Supertonic export and native runtime landed at
  `81969a92dd2e5515fa23ccdf9d87346cf3ba2ba2`.
- ExecuTorch PR [#22070](https://github.com/pytorch/executorch/pull/22070):
  bounded generic LLM worker cancellation and health propagation landed at
  `5bd86e50fcd986999e4c09b82de040a3ba224466`.
- ExecuTorch PR [#22208](https://github.com/pytorch/executorch/pull/22208):
  persistent Supertonic JSONL mode landed at
  `20ad5ee43ff53804030899d621590af3daadda53`. Its protocol-v1 ready frame
  reports `sample_rate: 44100` together with load and warmup timing so the
  Python adapter and native runtime enforce one schema.

The selected compatibility commit is a verified descendant of #22063 and
#22070 and contains the landed #22208 tree. Every ExecuTorch-built artifact
must be produced from this one clean checkout. Branch names, dirty checkouts,
and mixed Python/native revisions are forbidden.

Before setting `ready_for_release` to true:

1. Populate artifact sources, revisions, checksums, sizes, and exact licenses.
2. Run real generation, stream cancellation, and post-cancel generation.
3. Verify multiple Supertonic utterances reuse one warm process using the
   documented protocol-v1 ready frame.
4. Pass a clean-machine macOS arm64 end-to-end run.
