# Third-Party Notices

Product-owned source in this subtree is licensed under Apache-2.0. It
integrates with software and assets governed by independent terms; Apache-2.0
does not relicense models, exported programs, native binaries, fonts, or
third-party packages.

## Runtime dependencies

- **ExecuTorch**: BSD-3-Clause. Source and runtime artifacts are provisioned
  separately. Release builds require the immutable revision recorded in
  `config/dependencies/compatibility.lock.json`.
- **LiveKit Agents and LiveKit Server**: Apache-2.0. The temporary
  `packages/livekit-plugins-executorch` package records its API baseline and
  product-owned source provenance in `PROVENANCE.md`.
- **LiveKit Local Inference**: installed transitively by LiveKit Agents and
  distributed under `Apache-2.0 AND LicenseRef-LiveKit-Model`. The model-license
  terms restrict LiveKit model use to the LiveKit Agents framework; see
  `LICENSES/LIVEKIT-MODEL-LICENSE.txt`. This repository does not redistribute
  those model assets.
- **Supertonic**: the converted PTE and upstream assets remain governed by the
  BigScience Open RAIL-M license, including its use-based restrictions.
- **Muse Glimmer**: the PTE and tokenizer files are downloaded from the
  Apache-2.0 `meta-models/Muse-Glimmer-30B-ExecuTorch-PTE` repository.
- **Parakeet**: the converted PTE and tokenizer are downloaded from
  `younghan-meta/Parakeet-TDT-ExecuTorch-MLX` under CC-BY-4.0.
- **Inter**: Copyright 2016 The Inter Project Authors
  (https://github.com/rsms/inter), SIL Open Font License 1.1, consumed through
  `@fontsource/inter`.

A release must run the repository's license and publication checks and update
this file when any dependency, model, or asset changes.
