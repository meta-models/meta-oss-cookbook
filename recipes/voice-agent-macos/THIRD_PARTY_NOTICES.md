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
- **Supertonic**: consult the upstream source and model licenses before
  downloading or exporting assets. Assets are never committed here.
- **Muse Glimmer and Parakeet models**: use is governed by their respective
  model licenses. Model weights and exported programs are never committed here.
- **Inter**: Copyright 2016 The Inter Project Authors
  (https://github.com/rsms/inter), SIL Open Font License 1.1, consumed through
  `@fontsource/inter`.

A release must run the repository's license and publication checks and update
this file when any dependency, model, or asset changes.
