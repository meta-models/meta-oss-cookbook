# Hosted

Muse Glimmer served as an API by providers who run the hardware for you. One page per provider: the exact model string, what the provider publishes about it, what it costs, and a call that proves tool calling works.

| Provider | Model string | Page |
|---|---|---|
| Together AI | `meta-models/Muse-Glimmer-30B` | [`together-ai.md`](together-ai.md) |

## Why this sits in a local-first cookbook

Muse Glimmer is open-weight, and the rest of this repo is local-first on purpose: everything from [`quickstart/`](../quickstart/) onward runs on your own hardware, offline, with no API key and no gated access. That is the point of the model and it doesn't change. Open weights also mean nobody is forced to operate the hardware themselves — this folder is for when you'd rather someone else did, and it is the one place in the repo where an account and an API key are required.

If that trade isn't what you want, the alternatives are already here: [`inference-server/`](../inference-server/) to serve it yourself, [`platform/`](../platform/) for specific hardware.

## How to read a provider page

- **The model string is the contract.** It is the only thing your code needs, and it is the first thing on every page. Providers pick their own; it will not match the HuggingFace repo name.
- **Provider numbers belong to the provider.** Performance figures, quantization details and quality evals are attributed and dated. They are not re-run here, and they describe the provider's serving stack — not something you configure.
- **Prices and limits go stale.** Each page states the date it was read from the provider's own page, which is linked. Check it against the provider before you budget on it.
