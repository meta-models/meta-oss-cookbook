# assets

Shared material referenced across the cookbook:

- **`RECIPE_TEMPLATE.md`**: the canonical recipe structure. Every recipe follows it so numbers and UX stay consistent.
- **Partner vendor pages** have their own template: [`../platform/_template/VENDOR_TEMPLATE.md`](../platform/_template/VENDOR_TEMPLATE.md).
- **Diagrams**: architecture and agent-loop diagrams (add as `*.svg` / `*.png`).
- **Sample data**: small offline fixtures recipes can point at. Keep them tiny; large artifacts belong on HuggingFace, not in git.
- **Reference configs**: canonical `generation_config.json`, stop-token settings, and chat-template snippets that multiple recipes reuse.

Keep this folder small and offline-friendly. No model weights, no large binaries.
