# <Recipe Name>

<!--
  Shared recipe template. Copy this into a new recipe folder and fill each section.
  Every recipe in the cookbook follows this structure so partner-contributed
  recipes stay apples-to-apples.
-->

One sentence on what the reader ships with this recipe and why they'd reach for it.

## Recipe banner

<!--
  Report only what you measured on the hardware you actually ran on. Don't state a
  hardware tier or an expected tok/s for silicon you haven't tested.

  Every recipe is verified at bf16 and served with vLLM, so those two rows are
  fixed. If a recipe genuinely needs something else, say so and explain why.
-->

| | |
|---|---|
| Precision | bf16 |
| Model server | vLLM |
| Offline? | Yes / requires network for X |
| (optional) Max VRAM observed | e.g. ~60 GB (batch 1, 8K context) — the peak you actually saw |
| (optional) Requires | other major/agentic libraries required in the recipe |

## Quickstart

```bash
# ONE command that runs the recipe. Explanation comes after.
```

## What just happened

Explain the flow after the reader has seen it run.

## Make it yours

The extension hook: the two or three things a reader changes first, and how.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| … | … | … |

<!--
  Escape pipes inside table cells: write `<\|eot\|>`, not `<|eot|>`.
  An unescaped `|` splits the cell and breaks the whole table.
-->

## Next steps

Two or three cross-links to related recipes or reference pages.
