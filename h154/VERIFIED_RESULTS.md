# H154 — original MiniWoB++ multi-action source qualification

**Status:** Independent source/action QA verified on owner Windows, 2026-10-27.
**NOT a Hala 2B/35B model score.** H153's 15/16 pertains only to a
different original React Kanban drag test, and cannot be applied to these tasks.

## Pinned third-party source

- Upstream: `Farama-Foundation/miniwob-plusplus`, original external app.
- Git commit: `7fd85d71a4b60325c6585396ec4f48377d049838`.
- License: **MIT** (original LICENSE/copyright retained).
- Checkout: owner-private `private-h154/upstream/miniwob-plusplus/`.
- Run the actual upstream HTML + CSS + JavaScript through local Chromium.
  Source files stay **unaltered**, including attribution.
- Lightweight Playwright uses existing local browser binary; no new
  paid models, browser-service accounts, or dependency-heavy BrowserGym install.

## Frozen H154 source-only qualification

Selected three action families from a separate application, with 4
actual native browser episodes each:

| Original MiniWoB task | Action family | True original positive reward |
| --- | --- | ---: |
| `click-test` | Mouse click | **4/4** |
| `enter-text` | Click, type, submit | **4/4** |
| `choose-list` | Select dropdown, submit | **4/4** |
| **Total** | **3 families** | **12/12** |

Original `WOB_DONE_GLOBAL`, `WOB_RAW_REWARD_GLOBAL`, and episode count
were read **after** browser actions; the source oracle used only the
visible task instruction plus browser interaction and is **never**
reused as a model prediction. **12 unique new screenshot SHA-256s**
were retained under owner-private `private-h154/source-only-miniwob-v2/`.

The preceding *partial* v1 source smoke test verified 8 actions,
then stopped before `choose-list` due to a test-harness instruction
regex error. The corrected v2 is a separate create-once qualification.
Its 12 screenshots have **zero SHA overlap** with the nine captured
during incomplete v1. Neither source pilot is model evaluation;
neither is a valid final cross-app model holdout.

## How to repeat the qualification on a NEW, independent cohort

`python -m h154.miniwob_source_probe --source <original-checkout> --owner-root <owner-Hala-work-root>`

Current v2 source QA is create-once and **must not be overwritten**.
For any later experiment, preregister different tasks or seeds, new
private output path and disjoint image hashes before first model call.

## H155 transfer gate: what the model still needs

The frozen H142 2B research LoRA used by H153 only predicts
`START` and `visible_column_heading` coordinates. It **has no qualified
typed-action, dropdown-selection or click-target role**. It therefore
cannot legitimately be awarded the 12/12 source-oracle score.

H155 should freeze NEW train/dev/eval splits across more than one
application and action family, define image-to-action supervision
(`click(x,y)`, `type(text)`, `select(value)`, `submit`, `abstain`),
use **free non-commercial** compute for bounded research training
only when reproducible, and measure **true task completion** on a
fresh holdout. Original DOM/reference values must never enter the
model input or action controller at evaluation. Report failure,
unsupported action or abstention explicitly — no fabricated
cross-app success.

**Safety of research:** H140 sealed OOD BLIND untouched; 0 new
optimizer steps and 0 neural calls; Hala flagship 35B unchanged;
no media/weights published, repo main merge, deployment or release.
