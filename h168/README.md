# H168: fresh original-application TRAIN source and bounded private vision SFT

**Owner Windows evidence: 2026-10-28. Development branch only; no main merge/release.**

## New licensed original sources, not previous test families

Source checkouts are **owner-private on D:** and unmodified clean upstream Git trees. This project does not commit donor source, binary assets, private screenshots or owner-native supervision.

| Source | Pinned upstream commit | MIT license | Original upstream page | Status |
| -- | -- | -- | -- | -- |
| [Choices.js](https://github.com/Choices-js/Choices) | `e132628ac74dd427a9a454caf042c65d3065d170` | `LICENSE` | `public/test/text/index.html` | 16 original native typing TRAIN |
| [noUiSlider](https://github.com/leongersen/noUiSlider) | `57033672d07e8bac527d7147721a4a2e07e0d699` | `LICENSE.md` | `tests/origin.html` | 16 original horizontal/vertical dragging TRAIN |
| [flatpickr](https://github.com/flatpickr/flatpickr) | `548330e6e18871d6ca1ca78746c0e89cb12a0303` | `LICENSE.md` | `index.template.html` | pinned candidate only, **not captured/trained** |

Licensing checks require the original MIT license text to be present and git status clean. Checks against previous consumed source metadata use SHA-256 references only, never open previous OOD screenshot bytes.

## Completed TRAIN-only proof

`python -m h168.capture --owner C:\Users\Shehr\work\Hala --upstream D:\Hala-H168-Private\upstream --destination D:\Hala-H168-Private\train-pilot-32 --mode capture`

One-shot capture completed **32/32 actual original-browser successful reference actions** in Playwright Chromium (960×720). Sources were then opened again in a new browser session:

`python -m h168.capture ... --mode verify`

Independent pass **32/32 exact PNG byte replay matches**. No manually fabricated label; reference click/type/drag actions were actually executed and original app state was checked after the action. 16 Arabic and 16 English instructions, 16 real type and 16 real drag cases, two original upstream project families. The private image manifest and owner TRAIN-only gold reference are separate files on D.

Owner-private training ZIP was prepared and audited using `python -m h168.pack_train`: exact SHA-256 `d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6`, 32 TRAIN images, **0 DEV, 0 OOD**. Private Kaggle copy uses `.pkg` extension to prevent automatic archive extraction and preserves exactly the same digest.

## Boundaries for free GPU

The private Kaggle training experiment uses **only new TRAIN** ZIP and the exact previously frozen H164 language adapter (SHA-256 `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`). It trains rank-4 `visual.merger` LoRA, fixes seed and schedule to exactly 32 image exposures/steps, requires T4 and peak allocated memory <=13 GiB, freezes previous language LoRA and all base weights and stores **only the vision adapter privately**. There are no DEV or OOD model calls in this stage. Loss values or gradient signals are training diagnostics, **not evidence of native browser success or Arabic improvement**.

After the bounded GPU SFT, independently create entirely NEW unconsumed original-app DEV (no gold given to GPU model) and two or more unseen original-app OOD families, preregister outcome rewards, and then measure real native outcomes before using H168 for selection. Never use H140/H162/H163/H164 consumed OOD or any old DEV as training/selection.

No paid models/compute; no model published; no Hala 35B change; keep stacked Draft.
