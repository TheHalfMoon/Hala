# H173 — truly unseen ORIGINAL application OOD source preflight (NOT scored)

**Source-only preparation on owner Windows, 2026-10-28. No model inference.**

## Prior-model selection and strict holdout boundary

H171 rank16 visual merger research checkpoint is **FROZEN**: SHA-256
`8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e`,
on the frozen H164 language-only reference model. H172 one-shot
same-app-family DEV was measured **4/32 native H164 vs 15/32 native H171,
13 paired wins, 2 regressions, p=0.00738525390625**. H172 is
permanently consumed. There is **NO claim of general unseen-app
generalization** from this result and the flatpickr regression is real.

Both next candidate external apps were selected to be **different from
all previous TRAIN/DEV/OOD**. Do not use H140/H162/H163/H164 Pell/Pikaday
or H169/H172 scenes for any future evaluation. No model tuning or
training based on any consumed benchmark.

## Exact original MIT donor evidence, owner private D drive

| Completely new source | Pinned pristine source revision | Original upstream HTML | Native state smoke |
| --- | --- | --- | --- |
| [Tingle](https://github.com/robinparisi/tingle) — native modal | `9eadca70c72ef9adb9144cc23a4104978d3bc63b` | `doc/index.html` SHA `19ff3e667949ec52196b404c766c06f061004da5949057b589bc7bcfe4d388ea` | Actual original modal **opens and closes** in Chromium |
| [Tiny Slider](https://github.com/ganlanyuan/tiny-slider) — carousel | `4d709735c417c2483e77a22d017fc1b18c04f0d4` | `demo/index.html` SHA `76f433876dc2fd5373c62460c9e2151701d8b3c320e386c10337805a03bc8e2e` | Actual original slide index **4 to 7** after pressing next |

Both original source Git trees have original **MIT license** and the two
native interaction families are different (modal visibility vs carousel
index). Both local original demos were successfully loaded and
interacted with in Playwright Chromium using unchanged source HTML,
original vendor JS/CSS, real browser state, and no live external
dependency requests. The private owner-only smoke program is
`D:\Hala-H173-Private\smoke_unseen_apps.py`. **Zero model calls and zero benchmark scores** at the time of initial source
preflight. A subsequent private candidate source capture was attempted
(see quarantine below).

Also researched MicroModal and Glide.js; **neither was admitted** as a
scored source. They are not part of this H173 OOD, nor the frozen set.

## Source-capture attempt and mandatory quarantine (2026-10-28)

An owner-only source generator (`h173/capture_ood.py` in the owner's
Windows worktree) attempted 32 distinct screenshots, 16 Tingle modal
closure scenes and 16 Tiny Slider carousel-navigation scenes, 16 Arabic
and 16 English. The original browser reference action succeeded on all
32 candidate cases, and all image SHA-256 values were distinct from the
previous consumed source registry. Four Tingle browser zoom levels
(0.85, 1.00, 1.15, 1.30) were used to vary native display state; Tiny
Slider cases used the original carousel API to establish distinct
initial indexes. These are explicit browser-state variations, not edits
to upstream Git source files.

**The entire first H173 OOD capture is QUARANTINED, NOT QUALIFIED.** Its
independent exact decoded-RGBA replay stopped at Tingle case `173005`:
the original frozen PNG SHA began `f2ff381cf881`, whereas re-render
SHA began `c6155737492a`. Image comparison found a substantial modal
region discrepancy (RGB difference bounding box x=191..768,
y=235..484, 6,149 distinct RGB pixels), **not a harmless PNG
compression change**. Candidate source files live on the owner's D
only. No gold-free OOD ZIP was admitted or uploaded, **zero H173 model
inferences**, zero OOD native rewards, zero optimizer steps. This
capture MUST NOT be treated as a benchmark score or used for training.

The local Windows repository has an uncommitted source generator and
packaging/tests drafted; they were not promoted to GitHub after the
source failed qualification. Desktop Commander reached its monthly
usage limit while diagnosing this case; further owner-device execution
is blocked until that capability becomes available again. Do not retry
or bypass the quota. The GitHub Draft branch is a source-preflight and
failure-documentation artifact, not a qualified OOD.

**Recovery protocol:** diagnose Tingle's conditional modal rendering,
ensure deterministic pre-snapshot mouse/focus and animation state
without changing the original source, recapture into a *new* private
folder (never overwrite the quarantined original), verify *all 32/32*
exact decoded-RGBA pixel replays including the problematic case, then
and only then freeze a gold-free OOD ZIP and proceed with predeclared
64 frozen paired model proposals and one-shot native reward. No old
H172 DEV or prior OOD reuse.

## Next strict quality gates (NOT YET COMPLETE)

1. Capture exactly 16 *original-app-native successful* **modal**
   tasks and 16 **carousel** tasks in Playwright Chromium with 16
   Arabic/16 English instructions; preregister variants and native
   outcome. Must be visually meaningful and distinct scenarios.
2. Freeze all 32 original 960×720 screenshots and source HTML checksums,
   reject any image matching all previously consumed training,
   DEV or OOD by SHA registry; keep owner reference/gold separately.
   Run an entirely independent **32/32 exact RGBA original browser
   pixel replay**, rejecting entire source on any mismatch. No leakage.
3. Prepare a **gold-free OOD-only private ZIP**; pin immutable SHA
   and source code commit before GPU; NO model has seen these scenes.
4. On only FREE PRIVATE Tesla T4, run exactly **32 paired BEFORE and
   AFTER frozen H164/H171 model inferences**, no optimizer, no older
   consumed scenes. 64 paired proposals must pass independent receipt audit.
5. One-shot owner-only native browser reward: execute only the model
   generated pixel actions, then verify **modal visibility / selected
   carousel index in the original apps**. Invalid/abstained/wrong kind
   counts failure. Do not use gold coordinate proximity. Lock OOD as
   consumed immediately after scoring, compute paired exact McNemar,
   Arabic/English and app-specific successes/regressions.
6. Only then decide whether the H171 research checkpoint demonstrates
   novel-app generalization. Any regression or material safety failure
   blocks promotion. The Hala flagship 35B, real deployment and full
   project completion remain separate unqualified deliverables.

The file `source_preflight.py` deliberately rejects any attempt to
turn a source smoke into an OOD pass without all new 32/64 receipts.

No paid models/GPU, no public donor copying, source screenshots, owner
gold, model weights, output publication or main-branch merge. Draft only.
