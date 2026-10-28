# H173 — original-application OOD: independently verified, permanently consumed

**CURRENT STATUS (2026-10-29):** Owner v3 completed 32/32 original-app capture, 32/32 exact independent pixel replay, 64 real frozen GPU proposals on a PRIVATE FREE T4, and one-and-only 64-action original-browser native reward. Independent GPU and native audits PASSED, and the H173 source is PERMANENTLY CONSUMED. H164 baseline 2/32 (6.25%) versus H164+H171 vision 3/32 (9.375%), one paired win, no regressions, two-sided exact McNemar p=1.0 (NOT significant). Tiny Slider 0/16 for both. **No general computer-use or production qualification.** Owner images, click gold, predictions, and consumption receipts remain private. v1 and v2 permanently quarantined. NEVER rerun, rescore, train on, or select using v3. Future work requires genuinely new TRAIN/DEV/held-out OOD; see h174/README.md. All earlier statements below are preserved as dated source-preflight history, NOT current status.

## Historical 2026-10-28 source-only preflight (superseded)

**Source-only preparation on owner Windows, 2026-10-28. No model inference at that time.**

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

The initial Windows source generator remained local when the capture
failed. The GitHub Draft branch now includes an independently authored
replacement `h173/capture_ood.py`, `h173/stable_capture.py`,
`h173/pack_ood.py` and fail-closed unit tests. The replacement
**normalizes the pointer to (0,0) after target scrolling**, demands
two identical decoded 960×720 RGBA screenshot rasters, fails if the
modal close control is still hovered, and writes a private 32/32
independent-replay receipt. OOD packaging refuses missing, altered or
partial replay receipts and strips owner native reference actions.

These changes are *engineering safeguards*, **not an owner-device
qualification**. They have not yet been executed against a fresh
original Tingle/Tiny Slider capture on Windows. Desktop Commander
reached its monthly usage limit during diagnosis; do not retry,
reconnect or bypass that quota. The current Draft PR is a source
preflight and fail-closed repair candidate, **not qualified OOD**.

**Forensic hypothesis, not yet verified:** the replay SHA prefix
`c6155737492a` exactly matches the earlier independent owner pilot's
**Tingle modal 2 close-button-hover** screenshot, whereas the original
capture SHA prefix `f2ff381cf881` matches the pilot's non-hovered modal 2
screenshot. This makes **browser pointer/hover state persisting across
original modal scenes** a strong candidate for the 6,149-pixel drift.
Do not assert it is fixed without a fresh independent complete replay.

**Recovery protocol:** explicitly place the mouse at a stable neutral
position (e.g. `page.mouse.move(0,0)`) after original dialog setup and
before the screenshot; independently verify both repeated exact RGBA
raster and the real reference native close action, then diagnose
remaining modal focus/animation state if discrepancies persist.
Keep the original source unmodified. Recapture into a *new* private
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


## H173 additional engineering gates (2026-10-28)

The code now contains the **FULL fail-closed candidate evaluation
pipeline** in `h173/capture_ood.py`, `stable_capture.py`,
`pack_ood.py`, `infer_gpu.py`, `audit_gpu.py`,
`kaggle_private_paired_ood.py`, `native_reward.py`, and
`audit_native.py`. See `OWNER_ONE_SHOT_RUNBOOK.md` for the
exact private source and dataset boundary. This is **engineering-only**;
none of these files imply qualified H173 OOD model performance.

The source-only GitHub-hosted original-browser Playwright checks have
completed **32/32 original Tingle/Tiny Slider source-native actions and
32/32 independent exact RGBA replay on Linux**. The Windows hosted
equivalent also independently passed the 32/32 source-only check after
Git autocrlf was disabled before checking out the pristine upstream.
Both services use no owner source images, owner reference coordinates,
GPU model or old benchmark images; they do not replace the missing NEW
owner-private one-shot OOD and are not model metrics.

The model comparison code stays locked until an entirely fresh qualified
source ZIP SHA is pinned. The frozen T4 runner and independent GPU auditor
cannot pass with the literal `UNQUALIFIED` placeholders. Independent
owner-native scoring rejects any unverified GPU job and records an
irreversible consumption lock **before the first actual model action**.
Tiny Slider rewards require the exact reference **post-index**, so an
opposite-direction movement is counted as failure. Unit tests cover
invalid modalities, duplicate pixel images, unqualified source metadata,
mutated GPU receipts, false native successes and interrupted locks.

**Exact Git source pinning correction (2026-10-28):** Git commits cannot
contain their own SHA as a fixed literal and still keep that SHA valid.
The private standalone Kaggle runner instead reads the exact frozen
commit via `HALA_H173_PINNED_SOURCE_SHA` from the owner notebook
environment. It verifies the detached checkout SHA before importing
the model, and the independent auditor requires the owner to pass the
same SHA with `--expected-source-sha` (or via that environment
variable before native scoring). Mutable branch names or missing SHA
are rejected. The private ZIP SHA remains `UNQUALIFIED` until a NEW
32/32 qualified owner source exists. In addition, owner native reward
now checks every packaged GPU PNG SHA against its corresponding
frozen original owner PNG rather than trusting only a ZIP manifest.

**No automatic release:** original Windows owner capture, PRIVATE FREE
T4 64 frozen OOD predictions, real original-app one-shot reward, Hala
35B training and production-readiness remain unqualified.
