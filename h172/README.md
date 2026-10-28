# H172 — preregistered owner-only novel screenshot DEV for H171 evaluation

**This is same-app-family DEV, not unseen-app OOD.** The H171 visual
checkpoint was frozen and independently verified before any H172 model
evaluation. Do not train on, repeatedly score, or tune against H172.

Freeze exactly 32 new original-app Chromium screenshot scenarios before GPU:
16 pinned unmodified MIT flatpickr original page, **November 2026** dates
(not H171 TRAIN October dates), plus 16 pinned unmodified MIT Awesomplete
original autocomplete **Node.js** suggestions (not H171 JavaScript).
16 Arabic and 16 English; only native click actions. All source tasks must
succeed inside the *original* application when owner reference actions are
applied. The separate source folder has `manifest.json` and
`owner-reference-private.json` and 32 images; no owner gold can be uploaded
with model input. Exact browser RGBA pixel replays are mandatory.

Data must be disjoint from all previously consumed H164/H169 DEV/OOD and
H168/H171 TRAIN by SHA/source identity, with previous H169 DEV consumed lock
checked before generating the H172 scenes. The first `new-dev32` capture was quarantined: case 172113 changed one
real RGB pixel at (118,681) on independent browser replay. The original
source and first capture remain intact. Replace that single scenario with
an alternative original Node.js prefix and freeze a fresh `new-dev32-v2`
capture. If any of the 32 fail actual original native action or exact
pixel replay, quarantine the entire capture; never relax the check,
reuse old DEV or edit source screenshots.

Compare on a single preregistered frozen **32 paired cases x two models**:
(A) H164 frozen language adapter only, (B) same frozen H164 language
plus the **H171 rank16 visual adapter**
`8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e`.
Private FREE T4 only, no optimizer and no owner gold on GPU.
Owner-native one-shot Chromium reward verifies actual selected calendar date
or autocomplete value, rather than pixel-distance to gold. Score Arabic,
English, two apps, 32 paired wins/regressions. Mark H172 DEV consumed after
native reward and never use its scenes for tuning/retraining. Require
**unrelated original-app unseen OOD** after freezing model selection.

No paid GPU, no Hala flagship 35B change, no model publish or release,
no merging the stacked Draft PR into main.
