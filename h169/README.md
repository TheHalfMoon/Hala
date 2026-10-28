# H169 one-shot fresh DEV paired native reward — PREDECLARED

**Research objective:** compare real new original-app browser success of the
frozen H164 language adapter only (BEFORE) vs the same adapter with the new
H168 visual-merger adapter active (AFTER). A real paired quality comparison,
NOT a training or hyperparameter selection exercise. Reusing the consumed
H140/H162/H163/H164 Pell/Pikaday OOD is forbidden.

## Strict data boundaries

- **New DEV ONLY**, not OOD: same upstream app *families* as TRAIN
  (Choices.js and noUiSlider), but new source variants **16–31** vs H168
  TRAIN 0–15, new screenshots, different text entry seeds and slider states.
  16 original native typing tasks and 16 drag tasks; 16 Arabic/16 English.
- Owner Windows captured 32 reference-success cases on clean pinned MIT
  upstream original source. A second Chromium pass re-rendered all **32/32
  screenshots byte-identically** and checked SHA against the previous
  TRAIN and historical consumed source registry. No historical OOD pixels
  were opened. Owner reference gold stays under D drive exclusively.
- Private DEV screenshot/task input ZIP SHA-256
  `2dca00e6c6a0983529c625f47c863f9e5d19c94d12ac51341b4aa93bc1f2a215`.
  Manifest contains **zero training supervision, owner reference coordinates,
  owner reference actions, or DEV/old OOD gold**.
- Model A: the unchanged H164 frozen language LoRA, exactly SHA
  `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
  Model B: the same language LoRA plus exact H168 vision-only LoRA SHA
  `506f571d0ff45bbe8932d9ea1f56dd6e4b5b5dc5f90cfdd1397243285523cd88`.
- Run exactly **32 paired cases × 2 frozen models = 64 inferences**
  on FREE private Kaggle Tesla T4, no optimizer, no weight save, no labels.
- **Native owner browser reward:** replay each frozen PNG from the original
  upstream on Windows; execute ONLY model-parsed native pixel type/drag
  actions (invalid, wrong-kind, abstention = failure). For Choices, the
  source-native tag must actually be added matching the user's *visible*
  task instruction; for noUiSlider, the native slider value must increase.
  Never give oracle coordinates, selectors or private gold to the model.
- Calculate before/after success, Arabic and English success, type/drag
  success, paired wins and regressions. After one scoring run, mark the
  DEV scenes permanently consumed; do not tune or retest on them.
- This DEV is **same-app-family** and cannot establish unrelated new-app
  OOD generalization. Require future distinct new app families and locked
  unseen OOD before any broad model quality or release claims.

No paid model/GPU, no source or owner reference in GitHub, no public model
weights, no Hala 35B edits or main branch merge. Draft PR only.
