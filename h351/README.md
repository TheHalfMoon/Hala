# H351 — Prospective genuine-Qwen Arabic click-miss diagnosis (private, free-only)

**Status:** CODE PREPARED; **NO NEW GPU SUBMITTED, NO MODEL RESULT**.
No adaptation, new OOD, official TRAIN admission or release permission.

## Why this is different from another CPU test

H336's independently observed original-first-party, *public/exposed*
Qwen3-VL first-click study had EN **6/6** correct station-open/focus,
Arabic RTL **4/6** correct and **2/6 no station opened**. No raw click
points or individual case data were exported or retained. H333's simple
RTL guidance did not improve exposed source success. H337 then
implemented a source-native trusted evaluator capable of determining
whether a non-opening click was inside the intended button, within
12 pixels of its bounding rectangle, within 12–48 pixels, or farther
than 48 pixels. **H337 did not run a genuine model.**

H351 connects that *already-qualified* H337 evaluator to a real
preloaded Qwen and its model-time token filter in a distinctly new,
frozen **12-scene** research design. The true original browser observes
a genuine model's first-click action, inspects state **after** native
dispatch and returns only broad aggregate English/Arabic categories.
The model sees only the screenshot and user-facing task; no DOM,
target identifier, reference rectangle, gold point or source truth is
ever given to the model callback.

### Falsifiable interpretation

If Arabic misses cluster within 12 pixels of the intended button,
coordinate calibration is a plausible hypothesis. If they are mostly
farther than 48 pixels, visual station localization is a more relevant
training hypothesis. If the clicks are inside but inert, the source
interaction path needs examination. The experiment is **not** powered
to establish language generalization, adaptation, or a causal effect;
its public/exposed source is not blinded or independent OOD. No
individual predicted coordinate or per-scene result is published.

## Exact freeze

- Source GitHub `TheHalfMoon/Hala`, branch
  `ps/h350-development-training-official-admission`,
  immutable head `b925b2d9c66ff2a8a38e766eb1eec03d09c4d61c`.
- Original H319 BeaconBoard public HTML Git blob
  `6e8467e123c7f0fbac77ea553622ab8e659f9c16`.
- Pretrained `Qwen/Qwen3-VL-2B-Instruct` model revision
  `89644892e4d85e24eaac8bacfd4f463576704203`.
- Source original 12 scenes, alternating EN/AR; 12 true Qwen
  first-click model generations; model-time grammar per call; real
  native browser click/state/focus; no optimizer/weights.
- One distinct private Kaggle slug
  `abdulazizshehri/hala-h351-qwen-miss-distance-t4`.
- One script + private metadata, no datasets, screenshots, points,
  weights, labels, or private output in the uploaded payload.
- Only one optional private success file
  `hala-h351-exposed-qwen-miss-bands.json`, bounded to 4 KiB, or
  categorical stage-only failure without raw errors. Exact allowlisted
  private receipt must be parsed locally by
  `h351/verify_private_receipt.py`, followed by independently checked
  Kaggle job identity/privacy/weights/commit.

## Qualification and non-submission rules

CPU-only:

```sh
python -m unittest tests.test_h351_private_qwen_distance tests.test_h337_source_miss_distance_bands tests.test_h336_private_first_click_qwen -q
```

A dedicated Ubuntu/Windows x Python 3.11/3.12 workflow runs invented
contract tests. These CPU tests deliberately simulate category counts
and can never prove that real Qwen produced them. H351 code does not
start a GPU task. A future owner-private job may be submitted only after
exact-head hosted CI is fully green, privately authenticated rights for
this *source-exposed inference-only* use (not training), actual owner
Kaggle free quota and no concurrent GPU job are verified, and a
single create-only private kernel is approved. Lack of available
quota/privacy/consent is a **NO_GPU_SUBMITTED** outcome, not an
invitation to use paid compute or silently retry.

The model needs new rights-cleared TRAIN, source-family-disjoint DEV,
independently custodied fresh blinded OOD and genuine BASE vs ADAPTED
native bilingual gains before release. H199 official TRAIN registry is
empty, H343 rights review has no independent approval, H173 sealed,
H212 barred, H250/H251 quarantined, H293/H294 blocked. No merging,
publishing weights, paid quota, deployment or release.
