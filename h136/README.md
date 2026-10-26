# H136 — original small-handle GUI START-biased native LoRA pilot

H135 owner-private one-time SEALED18 replay selected native 960×720
preprocessing **before** opening sealed images. On those 18 new synthetic
scenes, the frozen external GUI Assist Grounder 2B completed 1 task, and
the H134 24-step LoRA research adapter completed 3. Paired exact p=0.5:
a promising **exploratory** signal, not statistically conclusive or
release-qualified, and not an improvement of Hala's 35B flagship model.
START handle hits were still 7/18 for frozen base vs 6/18 for H134 LoRA,
so H136 focuses on tiny START targets.

## New owner-original data, frozen before any H136 training

`h136/fixture.html` is a revised separately authored offline desktop
fixture derived from the owner-original H135 prototype, with smaller grips
(10-pixel table divider, 15-pixel small control), more distractor controls,
four palettes and six GUI interaction types. It is not from external models,
apps, or copyrighted screenshots. Each image has owner Chromium real
screenshot pixels, an actual instruction, native DOM handle/end bounding
boxes, and a **gold drag that passed with real Playwright pointer events**.

- TRAIN: **288** source-disjoint scenes, IDs 730000–730287, all 6 types
  balanced 48 each; frozen manifest file SHA-256
  `98d789170407261d693cd33de6ae4a2609446340e0720671c1e8017c096e2a13`.
- DEV: **24** independent scenes IDs 830000–830023, 4/type, manifest
  SHA-256 `94f357f317660719e74a4a74408cab18e6c0f8df5166a9380e86ec559c5f9181`.
- SEALED24: **24** independent new scenes IDs 930000–930023, 4/type,
  manifest SHA-256
  `a8d3ea230af8cab8d0902d1f4dfff2f6eded899a7d357bdbc2d4c4273a9e6244`.
- All **336/336** original Chromium reference drags passed. PNG SHA-256
  uniqueness and disjointness across all three splits is independently
  checked. No pixels reuse prior H131/H133/H134/H135 image sources.
- SEALED24 is **owner Windows only** and MUST NOT be attached to the
  Kaggle training or DEV pilot, used in model selection or test-run early.

## Predeclared next adapter experiment

- Frozen upstream external Apache-2.0 GUI Assist Grounder 2B weights:
  `patrickballou/gui-assist-grounder`, revision
  `31632edaa20749c7f532868a42635f7240033106`.
- Train a **new, separately branded Hala research LoRA adapter from base**
  (not Hala 35B), no reuse/ownership claim of third-party base weights.
- Source input stays at the original native **960×720** pixels and the
  pinned Qwen3.5 multimodal processor performs smart resizing.
- Rank8 language-only LoRA, frozen vision encoder, gradient checkpointing,
  optimizer steps capped at **64**, accumulation2 microbatch1. A fixed
  shuffled START:END sample ratio of **3:1** targets handle acquisition.
- Only private FREE Kaggle T4 noncommercial, immutable source git SHA and
  exact pinned runtime; no paid fallback.
- Record **actual optimizer steps**, losses and adapter tensor SHA; assess
  PRETRAINED vs trained START and END bbox hits on DEV24 independently from
  each raw native model response. Keep weights and all screenshots private.
- A DEV24 gain alone is not evidence of generalization. Freeze experiment
  settings/adaptor before **one** SEALED24 examination. The primary sealed
  metric is actual model-generated full browser drag success; secondary
  START handle hits, END hits, pixel errors and paired uncertainty.
  No subsequent tuning on exposed sealed cases.

H135 heldout data is now exposed and **not** used for H136 training or
selection. Genuine H71 real screenshot source-identifiability gate H127
remains open; such data is not being ingested here. No merge, publication,
deployment, production readiness or release. Preserve upstream license and
mandatory model attribution.

**Training status:** source and tests qualified. GPU optimizer completion,
adapter weights, independent DEV and SEALED outcomes must each be separately
verified before any model improvement claim.

## H136 final original-source / GPU / owner-browser qualification (2026-10-26)

**Owner-original source:** exactly 288 TRAIN, 24 DEV, 24 SEALED genuine
Chromium PNG screenshots/drag labels, all 336 reference drags passed.
Source manifests each SHA-qualified; no old H131/H133/H134/H135 PNG reuse.
SEALED24 remained unmounted in GPU training and independent DEV.

**Actual free PRIVATE Kaggle T4 optimizer:** exactly 64 actual language-only
rank-8 LoRA optimizer steps on original 960x720 images, START:END sample
frequency 3:1, frozen externally attributed GUI Assist Grounder 2B
(Qwen3.5, revision `31632edaa20749c7f532868a42635f7240033106`).
Private newly trained safetensors (21,851,336 bytes) SHA-256:
`183680c3224b024c546dbbd1428e7a05ffa1491d57c117aec350a11e04cba1a7`.
Owner Windows independently verified byte SHA, source manifests,
real optimizer-step evidence, all raw native JSON responses and bounding
box scoring.

| DEV24, same generator family | Before training | H136 64-step LoRA |
| --- | ---: | ---: |
| START hits | 0/24 | **20/24** |
| END hits | 16/24 | **23/24** |
| Both target boxes hit | 0/24 | **19/24** |
| Actual Chrome mouse drag task success | 0/24 | **20/24** |

**ONE-TIME preregistered SEALED24:** this set was held out before
training/DEV, manifest SHA-256
`a8d3ea230af8cab8d0902d1f4dfff2f6eded899a7d357bdbc2d4c4273a9e6244`.
A completely separate private FREE T4 run loaded only the frozen base,
the exact SHA-verified adapter and original SEALED24 source; **96 real
neural inference calls, 0 optimizer steps**. Owner Windows independently
reparsed the 96 real native model predictions, rechecked every original
Chromium screenshot byte SHA, then performed **48 real model-generated
mouse drags** with no oracle coordinate correction.

| SEALED24, held-out from training/selection | Before training | H136 LoRA |
| --- | ---: | ---: |
| START hits | 1/24 | **16/24** |
| END hits | 14/24 | **23/24** |
| Both boxes hit | 0/24 | **15/24** |
| **Actual complete browser drag successes** | **0/24** | **15/24** |

Exact two-sided matched discordant-pair p-values on this *specific*
preregistered synthetic test: START p=0.0002746582, END p=0.00390625,
full completed GUI drags p=0.00006103515625. Under the exact paired
test, training helped on this new synthetic heldout set. However all
H136 scenes share a family of owner-authored workstation GUI generators,
so **these results do not establish generalization to independent real
applications**, clinical workflows, novel UI frameworks, Arabic-heavy
tasks, or the separate flagship Hala 35B model. Model origins, external
attribution, and rights stay intact.

**Decision:** H136 is a valid research-adapter improvement for
self-authored synthetic small-handle drags. Keep this adapter and
all original screenshots/outputs privately stored. Do not continue
optimizing against exposed DEV24 or SEALED24. Next H137 must qualify
on genuinely different, legally permitted real application or
third-party-independent task families, Arabic + English, and establish
a cross-application eval-to-real-GUI action gate before any promotion.
H127 independent semantic human review on genuine H71 data remains open.
No publication, merge, deployment, paid cloud or release.
