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
