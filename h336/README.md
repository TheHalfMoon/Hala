# H336 — Preloaded visual first-click attribution bridge

**Status:** Source-only engineering; no H336 Qwen inference, GPU job, training, model-identity attestation, blind OOD or new measured gain.

## Why this is necessary

H333's one owner-private FREE-T4 pinned pretrained Qwen experiment was a **four-pair publicly exposed** BeaconBoard prompt comparison: control and visual-target candidate both succeeded on 2/2 English and 0/2 Arabic RTL scenes. All four Arabic arm failures had the broad `wrong_focus_or_state` code. This small, paired result rejects *observed improvement on those scenes*; it does not prove statistical equivalence, an Arabic-wide deficiency, or a LoRA gain.

H335 built a source-owned, real-browser five-category first-click evaluator, but did **not** connect it to the existing authentic H323 preloaded screenshot-conditioned model-time policy. H336 adds that missing bridge. In a **future separately authorized** owner-private study, a pinned preloaded model can be asked for just its first click, which is dispatched natively and classified by the trusted original H319 source oracle. The policy never sees DOM, gold station identity, expected point, or source focus state.

## Interface and controls

`run_preloaded_first_click_source(browser, source_path, processor, model, scene, prompt_variant="control")` creates a fresh H323 policy for a single source-exposed scene. The existing H335 evaluator obtains a real 960×720 screenshot, enforces RGB frame integrity and strict native parsing, dispatches **at most one** click, and classifies the source state. The bridge verifies the wrapped model instrumentation: exactly **one** model-time generation and token-filter installation, exactly one distinct image, zero typing, and a click/abstain state consistent with the trusted source result. It refuses extra/private fields, forged booleans, impossible counts and any unsupported attestation.

`run_preloaded_first_click_cohort(...)` runs the 12 **publicly exposed** first-party BeaconBoard scenes (6 EN, 6 AR) and returns only fixed categorical counts by language. A single failure aborts without partial results. The default control H323 policy constructor remains the original three-argument invocation; `visual_target_rtl` is an explicit opt-in and H333 already measured **no benefit** on four exposed scenes. No further identical GPU submission is justified by this bridge.

These counters are wrapper **claims**, not independently verified Qwen identity, GPU provenance or learned gain. The source-only test suite mocks model policy objects and uses invented categorical rows; it is **not** a model performance evaluation. The original H335 full-browser source oracle already passed on both Ubuntu Chromium and Windows Edge at H335 exact head.

## Scientific and privacy boundaries

No new Kaggle job is submitted. H199's official admitted TRAIN SHA set is still empty; H325 first-party RailKit scenes remain candidates only, H250/H251 rights are quarantined, H173 scored OOD is sealed, H212 barred, and H293/H294 owner-private safety blocks must never be bypassed. Independent lawful training rights, genuinely withheld source-disjoint DEV/blind OOD, and a prospective paired BASE-vs-ADAPTED Qwen task-state study are still required before any claim of learned improvement.

No raw model output, predicted points, screenshot, task text, per-scene identity, gold or private receipt may be written to GitHub. No weights, release, deployment, paid compute or protected-main edits. All changes remain in a stacked Draft PR.

## CPU-only tests

```sh
python -m unittest discover -s tests -p 'test_h336_preloaded_first_click_attribution_bridge.py' -v
```

Seven invented-only adversarial tests. The real source/browser and model provenance gates remain separate and are not simulated as passing.
