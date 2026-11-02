# H372 — Public source-browser first-frame substage diagnosis (NO TRAIN)

**Research-only, no weights, model forward/optimizer, private screenshots, legally admitted TRAIN or blind OOD.** Stacked on H371 PR #401 with Python 3.11 availability fix at head `f579b3622cf6e6cef9a05019259cd930ff066312`.

## Real failure motivating this improvement

H371's dedicated three-trial source-browser repeat matrix succeeded on **both Ubuntu Chromium and Windows Edge**: each ran 3×(9 original H319 + 7 H337) = 48 original public synthetic browser test invocations, 96 across both OSes. However a *separate inherited H319 Ubuntu Chromium job* failed again on exact H371 head. H319 original dynamic bilingual test `test_all_12_source_native_dynamic_sequences` yielded a bounded H370 diagnostic:

`H319 DYNAMIC_SOURCE_OR_POLICY_REFUSED stage=first_render_and_policy kind=browser_error`

Evidence: [run 41516883013 / Ubuntu job 125522594441](https://github.com/TheHalfMoon/Hala/actions/runs/41516883013/job/125522594441). The particular scene was public synthetic `beacon_04`. This DOES NOT establish a defective Qwen action, user/private data error, model gain, or that the whole first-render stage has a single root cause. Source-browser intermittence remains unresolved; H371 full inherited CI is NOT green.

## Precise bounded diagnostic added

The original `_observed_predict` starts with an original rendered screenshot, computes the decoded model-visible RGB digest, calls the source-test-owned fake prediction callback, then checks that the frame did not change during that callback. A Playwright `browser_error` can arise at multiple points, including the screenshot and DOM geometry lookup within the callback, so `first_render_and_policy` was too coarse to identify a causal fix.

H372 retains the original fail-closed source safety behavior, token/action validation, original page hashes and browser network blocking, and adds only an **optional internal callback recording a fixed enum** before each subphase:

- `first_image_capture`, `first_image_digest`, `first_policy_callback`, `first_post_policy_frame`
- For H319 second step, distinct `second_*` equivalents; H337 only uses the first group

The external output is still H370's original refusal prefix plus an allowlisted `stage` and `kind`, **never raw Playwright exception text, tracebacks, scene IDs, source text, screenshot/image bytes, coordinates, labels or secrets**. Existing call sites without a diagnostic callback remain compatible. H372 uses invented source-free failure injection to verify exact phase classification and secret suppression; a new 4-way Windows/Ubuntu × Python3.11/3.12 source-only CI job runs those tests. Inherited original public-browser CI remains independent and can now reveal which subphase fails if its prior intermittence reproduces.

**This instrument alone is not a root-cause fix**, and a green attempt does not explain prior failures. Fix only after a reproduced, bounded cause, without retries that hide red signals or weakening role/source, screenshot and native-action assertions.

## External blockers to actual Hala model completion

H199 approved lawful TRAIN manifests EMPTY; H355 independent reviewer key registry EMPTY and [rights Issue #384](https://github.com/TheHalfMoon/Hala/issues/384) unresolved; H356 independent sealed never-exposed source-family-disjoint EN/AR blind OOD custodian EMPTY and [Issue #385](https://github.com/TheHalfMoon/Hala/issues/385) unresolved. All H357/H358 synthetic examples are exposed public TRAIN/DEV **review candidates**, not official lawful TRAIN nor sealed OOD. No LoRA training or demonstrated learned adapted-vs-base model gain, no release or merge to main.
