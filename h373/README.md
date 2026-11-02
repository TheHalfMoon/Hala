# H373 — fixed, redacted screenshot failure signatures for public-source browser CI

**Source-only engineering science. NO legal TRAIN admission, private screenshots, weights, inference, optimizer, new sealed blind OOD, model improvement claim or release.** Stacked on H372 Draft PR #402.

## Exact observed problem

At H372 exact head `63ca1e05d66914a8cb987b88f11c3a27f9eeb864`, **61/62** inherited GitHub CI checks passed. The only failure was the standalone H371 Ubuntu Chromium public original-browser repeat workflow [run 41518622074, job 125528549757](https://github.com/TheHalfMoon/Hala/actions/runs/41518622074/job/125528549757), whose redacted failure was

`H371_RESEARCH_ONLY FAIL flow=h319 trial=1 stage=first_image_capture kind=browser_error`

H372's new diagnostic (4/4 Windows/Ubuntu source-only CI) isolated the failing original GUI step to **first screenshot capture**, rather than a Qwen prediction or the first-click/typing logic. H319 and H337 standalone browser workflows both passed on that head. This remains an **intermittent browser screenshot-path failure**, not a verified root cause.

Relevant public prior art: Playwright Chromium protocol screenshot capture can report an `Unable to capture screenshot` error ([microsoft/playwright#32144](https://github.com/microsoft/playwright/issues/32144)). Playwright screenshot capture may also wait for font readiness and time out ([microsoft/playwright#32849](https://github.com/microsoft/playwright/issues/32849)). Neither report proves the Hala failure shares its cause.

## Narrow, fail-closed addition

H373 reads only the already-caught Playwright **Error or TimeoutError** inside the source evaluator and transforms the raw error to one of exactly six static labels:

- `protocol_capture`: protocol screenshot capture unable to capture
- `target_closed`: closed page/browser/context
- `execution_context`: destroyed/unavailable JS execution context
- `font_wait_unresolved`: timeout with unresolved font-wait signature (not causal proof)
- `screenshot_timeout`: other Playwright screenshot timeout
- `browser_other`: unmatched error signature

The classifier applies **only** to fixed first/second screenshot capture and post-policy frame stages. H370 keeps its original refusal code, exact redacted stage and exception type and may append `capture=...` **only from this fixed allowlist**, never exception text/paths/URL/source tasks/screenshots/coordinates/gold. H371's existing subprocess-output scrubber now allows an optional known capture bucket **only for Playwright errors in screenshot stages**, preserving strict fail-closed behavior on unknown or malicious content. No browser action execution, screenshot timing, original source hash, network isolation, or admission policy changes. No retry suppresses a failure.

Invented synthetic exception strings containing private-looking labels probe every branch and verify that neither the browser error nor H371 aggregate runner can emit the raw text. A new dedicated four-way Windows/Ubuntu Python3.11/3.12 source-only CI job qualifies those invariants; inherited original browser CI continues to run independently. H373 can discriminate a **future reproducible failure**. It does **not** establish that a known signature has already occurred, or fix the intermittent capture fault.

## Hala training/release remains NO-GO

The H199 official lawful TRAIN SHA manifest allowlist is empty; independently qualified reviewer keys for screenshot and derivative model rights are empty (Issue #384). Genuinely never-exposed independently custodied source-disjoint bilingual EN/AR blind OOD keys are empty (Issue #385). Earlier public author-generated synthetic GUI rows are exposed TRAIN/DEV review candidates ONLY, not legally admitted TRAIN or blind evaluation. There is no demonstrated learned adaptation gain, no legitimate LoRA fine-tuning or release. Keep PR DRAFT and never merge main before independent review.
