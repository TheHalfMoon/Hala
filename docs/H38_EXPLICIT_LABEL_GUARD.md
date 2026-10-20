# H38 — optional exact-label target consistency filter

Status: experimental, opt-in, unqualified; no native computer actions or model-training claims.

## Scope

H37's actual 64 publicly authored browser-rendered synthetic cases gave 32 correct button hits, 16 wrong yet uniquely bound button hits and 16 points without unique control binding. The existing geometry gate rejects the last 16 but cannot determine that a uniquely selected button is the *intended* button.

H38 adds a simple optional filter to the experimental CandidateBackend. It supports only whole-sentence English or Arabic goals that explicitly quote the target label. The selected accessibility name must exactly match that label, and the full observed accessibility snapshot must have exactly one matching name. Unicode NFC is allowed, but not fuzzy matching or diacritic stripping. Unknown goals, duplicated names, inconsistent or malformed names are rejected. **Any declared omitted accessibility elements cause rejection**: a partial observation cannot establish label uniqueness across the actual screen.

The option is `CandidateBackend(engine, require_explicit_label=True)`; its default is `False` so historic candidate behavior is unchanged. A rejected proposal raises `invalid_backend_output`. An accepted proposal still has `execution_authorized=false` and `qualified=false`. The filter does NOT authorize a click or replace independent host permission and freshness checks.

## Actual H37 post-hoc results

On the unchanged H37 original 64 model predictions:
- 32/32 correct predicted buttons were retained.
- 16/16 unique but wrongly chosen buttons were rejected.
- 16/16 unbound points stayed rejected.
- No host action was executed.

These are retrospective safety-filter classifications. HalaType's *neural target selection* still scored only **32/64**. The narrow H37 text syntax supplies a literal expected name in the task, so the result does not establish general natural-language, real-site or Arabic application correctness.

The independent replay first verifies the original raw SHA256 `40b88ceb592446630817b9562238b40765b371fac61ccb4c06b427fab92cef1d` and CPU audit SHA256 `987c8338342faaf8d2718288623a9f485d058acdde0f21f5f3133befaa1451f2`. A deterministic per-case diagnostic is saved at `docs/reviews/h38-h37-explicit-label-guard.json` with SHA256 `5c47dc4c531b7efecd0ccc96a50ae0d2d4b518d1d8328811c844ca7907a14f83`. The original H37 receipts and evaluator are unchanged.

## Limits and next steps

This cannot parse open-ended instructions or unobserved labels; it may conservatively reject valid Arabic variations or screens with inaccessible/malformed labels. A future independent user-intent verifier must be evaluated on genuinely new real-interface tasks before any host action path is considered. H23 CUDA cancellation, independent linguistic and security reviews and release decisions are still open.

Run offline: `python -m unittest tests.test_h38_explicit_label_guard tests.test_h38_h37_actual_replay -v`.

## Actual local source qualification on Windows — 2026-10-20

- H38 initially focused **11/11** CPU tests passed; a subsequent explicit incomplete-observation red regression reproduced an incorrectly accepted proposal, then was corrected to fail closed (focused **12/12** after fix). One observed red test exposed ignoring a malformed bidi-spoofed observed name; code fixed to reject the full invalid snapshot, then red/green rerun passed.
- Full Hala regression after the incomplete-observation fix **520 tests run / 518 passed / 2 skipped**; `compileall` PASS. `python -m compileall` passed.
- Genuine Graft **wiring-only** build/check after the incomplete-observation fix passed **1,617 nodes / 4,337 edges**. Deep meaning tier was **not run**.
- Independent PStack / Alibaba LLM / Jev / D1 security review is not claimed. Alibaba OpenCodeReview preview was unavailable in this specific H38 terminal path; no paid API review was launched.
- Source stage CI must be verified at its **exact published head** after pushing. Neither this filter nor the H37 scorer validates trusted user/task provenance, genuine native click execution, or real UI performance.
