# H25 source follow-up — hardening and Kaggle notebook preparation

Date: 2026-10-18. **Hala repository only; no Orcel access or edits.**

## Actual defect fixed

Before correction, `h25.baseline.summarize()` recomputed typed proposal correctness but did **not** match the `error` category or require a bounded `error_type` on missing raw output. This could produce an internally inconsistent interim report, even though the separate external `tools.verify_h25_dev` later rejected the mismatch. The internal scorer now checks both, ensures raw replies are bounded strings and rejects fabricated inference-error metadata attached to actual responses.

Added focused regression tests for mislabelled inference errors, missing failure classification, oversize error classes and JSON-with-invalid-contract (a parsing failure, not an inference exception). **No model output was actually generated** by these tests: they use fabricated oracle fixtures solely to validate evaluation logic.

## Private Kaggle preparation, not an actual GPU trial

- New `tools/build_h25_kaggle.py` generates a prospective **private, new-file-only** Kaggle Save & Run All notebook and metadata pinned to an exact 40-character source commit.
- The notebook can fetch weights **only remotely**, after verifying Kaggle Batch runtime, exact reviewed branch SHA, clean checkout and exact pinned package versions. Every key guard is an explicit error, not an optimizable `assert` statement.
- A distinct prospective notebook ID is used; no H24 or H25 TEST data, no adapter training, no native Orcel GUI action and no direct model-output execution. Nothing is sent to Kaggle or a GPU by these source-only changes.
- Five new tests exercise private/GPU metadata, exact source pin, read-only generator behavior, no overwrite and no test/oracle export.

## Review evidence, properly bounded

- **375 complete Python tests, 374 passed, one skipped**, including the unchanged historical H19/H22/H24 and H25 freeze tests.
- Genuine Graft v0.21.1 `build` and `check` returned **exit 0**, 1,211 source-wiring nodes across 128 Python files, and reported `graph check: OK`. Its deep semantic graph was **not built**.
- Genuine Jev typed advisory on 8 named source/test/docs completed `ok:true`; see `docs/reviews/h25-baseline-jev.json`. This is advisory probability scoring, not independent audit/release approval.
- **NOT completed:** independent fresh-context PStack, bilingual Arabic/English professional semantics adjudication, Alibaba independently model-driven Open Code Review, live free Kaggle experiment, true held-out multi-step agent benchmark, H23 real CUDA cancel/restart, HalaType RAQM/screenshots or Orcel native authorization checks.

**Disposition:** Update the existing **Draft Hala PR #8**, signed+DCO, verify its exact-head GitHub CI. Preserve all original H25/H24 and H22 source/result bytes; no main merge, training, paid compute or model adoption. Private notebook generation after CI is allowed for inspection, **but do not submit a Kaggle GPU job before the missing source/bilingual governance gates are closed**.
