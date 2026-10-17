# H21f independent combined review

Verdict: no source correctness, parsimony, or product blocker found. Source-only review; operational Kaggle/GPU and quality qualification remain pending.

## Must-fix

None found.

## Worth-considering

- `specs/H21f-kaggle-evaluation.md:9` and `docs/adr/0005-qualify-kaggle-evaluation-separately.md:6`: phone verification is described as currently pending, while the supplied current user state says it is completed. Change to “user reports phone verification completed; last observed GPU UI remained disabled and actual runtime eligibility is unconfirmed.” Keep runtime/GPU qualification pending.

## Skip-it

- Do not relax package/font/CUDA/cache or inference contracts merely to support Kaggle. No evidence in this diff justifies doing so.

## Evidence

- Reviewed uncommitted provider/runtime/tests/spec/ADR against `5b6eb4bf29acbaf21aa199f113f78af4f268b3b7`.
- Focused bundled-Python run: `-m unittest tests.test_development_provider tests.test_development_adapter -q`: 12 tests passed in 1.257 seconds. No ML/GPU/model downloads.
- Independent AST comparison: SETTINGS, complete QwenInfer, original training remote_guard, MODEL_ID, REVISION and VERSIONS exactly unchanged. Baseline source SHA256 confirmed `da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35`.
- Invalid providers/platform markers/path/mode/opt-in fail before ML imports for baseline and candidate. CLI defaults to Colab; Kaggle requires explicit selection. Candidate external identity validation remains ahead of ML. Both result paths retain provider evidence, task callbacks, independent scores and nonqualification flags.
- Only runtime admission/provenance and test seams changed; no training/frozen-input/package-pin/model-setting change observed. Independent security/Jev/Graft gates belong to root and were not substituted by this review.
