# H400 — optional research GPU dependencies without compromising source-only CI

H398 Kaggle's private synthetic T4 CNN trained successfully; H399's private Kaggle pinned Apache-2.0 Qwen3-VL-2B synthetic LoRA performed 16 real optimizer steps with nonzero adapter tensors. Neither model or benchmark qualifies the Hala flagship or original GUI training rights.

**Reproduced CI blocker:** H398 added two `unittest` checks that directly import PyTorch, but the general `cpu-qualification` workflow intentionally does not install PyTorch. Exact H398 PR CPU run #41718878558 failed with `ModuleNotFoundError: No module named 'torch'` in those two tests; other thousands of CPU tests ran without model weights. This is a mismatch of optional research tests to the ordinary stdlib-only CI, not a Kaggle training failure.

Tests that require optional `torch` (H398) or optional `PIL` (H399) now use explicit `@unittest.skipUnless(importlib.util.find_spec(...))`. They remain **fully active** in the pinned owner Windows Python with packages installed. The receipt source-integrity and private Kaggle kernel metadata checks remain **mandatory** pure-stdlib tests for every CI host.

**Verified before review:** A real Windows stdlib-only Python execution `python -B -S -m unittest tests.test_h398_original_synthetic_gpu tests.test_h399_original_synthetic_model tests.test_h399_receipt_integrity -v` passed **7 tests, 3 correctly skipped**, without importing any site package. Full Windows owner runtime with PyTorch/Pillow available passed **7/7 tests**, zero skips. No package installation or permission/rights workaround is required for hosted CI.

H199 independent screenshot rights allowlist remains EMPTY, issues #384 (independent rights signoff) and #385 (never-exposed independent bilingual OOD) remain open. No protected main merge, no model weight release.
