# Actual held-out H21 baseline review

PASS for recorded integrity; model outcome failed all eight tasks. Independently reran the standard-library audit: **1084 checks, zero inconsistencies**. Verified all four actual PNG byte hashes/dimensions, state-derived filenames and global/local call bindings. All 78 global calls bind: ten standalone Mind and 68 episode calls. Every final state retains save revision 0; success is English 0/4 and Arabic 0/4. Episode Mind semantic accuracy is 0/64; standalone Mind exact correctness is 0/10.

The four logged `unsafe_proposals` are strict protocol rejections: all four Type outputs were fenced JSON rather than required bare JSON. They caused no state mutation or native execution. This counter is not four demonstrated harmful actions or a calibrated safety score. English episodes made no Type calls; Arabic made four rejected Type calls. Neither language exercised a successful save or recovery.

The standalone Mind cases/prompts exactly repeat DEV's same ten authored cases. Across DEV and TEST there are **ten unique standalone cases**, not twenty independent observations. Application-family separation does not remove shared workflow/template dependence. No test-driven source repair, tuning or schedule change was performed or recommended. The previously declared fixed 116-step pilot must remain selected prospectively; its training gate uses DEV deficits only.

Result bytes match external SHA256 `112d7c09a5d6cabbf68d8d74518c98b1f5d53d114ddc9c58dd5cf65c59bb2ad4`. The 65,499-byte ZIP matches `a0f2efee7ee54041439fe7b9e6c20b7649be48b3d9d368edd72df8bfeee69f9f`; all six ZIP files match extracted bytes. Baseline JSON and ZIP also match local Git objects at retained archive commit `c117f1d30e43aadcb4e9842f421c9029141a584e`.

Compared runtime plus all eight development source files to Git objects at recorded source `6b9bdff139d57ca29d0e4c5774c2d71805e8188e`. All match; runtime SHA256 is `da6622d6c5c5e32724d4caa8b523e4d3c9186984f5dff63e32cbf13b7d3a0a35`.

Reported Tesla T4 telemetry: wall 261.79s, model generation 232.54s, load 20.08s; peak allocated 4,513,904,128 bytes, peak reserved 4,575,985,664 bytes and maximum sampled full-device usage 4,744,675,328 bytes. These are recorded process samples, not independent hardware or production-cost measurements.

Limits: no hardware/honest-run or signature attestation; PNG bytes/signature/dimensions checked without new Arabic visual QA; no native/generalization/quality qualification. No ML/GPU execution or repository source edits by reviewer. Machine-readable checks and frame hashes: `work/h21-base-test-actual-review.json`; rerun audit: `work/h21-base-test-review-audit.json`.
