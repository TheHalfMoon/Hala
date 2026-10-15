# H15 combined delta review 1

Scope: loss-trajectory repair only, against the original combined review. Reviewed current `training/scripts/qualify_lora.py` SHA256 `8fee099a627fdafd05d737d188288ae05aa37f69e9b273447de0318bef812523` and `tests/test_lora_qualification.py` SHA256 `caeead2f9e138c82b1775a16148577c72417da43ef9a2e2cf8b6864475821e32`. No source edits or GPU execution by this judge.

- **Must-fix: none.** Original loss-comparison finding is resolved at `training/scripts/qualify_lora.py:104-106`: complete ordered step/loss rows enter the same comparison map whose aggregate `within_tolerance` controls `runtime_qualified`; floating loss uses existing 1e-6 absolute / 1e-5 relative tolerance. Elapsed seconds remain observed evidence.
- **Correctness validation:** `tests/test_lora_qualification.py:13-26` invokes the actual comparison hook with equal adapter/saved state and a changed second loss. The mismatch fails aggregate qualification comparison; equal losses with changed elapsed time pass. Independently reran all 7 qualification control tests successfully with bundled Python 3.12.14. Parent-supplied RED history was not independently replayed.
- **Parsimony/product:** direct reuse of the existing comparison helper closes the acceptance gap with no new abstraction or scope. No new findings.
- **Worth-considering:** prior optional interruption-publication regression-test improvement remains unchanged; source ordering was already correct and is outside this delta.

This review qualifies the narrow source repair only. Actual GPU/model execution and fresh full-suite/Jev/Graft gates remain the parent's work.

panel: delta ✓ independent Codex combined correctness/parsimony/product · security — separate judge, no new deserialization delta · GPU qualification — pending
