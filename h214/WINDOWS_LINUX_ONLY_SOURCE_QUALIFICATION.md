# H214 — owner-approved Windows/Linux-only GUI research qualification

**DRAFT ONLY. No merge, release, publication, paid compute, owner-private data, model training or OOD evaluation.**

## Constraint repaired

H213 PR #234 introduced a six-job hosted matrix including macOS, and its historical description also records an earlier owner Mac experiment. That work is **not authorized as a basis for any further Hala execution** under the current Windows-only owner-device policy. This stacked H214 change does not erase that history or misrepresent its provenance; it stops additional Mac jobs and limits the future hosted source-only portability matrix to Linux and Windows.

The previously queued H213 hosted run `40964664463` at exact head `d5c2e9d6f45d666f64ad5d66b4a0407873452193` was submitted for cancellation on 2026-10-30 before any of its six jobs started. Cancellation confirmation and final GitHub status must be verified independently. No macOS runner is requested or used by H214.

## New enforceable source controls

- `.github/workflows/h213-cross-platform-frozen-source-cpu.yml` uses exactly `ubuntu-24.04` and `windows-2025`, Python 3.11/3.12, pinned Pillow 12.3.0 and no model weights.
- `tests/test_h214_no_mac_ci_policy.py` rejects unexpected host entries, extra runner jobs, missing Windows/Linux, duplicated matrix, unpinned Pillow, and missing optimized checks. It runs under both normal Python and `python -O`.
- All 96 self-authored H212 research PNG and decoded RGBA fingerprints are regenerated and compared to frozen source-only metadata on those approved hosted systems. This is a portability check, **not** native owner Edge qualification or a source-disjoint data admission.
- Owner Windows Abdulaziz is the **only authorized owner computer**. Hosted Linux and Windows are public source-only, zero-cost qualification runners; neither may access owner-private files or historical gold.

## Science firewall

All 96 images remain `RESEARCH_*_NOT_ADMITTED`. Official TRAIN manifest stays empty; DEV is not sealed and no new OOD has been evaluated. Historical source/fork and decoded RGBA completeness is still unproven. H173 v3 and all earlier consumed DEV/OOD remain permanently sealed, with no rerun or tuning. No optimizer steps, GPU or new model inference are permitted by this change.

Before claiming qualification, check the exact H214 Draft head, successful Windows/Linux CI jobs, clean owner Windows Git state and protected `main`. Do not interpret a queued, failed, cancelled or superseded workflow as PASS.
