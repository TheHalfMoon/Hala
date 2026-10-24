# H80 — Last-chance checkpoint salvage on a live T4 monitor deadline

**Stacked source:** H79 Draft PR #63, SHA `c3d837e126172e963480823ef85cd8a72b3d66c2`.
**Scope:** host-only bounded-attempt operational recovery. No new GPU, model weights, optimizer update, actual private upload or training qualification.

## Problem

The H65 host supervisor checked the checkpoint during each loop, then checked whether a previously allocated free Colab T4 remained present. If the time limit elapsed it terminated its operator child and **returned immediately**. A checkpoint atomically published between the previous periodic host egress and child termination could be lost even if the T4 was still reachable. The premature return also skipped H65's normal explicit `final` receipt and returned a reduced report lacking private-readback/completion fields.

## Change

After stopping the timed-out child, when the provider's **most recent probe was confirmed T4**, try **one final read-only H65 `snapshot_once`** into the existing SHA-verified host staging. No retry loop, no new session, no paid compute. The existing egress component enforces remote paths, artifact SHA256s and per-download timeout/byte bounds. Its total wall-clock time can exceed the monitor deadline when an artifact is large: H80 limits number of extra attempts to one, **not** overall elapsed time.

When the T4 was lost (three missing probes), do **not** attempt another provider download; report `SKIPPED_PROVIDER_MISSING`. On a still-present deadline, report `ATTEMPTED_ONCE`, the egress status and step or the redacted exception class. Both early-return branches now append a canonical `final` receipt and maintain `FAILED` with reason `monitor_deadline` or `provider_session_lost`, `completed_checkpoint_evidence: null` and `private_provider_readback_verified: false`, irrespective of recovered files.

Crucially, a verified final partial host snapshot can then enter existing **explicitly opted-in H77 private backup**, without rewriting the failed experiment into a successful completed training run. No changes to H78's completed-step proof, H79 source admission, H71 private datasets or licensing/compute eligibility.

## Verification

Six synthetic no-network CPU tests exercise:
- Two snapshot calls at deadline, with a new complete-format SHA-verified checkpoint appearing **only on the final one**.
- The deadline resulting in **FAILED** despite recovery, and an appended canonical final receipt.
- No false positive when remote latest metadata is unavailable.
- Redacted, classified final-download failure while preserving failed status.
- Three missing provider probes skip all additional remote calls.
- End-to-end H65 stop → late salvage → H77 mocked authenticated private readback **without changing FAILED status or deleting host copy**.

These are **synthetic CPU test scenarios**, not verified real Colab checkpoints or real Kaggle private-storage readbacks. First actual zero-cost free T4 LoRA step, credentialed provider backup, and held-out bilingual GUI quality measurements remain blocked/unmeasured until an authorized genuinely free allocation is available.
