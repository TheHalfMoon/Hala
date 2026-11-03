# H382 — One-shot ShiftDesk source candidate handoff (NO TRAIN)

**Original ShiftDesk 24-scene source-only contract hardening.** This does not
admit legal TRAIN, DEV, or OOD data, train a model, open private media, or
publish screenshots, labels, model weights or external reviewer material.
Stacked on fully qualified H381 Draft PR #411.

## Source-backed failure mode

H285's original `produce_train_only_batch` rigorously checks all 24
first-party browser scenes and unique PNG/RGBA hashes **before** invoking
its downstream callback. However, its legacy `sink(png, task, label)` is
then called 24 times. If a caller persists data and fails on the third
callback, it may have retained three rows even though H285's function
failed and never returned a successful receipt. Complete validation does
not guarantee atomic downstream consumption, and an incomplete candidate
set must never be treated as a complete TRAIN release.

H325 already has a separate opt-in single-batch handoff for its two-step
source candidates. H382 introduces the analogous H285 option while
preserving the original behavior and test compatibility.

## Changes

- Optional `trusted_batch_sink` receives **one tuple of exactly 24**
  immutable `(PNG bytes, task str, native action label str)` tuples
  only after all 24 source scenarios, screenshot hash distinctness,
  native state transitions and other original guards complete.
- The new callback must return the **literal boolean True**; false,
  numeric 1, None or nonboolean values fail closed. Exceptions propagate
  without retry.
- Exactly one sink mode must be provided: legacy per-row `sink` or
  opt-in `trusted_batch_sink`. Both and neither are rejected.
- A bounded non-media receipt indicates `candidate_handoff_mode`
  (one_shot or legacy_per_row). It does not imply data admission.
- The original 24 source scenes, source file Git blob, offline network
  isolation, browser-first-render H381, H322 screenshot recovery,
  label parser, PNG/RGBA dedup, role, source contract and legal
  admissions remain unchanged.

**Important:** a single callback is not a storage transaction. Only a
trusted consumer with its own staging, commit/rollback and external
rights approval can enforce storage atomicity; neither that storage
transaction nor legally admitted training is implemented here. The
legacy per-row interface remains non-atomic by design for backward
compatibility.

## Tests and evidence

Invented-only tests verify complete 24-tuple one-shot delivery, complete
EN/AR counts, refusal before any consumer call on a failed scene or
duplicate SHA, literal True acknowledgment, no retry on callback errors,
exclusive sink choice, preserved legacy 24-row success and demonstrated
partial legacy deliveries if the consumer aborts. They do not open a
browser or generate PNGs/coordinates. Dedicated 4-way GitHub
Windows/Ubuntu Python3.11/3.12 stdlib source-only tests inherit H285
and H381 contracts; separate existing H285 real Ubuntu Chromium/Edge
CI continues independently. Check exact-head status before claiming
qualification; do not selectively rerun a failing CI for a badge.

## Model release is still blocked

H199 official lawful admitted TRAIN SHA registry EMPTY, H355 independent
screenshot/derivative model rights reviewer trust registry EMPTY,
GitHub Issue #384 OPEN; H356 independent novel never-exposed
source-family-disjoint sealed bilingual EN/AR blind OOD custodian trust
registry EMPTY, GitHub Issue #385 OPEN. H285's synthetic 24 public source
candidates are EXPOSED and not lawful TRAIN, sealed blind OOD or
measured model learning. Keep the PR DRAFT; do not merge main, deploy,
train or claim adapted-vs-base gains.
