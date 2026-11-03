# H384 — pixel-distinct review-source audit, no TRAIN admission

**H359 H199-native two-family public synthetic review ONLY.**
No real trained Qwen/LoRA, no private source media, no patient data,
no independent legal rights or sealed novel blind OOD.

## Reproduced source-backed defect

The earlier H359 `audit_review_bundle` counted 64 ControlGarden
and 128 NoteRelay source PNGs as `distinct_images` based solely on
file SHA-256 and exact PNG width/height/CRC. A PNG can have a distinct
file hash because of an ancillary `tEXt` metadata field while decoding
to **exactly the same 960×720 RGBA pixels**. No decoded-frame
distinctness check existed. The review report therefore could
overstate the number of visually independent screenshot candidates
without violating any earlier checks. This does not establish that
the *real* H357/H358 source bundles contain such duplicates; it
establishes an acceptance vulnerability in their reviewer.

On owner Windows with an exact unmodified H359 bridge, invented-only
`tests/test_h384_native_review_decoded_pixel_distinctness.py`
reproduced the issue: 64 byte-distinct PNGs with unique metadata
but identical raster were incorrectly accepted (test FAILED because
no refusal occurred). The companion test needed a decoded-frame
count absent from the prior report (KeyError). This is direct
counterexample evidence, not a claim of actual data compromise.

## Minimal fail-closed correction

After existing strict source file hash, sample identity, manifest
unadmitted-role, native completion, prompt-to-task and PNG-byte
SHA256 checks, H384 **retains original `Pillow.Image.verify` CRC/PNG
structure check**, and then opens the SAME SHA-matched in-memory
PNG bytes a second time to decode into 960×720 RGBA pixels. SHA256
of the exact decoded RGBA bytes must be unique across ALL reviewed
candidate images for the same source family, regardless of metadata
or compression encodings. No second filesystem read or TOCTOU gap
is introduced.

The original `distinct_images` byte-SHA field remains unchanged,
and a separate `distinct_decoded_rgba_images` count makes the
review report's pixel-level identity explicit. A source bundle
which meets file-level identity but duplicates decoded pixels now
fails closed before emitting even its aggregate report. Existing
exposed-not-admitted rights, roles, exact original HTML SHA,
192-row language/action-count guards and source-only message
projection are unchanged.

## Scope and tests

- Invented-only synthetic CPU test 1 creates **64 individually
  SHA-distinct compressed PNGs** with different PNG `tEXt` metadata
  but identical actual decoded pixels; H359 must refuse.
- Invented-only synthetic CPU test 2 creates 64 genuinely distinct
  decoded RGB/RGBA frames in equal EN/AR counts; H359 must accept
  them as **candidate-only**, report 64 distinct decoded frames
  and preserve all false legal TRAIN/rights/blind OOD/release flags.
- Owner Windows exact repaired-source commit `23e3608` source-only
  two new + existing H359 contract tests: 5 tests total, **4 PASS,
  1 existing real-browser opt-in intentionally SKIPPED**, 0 fail.
- New hosted H384 4-way Windows/Ubuntu × Python3.11/3.12
  CPU-only Pillow-pinned tests. Existing H359 dedicated original
  browser CI is ALSO triggered by this `h359/` source change;
  it must independently re-evaluate all **192 actual original
  screenshot rows** in both original families. Do not claim
  full qualification until exact HEAD hosted source+browser jobs
  finish. Original browser source tests must not be silently
  excluded when authorizing the final branch status.

This is a visual candidate-source integrity guard, NOT a measure
of visually independent task difficulty or model generalization.
Pixel-unique source screenshots can still be near-duplicates,
share one source family, lack rights or be exposed to training.

## Hala model remains NO-GO

H199 official independently lawfully admitted TRAIN SHA allowlist
EMPTY. H355 independent screenshot/title and derivative weights
rights-review trust keys EMPTY (Issue #384 OPEN). H356 independently
custodied genuinely NEW, never-exposed source-family-disjoint,
sealed bilingual EN/AR blind OOD custodian keys EMPTY (Issue #385
OPEN). Public H357/H358 192 source candidate rows remain EXPOSED,
not legally qualified TRAIN or new blind OOD, and do not prove
an adapted-vs-base Qwen LoRA gain. Keep the PR DRAFT; do not merge
`main`, release weights, train or publish private pixels/labels.
