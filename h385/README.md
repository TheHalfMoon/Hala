# H385 — auditable exact source-row snapshot for native Qwen processor

**H359/H361 public-original screenshot REVIEW and processor-only qualification;
NOT legally admitted TRAIN, blind OOD, model inference, learned LoRA or release.**

## Actual source-backed trust-boundary weakness

H361 `_candidate_rows(root,family)` previously invoked the complete
H359 `audit_review_bundle` on 64 H357 ControlGarden or 128 H358
NoteRelay public screenshot/action rows, THEN reopened the
`candidate-rows.jsonl` file. H359 checked the SHA-256 of each referenced
PNG, bilingual counts, H199 native action syntax, duplicate/visually
identical frames, rights-unadmitted flags and exact original source
identity. But after H359 returned, a local substitution of the row file
could change a *model-visible task or target* without changing row count,
record shape, or the row's original PNG identity. H361 independently
checked some selected row fields, not the original H359 audit snapshot.
This left a genuine local artifact time-of-check/time-of-use boundary
just before construction of real pinned Qwen processor inputs.

Owner Windows reproduced this on the **unfixed** original H361:
an invented-only test mocked the source audit after taking the original
row-file snapshot, replaced the file with the SAME-LENGTH 64-row JSONL
containing a different but valid task, then H361 accepted the swapped
rows. The unfixed 4-test suite had **8 expected-refusal failures**
(including missing/invalid audit digest), with no browser or GPU.
This is a synthetic attack proof, not evidence any actual owner
source file was compromised.

## Minimal fail-closed correction

- H359 now computes `audited_rows_sha256` of the **same bounded, single
  raw JSONL byte snapshot that it parsed and audited**. It returns this
  64-character lowercase digest as a content-identity receipt, not a
  cryptographic *signature*, trusted third-party approval or source
  media/label export. Existing per-row SHA/decoded RGBA checks remain.
- H361 now demands that exact digest and verifies the SHA-256 of ONE
  independently bounded (max 500 KiB) `candidate-rows.jsonl` byte
  snapshot before parsing that same in-memory snapshot for the model
  processor. Any changed/truncated/oversized row file, missing/malformed
  digest, nonmatching SHA or unsafe candidate path fails closed before
  opening a PIL image or invoking the supplied processor.
- The original public HTML source blobs, rendered PNG bytes, candidate
  roles/splits, bilingual/action counts, H199 one-step parser, assistant
  masking, processor pin SHA, native image tensors and existing no-TRAIN
  or no-OOD flags are unchanged.

## Tests

Four new invented-only tests prove:
1. An attacker replaces all 64 task rows after audit with an equally
   sized syntactically valid file: **no processor rows are accepted**.
2. Untouched byte-identical audited rows are accepted as source-only
   processor candidates, without TRAIN rights.
3. False/None/uppercase/nonhex/bool/numeric digest refuses.
4. A report without an audited digest refuses.

Owner Windows at source-fix commit `295efb7` ran H385 + inherited
H359/H361 source-only stdlib tests on Python3.11 and 3.12:
**9 total, 7 PASS, 2 intentional real-browser opt-in SKIP per Python**.
The pre-fix negative tests failed as expected, establishing a
red-to-green behavior change. Full real H357/H358 source browser and
simulated processor on both operating systems is required before
the final qualification verdict. The existing registered 4-way
H361 source+fake-processor GitHub workflow is extended to execute
H385 tests too; do not substitute a fake for the separately pinned
real Qwen3VLProcessor CPU evidence. No screenshots/gold, models
or tokens should be persisted by these tests.

## Model NO-GO remains unchanged

The 192 public H357/H358 source candidate rows are openly EXPOSED
and cannot qualify lawful official TRAIN or fresh sealed blinded OOD.
H199 official admitted TRAIN SHA registry EMPTY; H355 independent
screenshot/title/derivative model weights rights reviewer keys EMPTY,
Issue #384 OPEN; H356 new never-exposed family-disjoint independently
sealed bilingual EN/AR blind OOD custodian keys EMPTY, Issue #385 OPEN.
Those requirements cannot be self-signed or replaced with more source
tests. No training, learned adapted-vs-base gain or weight release.
Keep H385 as Draft, leave protected main unchanged.
