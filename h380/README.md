# H380 — Bind development trainer Mind/Type projection to frozen, verified bytes

**NO TRAIN, NO model weights, NO inference, NO proprietary/private media, NO
new rights or blind OOD admission.** Source-only hardening stacked on H379.

## Verified source-level flaw

The historical `training/development_data.py::load_rows` first called
`development.freeze.verify_bundle`, which hash-verified frozen
`settings.json` and `training-rows.jsonl`, but then reopened those files
with `read_text` to produce model-visible prompts and supervised
Mind/Type targets. A file replacement between the verification and second
read could therefore project **different bytes from those verified by the
externally pinned manifest**. H349 previously fixed a similar image
read/verify race in `encode_row`; it did not bind these two text files.

A hash check once before a later reread is not a safe training-input
provenance boundary. This is a *potential local artifact substitution*
when the old development trainer becomes eligible; it is **not** a report
of an observed attack, a legally admitted dataset, or an actual model run.

## Independent frozen manifest SHA-to-parse TOCTOU

Auditing the upstream verifier also revealed `development.freeze.verify_bundle`
computed the externally pinned hash of `manifest.json` using one `read_bytes`
call and **then separately reopened** the same path with `read_text`
to obtain the manifest fields. A substituted manifest between those reads
could make the returned source/file claims differ from the externally
pinned bytes, undermining even the downstream snapshot hashes.

The original development/freeze.py is independently byte-frozen:
the inherited CPU source-identity tests correctly refused a direct edit to
that verifier. H380 therefore leaves that original source **byte-for-byte
unchanged** and instead adds a TRAIN-SIDE _verified_frozen_manifest guard:
it reads a bounded snapshot of manifest.json, checks its external SHA-256,
parses that exact snapshot with strict JSON, then invokes the historical
verifier and rejects any returned metadata differing from the already
pinned snapshot. No edited historical reference implementation is needed.

An invented regression test simulates the old verifier returning forged
metadata after a local file swap; the training-side guard must deny it.
Wrong or absent external digests fail before invoking the verifier. This
protects the historical training projection, not every other caller of
the immutable freeze verifier or any legal signing authority.

## Minimal source-only change

After the new exact-manifest identity guard and original full `verify_bundle` check, H380 obtains each
registered projection file as a **single bounded byte snapshot** (16 MiB
maximum), refuses symlinked projection files and compares SHA-256 of
those exact bytes with the hashes in the already independently qualified
manifest. Only these verified in-memory bytes are decoded and projected.
`strict_json` rejects duplicate JSON object keys and nonstandard
constants instead of the permissive legacy `json.loads` reparse.
If the file changes between freeze verification and projection snapshot,
its hash differs and the future training preflight refuses; changes
*after* the snapshot cannot change the projected model-visible text.

Original frozen manifest admission, verified source/media, family splits,
English/Arabic co-primary selection, model settings and H350 legal
TRAIN admission check are unchanged. No source media, weight, GPU,
optimizer, output or actual train process was opened.

## Tests

Invented-only in-memory bilingual Mind/Type rows and temporary files
prove exact valid hash-bound projection, forged training row substitution
after the verifier returns, changed settings after verification,
duplicate keys, oversize projection, malformed manifest digest, unknown
filenames and symlink refusal where supported. Additional invented tests check
manifest substitution between SHA verification and JSON parsing, and a
forged external SHA. Dedicated hosted
Windows/Ubuntu Python3.11/3.12 source-only CI also checks inherited
H350 legal-admission fail-closed behavior.

No independent legal rights have been granted. H199 lawful admitted
TRAIN manifest digest registry EMPTY, H355 independently qualified
rights reviewer keys EMPTY (Issue #384 OPEN), H356 independently
custodied truly fresh bilingual EN/AR family-disjoint sealed blind
OOD custodian keys EMPTY (Issue #385 OPEN). This security guard does
not make old Mind/Type data eligible to train, does not validate adapted
Qwen weights, and cannot authorize release. Keep DRAFT, no main merge.
