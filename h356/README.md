# H356 — independent sealed bilingual OOD custody intake

**Engineering-only DRAFT. No independent custodian is enrolled, no private
OOD source has been received, and no blind model scoring is authorized.**

Hala's H173 OOD source was **consumed** by a previous frozen, one-shot
research evaluation, and the later H351–H354 real-Qwen first-click studies
used already exposed first-party H319 BeaconBoard. They cannot be rebranded
as independent new blind evaluation. H199 has **zero** admitted TRAIN
manifest digests, and H343/H355 rights review still lacks an external
approved signer. The two public work requests are [#384 independent
training rights](https://github.com/TheHalfMoon/Hala/issues/384) and
[#385 independent OOD custodian](https://github.com/TheHalfMoon/Hala/issues/385).

## What a future separate evaluator can submit

H356 defines a read-only, exact-field, bounded, Ed25519-signed
*metadata-only* claim for a freshly selected benchmark in external
custody. It contains **only salted, opaque 64-hex commitments** to
candidate source families, sealed task set, sealed gold, source-image
rights evidence, independent custody/conflict evidence, source
non-overlap audit and preregistered scoring protocol. It declares
3–12 source families, 32–256 total tasks with at least 16 each for
Arabic and English, signature/expiration dates, and explicit
restrictions on disclosure, rights, patient data and training usage.
Never send raw tasks, screenshots, source filenames, application names,
gold points, reference state, model predictions, scores, API secrets or
case identifiers via this interface or a public GitHub comment.

An eligible independent custodian would first arrange private
rights-safe custody and reviewer identity **out of band**. An
independently approved public key and identity must then be enrolled in
a separate reviewed change. The candidate envelope **cannot enroll its
own key**. The production `TRUSTED_CUSTODIANS` registry is currently
immutable and EMPTY. H355's legal reviewer registry is separate, also
EMPTY; neither registry automatically trusts the other.

## Explicit limits of signatures and opaque hashes

The verifier can authenticate only that a previously independently
trusted key signed those exact metadata bytes. A SHA256 commitment
does **not** prove that a private dataset exists, was legally obtained,
differs from exposed examples, is still unseen, has no asset/template
overlap, or is truly under independent custody. A signed assertion
about independence is not independent evidence of it. These claims
must be audited by an actual qualified custodian under controlled
access against ALL consumed, TRAIN and DEV family/asset/behavior
sources. No private content may be shown to Hala's developer or tuned on.

**Even in a synthetic test where a valid pre-trusted external key is
temporarily simulated**, the successful return always says:
`actual_family_nonoverlap_proven=false`,
`rights_proven=false`, `blinded_ood_scoring_admitted=false`,
`new_training_authorized=false`, `model_gain_proven=false`,
`can_release=false`. No source/score/gold/commitment is echoed by
the verifier. A separate human-controlled, audited and irreversible
one-shot sealed evaluation protocol remains required.

## Engineering verification

```sh
python -m unittest tests.test_h356_sealed_ood_custody tests.test_h355_signed_rights_intake -q
```

Dedicated CI runs the verifier tests with pinned `cryptography==50.0.1`
on Windows/Ubuntu x Python3.11/3.12. Tests use invented ephemeral
**in-memory** Ed25519 keys, never real external keys, licenses, legal
decisions, hidden cases or benchmarks. Repository-wide stdlib-only
testing can import the test module without `cryptography`; signature
tests skip in that mode, while the dedicated workflow explicitly
asserts the signature library is available and executes all tests.

No GPU/LLM job, paid service, private artifact, data upload, OOD score,
training or model weight change is created by this PR. H173 remains
consumed, H199 TRAIN remains empty, H343/H355 human rights review
remains unsigned; Hala's release decision remains **NO-GO**.
