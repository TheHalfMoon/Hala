# H355 — Authenticated independent screenshot-training rights decision intake

**STATUS: VERIFIER IMPLEMENTED; NO TRUSTED REVIEWER REGISTERED;
NO ACTUAL EXTERNAL RIGHTS APPROVAL. H199 TRAIN ADMISSION STILL EMPTY.**

H343 already packages the precise public-original source identities
of RailKit and SignalDesk into a bounded content-hashed set of
eight questions for a qualified, independent external reviewer. An
H343 request is **not** a grant. H355 provides a *separate* offline,
fail-closed Ed25519 detached-style signed-decision verifier that can
authenticate a future human review's stated scope **only after an
independently approved and code-reviewed public-key trust anchor is
enrolled**.

## Threat model and actual limitations

- A source author, a maintainer or any person with repository access
  may fabricate an approval-looking JSON, a fake author name, a
  public key and a genuine Ed25519 signature of their own. **That is
  self-approval, not external rights review**. H355's shipped
  `TRUSTED_REVIEWERS` is an **immutable empty registry**; the
  envelope cannot enroll its own public key. All external reviews
  are refused until a separately reviewed, evidence-backed change
  registers an independently confirmed reviewer key and exact
  independent reviewer identity.
- A valid signature authenticates byte-level authorship by a
  pre-trusted key, **not copyright title, license sufficiency,
  originality, independent employment, consent, or an external
  judge's professionalism**. An independent human must review
  contributor/title history, raw agreements, third-party assets,
  scope and expiry and custody before deciding whether any
  source may ever be admitted.
- The signed payload ties the decision to the **exact SHA256
  H343 request** (which binds both original HTML source Git blobs
  and introduction commits), independently custodied review
  evidence SHA256s, reviewer identity, signed/expiry dates,
  explicit rendered screenshot training rights, derivative
  model/weight distribution, commercial scope, title and
  asset-exclusion findings for **both families**.
- The verifier rejects fake key IDs, expired/future decisions,
  source swaps, changed signatures, noncanonical Unicode/JSON
  duplicate keys, unknown fields, unrecognized claims, invalid
  permission types, invented extra family and missing family.
- Even a genuine approved independent decision with full scope
  would return `official_h199_train_admitted: false`,
  `can_train: false` and `can_release: false`. A **separate**
  human-controlled code-reviewed H199 TRAIN manifest admission,
  family-disjoint DEV, and genuinely new independent blinded OOD
  are needed. Nothing in H355 alters H199's empty registry.

## Independent reviewer handoff (no private inputs in repo)

1. Start from the canonical H343 public request and verify original
   exact Git source bytes, contributor history, full license
   obligations and ML-rendered-image rights with documentary
   evidence. Refuse if independently verifiable title is missing.
2. Use an independent signing identity with a preverified,
   independently custodied Ed25519 public key. The organization
   must approve this identity/key out of band; do not commit
   a private key or enrollment request as proof.
3. Create an exact `hala.external-screenshot-rights-review.v1`
   payload with request SHA, evidence custody SHA, independent
   reviewer evidence SHA, dates, exactly two original family
   source hashes and grants, and explicit training/derivative
   weight rights. Sign the canonical sorted-ASCII JSON **payload
   bytes** with Ed25519; construct an envelope with `schema`,
   `key_id` (SHA256 of the 32-byte public key), `payload`,
   and `signature_b64`. No unsupported fields.
4. Before H355 can verify any positive review, the externally
   validated public key and exact reviewer identity need a
   **separate independently reviewed trust-anchor PR**. Re-run
   source checks with the independently signed decision from
   the custodian, never project-authored test keys. Keep any
   raw agreements out of the public repository and use only
   SHA256 metadata, with explicit privacy approvals.
5. Passing H355's signature check is only **authenticated review
   intake**. A subsequent official H199 TRAIN admission
   requires independent legal and split-qualification approval
   and must remain an independent code-reviewed decision.

## CPU qualification

```sh
python -m unittest tests.test_h355_signed_rights_intake tests.test_h343_independent_rights_review_request -v
```

The eight H355 tests generate **ephemeral invented keys in memory**
and patch an invented test-only trust registry to simulate a
cryptographically well-formed review; their positive path still
**always refuses actual training**. No example private key, signed
approval, fake decision JSON, screenshots, model outputs or gold
is committed. Dedicated GitHub CI checks the frozen H343/H355
flow on Windows and Ubuntu with Python 3.11/3.12.

## Scientific NO-GO

The latest H354 real-Qwen owner-private observed result remains
EN 6/6 unchanged, AR 4/6 to 6/6 on already exposed original
BeaconBoard; only two of twelve actions changed. This is
screenshot-only first-click *postprocessing*, **not learned
training** or a source-family-disjoint blinded benchmark. The
H343 external review PR #371 had zero independent reviews as of
this H355 work. H173 sealed, H212 barred, H250/H251 quarantined,
H293/H294 blocked; no release, model-weight publication,
patient-data use, paid training, or main branch merge.
