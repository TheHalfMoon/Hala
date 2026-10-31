# H302 — decoded model-visible pixel freshness lock

## Defect and scope

H298's two-stage policy previously identified a fresh screenshot using
`sha256(png)` over compressed PNG **file bytes**. A second PNG with different
ancillary metadata or compression but identical visible pixels would have a
different byte digest and could be accepted as a new model observation.
This is a concrete protocol-integrity weakness, not a measured Qwen failure.

H302 instead fully decodes the bounded original 960×720 PNG and hashes
the **RGB pixels actually supplied to H208's visual processor**. The
second model-time step is refused and the existing H301 terminal latch is
set if its RGB pixels match any earlier model-visible screenshot. PNGs
with invalid pixel content or a changed viewport fail before model.generate.
No screenshots, hashes, coordinates, gold, or model text are printed,
persisted, or placed in the public receipt.

The separate H296 pre-action screenshot lock still checks that the browser
does not change **during** each inference; H302 addresses a different
**between-step** freshness identity. Neither check is a proof of atomic
browser action or actual model identity.

## Evidence boundary

- New invented-only tests compare two PNG encodings with identical RGB
  pixels but different ancillary metadata and verify terminal refusal
  without a second model call.
- A visually changed invented screenshot still permits the strict second
  `type_text` action; malformed/wrong-size PNGs refuse before inference.
- Existing H298/H301 fake-VLM tests use newly generated **invented in-memory
  valid PNGs** rather than non-decodable header stubs.
- Hosted source-only Linux Chromium and Windows Edge browser fixture
  qualification is required on the exact final commit; those fixture
  actions **are not Qwen inference**.
- The previous real pretrained H300 FieldPort **12/12 on exposed original
  first-party research** remains historical evidence. H302 has **not**
  been rerun with genuine Qwen, does **not** attest its own neural task
  success, and cannot retroactively change the H300 pinned-source result.
- H291/H292 adapted-vs-base comparisons still show **zero** learned gain.
  H199 official TRAIN admission is empty. No independent blind OOD,
  new GPU experiment, model release, or generalizable gain is claimed.

All source and invented tests are public on this stacked **DRAFT** PR.
No protected-main changes, weights, private media, paid compute, or
previously consumed scored OOD.
