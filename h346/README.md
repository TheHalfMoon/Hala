# H346 — bind legacy Qwen GUI trainer to H199 official TRAIN admission

**Status:** security/custody enforcement only. No GPU run, adapter training, model improvement or release.

## Verified gap and remedy

The legacy `training/scripts/train_adapter.py` previously accepted a training
manifest when its own JSON asserted `rights_cleared: true` and source families
did not overlap a self-declared held-out list. Its runtime path did **not**
consult H199's official approved TRAIN manifest digest registry.
H199 currently contains `ADMITTED_TRAIN_MANIFEST_SHAS = frozenset()`.

H346 now checks the SHA-256 of the **same bounded raw manifest bytes** later
parsed by the trainer. The exact digest must be in H199's official allowlist.
Check happens before baseline reading, training-record loading, model weights,
processor allocation, optimizer steps, output creation and any GPU probe.
Symlinked manifest paths are denied. Merely setting an opt-in environment
variable, inventing a `rights_cleared` flag, or supplying a self-declared
`official_admitted` field cannot grant admission. A stale or changed manifest
does not match the registered digest.

Training can only become possible after a separately reviewed, source-bound
rights/title/screenshot-ML grant and official manifest admission through the
existing H199 governance and split controls. H346 does **not** add any digest
to that registry or certify independent review.

## Verification

- New tests demonstrate denial from the actual `train()` entrypoint before
  baseline access, GPU use or output creation, even with a forged rights flag.
- Positive digest matching is exercised **only** under a locally patched,
  invented unittest registry; production H199 remains empty.
- Existing training integrity tests use a test-scoped invented allowlist solely
  to reach downstream media-corruption and completed-resume checks.
- Hosted H346 CI covers Windows and Ubuntu / Python 3.11 and 3.12.
- No private source, gold, media, weights, credentials or receipts are added.

This gate applies to the legacy `train_adapter.py` GUI SFT entrypoint; it
must not be misrepresented as a comprehensive audit of all earlier experimental
runtimes or as evidence that Hala has learned. H340 bilingual co-primary
familywise analysis and independent new blind OOD remain separate requirements.
