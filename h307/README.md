# H307 — one-shot private TRAIN-candidate handoff

## Verified engineering gap

The H286 source path validates all 24 bilingual ShiftDesk browser candidates
before invoking its trusted sink, but then invokes that sink **24 times**.
If a downstream writer fails after some calls, earlier writes may already
exist. The original H286 documentation explicitly says the sink is not
atomic. This is a data-integrity and partial-batch contamination risk,
**not** a finding of model improvement or unauthorized data release.

## Narrow opt-in mitigation

`h307.one_shot_train_candidate_handoff.handoff_train_candidate_batch_once`
uses the existing unmodified H286/H285 real-browser source validation and
collects all 24 candidate rows privately in memory. It verifies the
original ShiftDesk Git blob, TRAIN-candidate role, complete 12 EN / 12 AR
source proof, relative-coordinate supervision count, no model/optimizer
work, unique screenshot bytes and no official admission. **Only after**
those checks does it call one explicitly supplied batch consumer, once,
with an immutable tuple of rows. Failure before that point invokes the
external consumer **zero** times. No row, screenshot, label, coordinate,
prompt or digest is included in the returned summary.

The external consumer MUST implement its own transactional all-or-nothing
commit and return literal `True`. This boundary **cannot prove or roll
back external storage atomicity** if a consumer writes partially and then
raises; it fails categorically and does not retry. The H286 legacy
per-row sink remains unchanged for reproducibility of frozen historical
research. New private training pipelines may opt into this stricter
handoff, but this does **not** authorize a Kaggle run or training.

## Source-only invented adversarial qualification

```bash
python -m unittest discover -s tests -p 'test_h307_one_shot_train_candidate_handoff.py' -v
```

The tests use invented PNG-shaped byte strings and mocked source proof,
never real images, target labels, model outputs or private Kaggle data.
They cover complete handoff, mid-source failure, duplicate media, forged
admission, bool-as-int, forbidden BLIND role, consumer exception,
non-boolean commit acknowledgment and incomplete batch.

## Scientific and safety boundary

H199 `ADMITTED_TRAIN_MANIFEST_SHAS` remains EMPTY. Public ShiftDesk
TRAIN candidates are not officially admitted; Queueboard, RouteBoard,
ClinicGrid and FieldPort are exposed research, not untouched blind OOD.
H173 scored OOD is consumed/sealed, H212 scored rows barred, H250/H251
third-party sources quarantined pending rights and historical lineage.
H293 private result retrieval and H294 provenance write safety blocks
are untouched. No owner device, GPU, paid service, model weights, source
gold or private media are used here. This source-only PR is DRAFT and
must not be merged or released. The last genuine paired LoRA study
remains BASE 6/8 vs ADAPTED 6/8 with zero measured gain.
