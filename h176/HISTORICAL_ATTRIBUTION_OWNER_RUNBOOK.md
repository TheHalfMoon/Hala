# H176 owner-only historical family attribution firewall

**Scope: source identity metadata, not dataset admission.** No model calls,
GPU, media, gold, predictions, screenshots, prior OOD reopening or publication.

## Why this additional gate exists

H174's original validate_with_source_history() requires the known H173 app IDs
and frozen source hashes, but its legacy schema is composed of independent
arrays. A repository list containing a single placeholder could pass even
while earlier TRAIN/DEV/OOD source repositories remain unattributed.
H176 adds **paired owner-only records** for every app in the historical
identity list, cross-checking the repo/Git/HTML against the old arrays and
the two pinned H173 source SHA pairs.

## Paired JSON schema (actual owner manifest must remain OFF GitHub)

The top-level object has exactly schema and families:

- schema: hala.h176.owner-historical-source-family-attribution.v1
- families: nonempty list, at most 100000 entries
- Each entry must have precisely: app, repo, git_sha, html_sha256
- app: known original app ID; repo: canonical owner/repository
- git_sha: 40 lowercase hexadecimal characters
- html_sha256: 64 lowercase hexadecimal characters

These are *field shapes*, not evidence or usable example identities.
All 11 previously known TRAIN/consumed families and the 2 H173 pinned
families, PLUS six independently identified public H137-H163 source IDs,
must be included, alongside ALL additional prior source families.
The six extra source names are splitjs, react-dnd-kit-board,
react-html5-board, react-dndkit-board, todomvc-es6 and
gridstack-original-column-demo. Five of these have exact public source
Git pins in H138/H140/H158/H163; splitjs was an H137 npm-based evaluation
library without an equivalent Git-commit identity recorded in that phase.
Never fabricate placeholders or infer a repo merely from an application name.
Two apps originating from the same upstream repository remain one source
family for future split eligibility, even if their old Git revisions differ.

## Offline owner-only command

Prepare four small metadata-only JSON files outside the public repo:

1. Fresh proposed H174 candidate metadata (schema, sources, cases).
2. Prior-image hash-only registry (image_shas and visual_shas), recovered
   from existing hash metadata only, never by reopening old images.
3. Legacy H174 source history (historical_app_ids,
   historical_repositories, source_git_shas, source_html_shas).
4. Paired owner H176 historical family source metadata as described above.

Run this command locally from the Hala checkout, substituting your real
OFF-REPO metadata JSON paths (paths shown below are placeholders):

    python -m h176.audit_owner_history --candidates D:/PRIVATE/candidates.json --prior-pixel-hashes D:/PRIVATE/prior-hashes.json --prior-source-history D:/PRIVATE/source-history.json --paired-source-history D:/PRIVATE/paired-sources.json

The CLI reads ONLY these four named JSON files, caps each at 4 MiB,
rejects duplicate JSON keys, malformed structures and unsupported fields,
prints **no owner source identities or paths**, writes nothing, and never
opens benchmark media. Failure returns code 2. Passing returns 0 plus only
counts and fingerprints. Its terminal status remains:

    HISTORY_COMPLETENESS_UNPROVEN_NO_TRAIN_DEV_OOD_ADMISSION

## Remaining blockers before any fresh model work

- Independently audit source manifests for *every* historical TRAIN, DEV
  and consumed OOD app, repository, alias, Git and HTML identity, not just
  the 19 known minimum. Internal matching cannot establish completeness.
- Independently verify original source provenance, legal notices, full
  history and family-disjointness against proposed TRAIN/DEV/OOD.
- Reproduce authentic original browser-native behavior and independent
  decoded-RGBA replay before new TRAIN/DEV admission. Keep OOD strictly
  as untouched *source reservation only* until training is frozen.
- Never revisit H173 v3 or earlier consumed OOD to tune or rescore as new.
- Only after all gates pass may fresh TRAIN-only tuning on free allowed GPU,
  DEV-only selection and one-shot genuinely unseen OOD scoring occur.
  Report independent baseline, n/N and statistical uncertainty.
  H173 result 2/32 vs 3/32, McNemar exact p=1.0 does not show improvement.

Keep this work in Draft; no protected-main merge, dataset publication,
public model, paid compute or human-review requirement.
