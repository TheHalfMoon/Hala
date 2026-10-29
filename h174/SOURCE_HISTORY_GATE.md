# H174 historical original-source family crosscheck — source metadata only

**This is not a completed historical audit, dataset admission, native browser
qualification, training result, new OOD, or release permission.** It extends
the H174 v2 new-split firewall with the missing *historical repository*
comparison. Two different Git revisions or HTML builds of the **same upstream
repository** are one app family: no renamed source from previous TRAIN,
consumed DEV, or scored OOD may silently become a fresh donor.

## Owner-only protocol

1. Without opening any historical pixels/gold/predictions, assemble a
   **source-identity-only** prior registry using verified immutable metadata
   already recorded for every previously used donor. It has exactly the
   `hala.h174.owner-prior-source-identity-history.v1` fields:
   `schema`, `historical_app_ids`, `historical_repositories`,
   `source_git_shas`, `source_html_shas`. All lists contain strings,
   not source files, photos, references, coordinates, actions or task labels.
2. Include all known prior source app names from
   `h173.source_preflight.TRAIN_AND_CONSUMED_FAMILIES` plus
   `h173.source_preflight.SOURCE_PINS`. Include H173 pinned immutable
   Git and HTML hashes. Audit that the registry covers **all** older
   TRAIN/DEV/OOD applications and upstream repositories separately; this
   completeness check CANNOT be inferred from the manifest alone.
3. Keep that complete historical owner-only registry off GitHub. Pass it
   together with **fresh** candidate metadata and the owner historical
   SHA-only PNG/decoded-RGBA registry to
   `h174.source_history_gate.validate_with_source_history()`.
4. The function first runs the H174 v2 firewall. Then it rejects
   overlapping historical app IDs, SHA aliases, and upstream repository
   identities (case-insensitive), including a previously consumed donor
   under a different app name, Git SHA, HTML SHA, or reserved OOD role.
5. On a passing source-only *crosscheck*, independently audit provenance,
   complete historical registry coverage, permissive upstream licenses,
   exact original Git/HTML bytes, owner-only genuine browser-native actions,
   visual SHA uniqueness, source-disjoint TRAIN/DEV and reserved OOD, and
   independent fresh decoded-RGBA replay BEFORE any data admission.

A returned registry SHA-256 fingerprint binds the supplied **identity-only**
manifest's values and is independent of list order. It is **not** proof of
historical completeness, trustworthy origin or training authorization.

The older `split_firewall.validate()` remains a metadata-only V2 check
for backwards-compatible synthetic tests; it DOES NOT check historical
repository identity. Future actual admission must use the stronger
crosscheck plus the independent owner-only gates listed above. Neither
function touches historical media or scores, runs browser code, opens a model,
calls a GPU, or issues a native action.

All H173 v3 and earlier DEV/OOD remain permanently consumed. The actual
H173 frozen model result remains 2/32 vs 3/32, McNemar exact p=1.0 (not
significant). No new benchmark has occurred. No paid compute, no merging,
no release/publication; source-only synthetic GitHub CI is not owner Windows
native-app QA.
