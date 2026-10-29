# H174 — new-app split firewall and research qualification

**Status: METADATA CONTRACT ONLY.** No H174 TRAIN/DEV captures, no new OOD, no training and no model scores. This branch does not reopen historical evaluation.

## Frozen H173 original-app OOD (2026-10-29)

Owner v3 passed 32/32 new source-native captures and exact independent RGBA replay. Frozen H164 language scored 2/32 (6.25%) and H164 language + frozen rank16 H171 vision scored 3/32 (9.375%) true browser-state successes. Paired wins=1; regressions=0; two-sided exact McNemar p=1.0 (**not significant**). Arabic 1/16 to 2/16, English 1/16 to 1/16, Tingle 2/16 to 3/16, Tiny Slider 0/16 to 0/16.

Exactly 64 private free-T4 frozen model proposals and 64 original native actions were independently audited. H173 v3 is permanently **consumed and locked**; prior v1/v2 were permanently quarantined. All owner-only screenshots, gold, prediction JSON, and receipts remain off GitHub.

H173, H172 and all earlier DEV/OOD are NEVER reusable for training, prompting, hyperparameter/model selection, evaluation or rescoring. This OOD result does not qualify general computer use, Hala 35B, or any release.

## Fresh H174 approach: copy-first, strict disjoint splits

Adapt the existing H173 source identity/provenance contract, **not historical media**. TRAIN and DEV must use completely NEW permissively licensed original-app donors with immutable original Git + HTML checksums; OOD source apps must be **reserved by identity before training** and remain unseen. Do not capture reserved OOD images or access native reference gold during TRAIN/DEV model engineering.

The H174 split_firewall validates only metadata. Its **v2 source contract** requires canonical `owner/repository` provenance and rejects reuse of the same upstream repository across TRAIN / DEV / OOD_RESERVED, even at different commits or under case-variant aliases. The earlier v1 metadata contract is intentionally rejected because distinct Git/HTML SHAs alone did not prove source-family disjointness. It requires a nonempty **owner-provided SHA-only** historical registry, source-disjoint TRAIN / DEV / OOD_RESERVED identities, new TRAIN/DEV case PNG and decoded RGBA SHA uniqueness, allowed language and action types, and strict rejection of owner gold/reference fields. Actual historical registry *completeness* cannot be proven from this manifest: independently audit it before using it for admission. A passing metadata check does not prove source license, vendor source identity, browser replay, real native reward, or model performance.

Next gates: independently verify donor licenses, source SHAs and identities; capture only TRAIN/DEV with a separate one-shot replay; balance Arabic/English genuine native GUI actions; train a frozen candidate only from approved fresh TRAIN using free compute; select using new DEV only; freeze model/code; capture truly new reserved original-app OOD once and assess once. Negative results must remain visible.

**No private image/gold upload; no paid compute; no main merge; no release/publication; no H173 benchmark-driven tuning.**
