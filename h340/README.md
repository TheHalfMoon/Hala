# H340 — Pre-register family-level paired Qwen evidence before blind OOD

**Status: statistics primitive only; not a blind evaluation, legal approval, Goldfish review, trained model or release.** CPU-only, stacked on H339 source custody. No private data are read and no real Qwen results are generated.

## Scientific evidence and decision

H333's owner-private genuine pretrained **publicly exposed** paired prompt study had four ties: EN control 2/2 and candidate 2/2, AR both 0/2. H336's genuine exposed first-click pretrained study had EN 6/6 and AR 4/6, with two AR no-station-open outcomes. Historical H291/H292 paired BASE 6/8 vs ADAPTED 6/8 showed no learned gain. None qualifies a new adapted-vs-base source-family-blind OOD claim. H199's official TRAIN allowlist is empty; H173 historical scored OOD is sealed, H293/H294 safety blocks remain.

H325 RailKit and H338 SignalDesk are only **two publicly exposed synthetic source families** and both are unapproved TRAIN candidates. Independent copyright title and generated-screenshot ML grants are unreviewed. Splitting scenes within one family cannot produce a new independent source family. Their 288 action-stage candidates are not 288 independent evaluation units. Do not evaluate a blind OOD from either family.

A separate independent custodian must prospectively freeze a **new never-exposed source-family-disjoint private OOD** before model selection or evaluation, including rights/title, model/data pins, identical two-arm native screenshot-conditioned GUI protocol, paired scene outcomes, sealed gold and safety thresholds. This document and code cannot grant those facts.

## Frozen statistical design (for future authorized private evaluation)

- **Independent unit:** original GUI source family, not screenshot, scene, action stage or synthetic permutation. External reviewer must attest family identity and source history. Code intentionally accepts no family IDs or gold.
- **Arms:** genuine pinned Qwen BASE versus genuinely trained pinned ADAPTED weights on the same never-exposed tasks with the same processor, grammar, native browser conditions and no model-selected retries. Prompt-only arms are not adapted-model evidence.
- **Co-primary endpoints:** English and Arabic RTL. Every family contributes at least one paired scene in each language. This cross-language matching must be independently verified.
- **Input per family/language:** paired 2×2 counts (both succeed, base only succeeds, adapted only succeeds, neither succeeds) and per-arm abstention/unsafe counters. No images, IDs, tasks, points, model outputs, gold, weights or receipts.
- **Descriptive effect:** pooled adapted-minus-base success fraction. Within-family direction is the sign of adapted-only minus base-only. Tied families do not count as positive or negative signs. The sign test treats families, not scenes, as independent.
- **Family-aware uncertainty:** exact **two-sided** binomial sign p for each language, then Holm step-down across exactly two hypotheses at alpha 0.05. Positive evidence additionally requires improvement direction in each language: significant deterioration is not gain. A p-value cannot establish statistical power, legal rights, blind custody or release.
- **Sanity bound:** with six all-winning independent families in both languages, each raw p is 1/32 and adjusted p is 1/16, so neither passes. Seven all-winning families in each language yield adjusted p=1/32. These are mathematical examples only, not sample-size or power guarantees.
- **Safety:** an independent evaluator must separately assess abstention, unsafe actions and familywise uncertainty. Counts alone never qualify safety. An unsafe action cannot be redeemed by successful final task state.
- **Privacy:** do not feed real private aggregates into public GitHub CI or PRs. Tests use invented numbers only. Any future private aggregate disclosure requires separate explicit permission and custody.
- **Fail closed:** reject extra fields, malformed counts, bools as integers, impossible abstention bounds, unknown languages, empty or oversized cohorts. Never reuse H173, H212, H250/H251 or H293/H294 blocked material.

The H340 source returns exact rational p-values and bilingual aggregate counts. Every output explicitly marks independent family custody, rights and TRAIN admission, new blind OOD, genuine pinned adapted/base model attestation, safety, learned gain and release **false**, even if invented examples are mathematically significant.

## Alternatives and EGM-style decision

| Alternative | Reason |
| --- | --- |
| Per-scene pooled accuracy and ordinary binomial CI | Pseudoreplication: 128 near-duplicate SignalDesk scenes are not 128 independent source families |
| Scene-level McNemar alone | Useful within a family, insufficient for independent cross-family generalization |
| Reuse sealed H173 or exposed BeaconBoard | Violates blind custody |
| **Exact two-sided family sign test + Holm EN/AR** | Conservative, transparent, low power; depends on independently verified source families |
| Immediately claim LoRA gain | No admitted TRAIN, new independent blind OOD or real new adapted-vs-base evidence |

**Acceptance for this Draft:** invented-only deterministic CPU tests for Holm, significant deterioration rejection, scene inflation, malformed/private fields and exact-head hosted CI. **Not accepted:** independent EGM Goldfish review (REVIEW_BLOCKED), copyright grants, TRAIN/DEV admission, private OOD, GPU inference/training, learning or release.

Run the invented regression with: python -m unittest discover -s tests -p test_h340_familywise_paired_inference.py -v

Next requires an independent rights/custody reviewer for RailKit and SignalDesk, a separately authorized never-exposed source-family DEV and blind OOD custodian, and a genuinely adapted Qwen trained on lawful admitted data. H340 changes no H199 admission or protected main.
