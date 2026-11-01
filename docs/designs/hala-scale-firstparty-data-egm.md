# H338 — Scale source-owned GUI data without laundering TRAIN or OOD approval

**EGM stage:** Elephant specification draft, **NOT reviewed by isolated Goldfish** and NOT approved for training. Grounded in Google [Elephant–Goldfish Model](https://github.com/google/elephant-goldfish-model) at commit `403d8881d62022490637d740657b33f01e321ea8`, read `resources/document_standards/feature-spec-standard.md` and `implementation-readiness-standard.md`. No EGM agents ran in this ChatGPT context; review state is `REVIEW_BLOCKED`.

## 1. Evidence and problem

- Owner-private genuine pretrained Qwen H336 (fixed model revision `89644892e4d85e24eaac8bacfd4f463576704203`) on 12 **public/exposed** original BeaconBoard scenes: English first-click **6/6 correct-open-and-focus**, Arabic RTL **4/6**, with **2/6 no-station-open**; no wrong-station, wrong-focus or abstention. This is NOT end-to-end task success or blind OOD.
- Real H333 matched prompt comparison: no observed improvement, control EN2/2 AR0/2 and candidate EN2/2 AR0/2. H291/H292 genuine BASE6/8 ADAPTED6/8: no learned gain.
- H325 RailKit has only 16 first-party original, source-browser-proven two-step candidate scenes/32 native action examples, NOT TRAIN-admitted; expansion by merely reusing same screenshots would be weak and potentially duplicate visual frames.
- H199 `ADMITTED_TRAIN_MANIFEST_SHAS` is intentionally EMPTY. H329 established one source introduction, not independent rights/title to all HTML or screenshot training. H173 OOD sealed, H293/H294 blocked, H250/H251 rights quarantined.

## 2. Goals, exclusions and decision

**Goal:** add a *distinct first-party* fictional GUI source family and exactly **128 deterministic bilingual scenes** (64 English, 64 Arabic RTL), each with native browser click → focus → second visual stage → type, yielding **256 unique decoded RGB source images** and 256 candidate action messages in an owner-controlled **in-memory** batch after 100% source checks. That is 8x H325's scene volume and 8x its action messages, but is NOT 8x effective model generalization.

**Not goals:** admission of TRAIN/DEV/OOD; using real accounts/patients/third-party images, existing donors or hidden gold; model training; owner-private Kaggle inference; LoRA gain; releasability; benchmarking 128 synthetic tasks as fresh blind OOD; installing EGM globally; passing a same-thread critique off as independent Goldfish.

**Chosen option:** create an original horizontal-row SignalDesk dispatch board in `h338/signaldesk.html`, unlike H325's 2×2 RailKit card grid, with no external assets, requests, fonts, JS imports, downloads or identity data. Exact scenario Cartesian product:

`language(EN,AR) × display_order(4) × theme(2) × row_density(2) × target(4) = 128.`

A visible generic **fictional scenario number** makes every scene's first screenshot independently distinguishable *without highlighting the answer or showing target gold*. A new revealed input after the native click makes its second screenshot distinct from stage one. Original target labels/typed codes exist only in source-trusted dataset construction, never in source-independent VLM evaluation.

## 3. Alternatives / trade-offs (EGM alternatives rule)

| Alternative | Advantage | Why insufficient |
| --- | --- | --- |
| Multiply H325 RailKit target+code permutations | Fast, small source change | 16 visible first-stage configurations would repeat RGB frames; pseudo-sample inflation and same GUI family |
| Immediately use third-party screenshot datasets | More real-world diversity | License, derivative media rights, provenance and H293/H294 privacy gates are not cleared; prohibited |
| **Original SignalDesk with 128 source scenes** | Different GUI arrangement, systematic EN/RTL, visually distinct 256 stages, clear custody and zero cost | Still highly correlated synthetic family; rights and independent validation incomplete; 128 is candidate count, not independent evaluation power |
| Generate random screenshots without browser actions | Very fast count | Fails native click→focus→type state transition and source oracle trust boundary |

## 4. Exact interfaces and implementation decisions

- `h338/signaldesk.html`: static 960×720 inline CSS/JS with 4 fictional connectors, deterministic `configureSignalDesk({lang,order,dark,dense,run})`; `window.signalDeskState()` source-only state array `[{id,open,value}]`; initial 4 visible buttons, after click EXACTLY intended input focused, one code text entry. No external resource dependency. `run` must be integer 0..127, shown as non-target generic visible scene number.
- `h338/original_large_train_candidates.py`: fixed source Git blob check, frozen 128 scene tuples, source `file://` and offline Playwright context, strict 960×720 source screenshot and decoded RGB hashing, ensure all 256 stage frames globally distinct. Use H199 strict native JSON click+type parser and H263 native `execute_one`. Require complete source DOM/oracle state, target focus, exact typed code, no image mutation before dispatch. Reject any failure with zero consumer calls. Opt-in `trusted_batch_sink` only, receives an immutable tuple of 256 `(PNG, task, native action JSON)` triples once **after all** checks, returns literal `True`. Absolutely no per-row streaming sink, no persistent media/model weights/gold/coordinates.
- `tests/test_h338_original_large_train_candidates.py`: six+ adversarial CPU contracts, opt-in original full 128-case Chromium/Edge browser qualification; verify 256 frames are distinct, 128 source states complete, role refusal and missing target/modified source refusal, no sink until full successful batch.
- `.github/workflows/h338-original-signaldesk-large-candidates.yml`: source-only GitHub Actions Chromium Linux and Edge Windows jobs, no model/paid GPU, 25-minute timeout; no screenshot artifacts or logs.
- `docs/designs/hala-scale-firstparty-data-egm.md` and `h338/README.md`: scope/rights/separation and reportable safe aggregate only.

**Technical outputs:** scene count 128, 64EN+64AR, 256 verified native actions, 256 unique decoded RGB frames, candidate-only status and all promotion flags FALSE. Do **not** emit tasks, target names, model-visible prompts, action coordinates, raw PNGs or hashes in CI logs, outputs or PR description.

## 5. Risks and mitigations

- **Rights laundering:** first-party source commit/Apache license does not independently prove copyright title or screenshot ML training grant. Independent explicit owner/reviewer source and rendered-image license decision, source-history custody, and separate reviewed H199 manifest admission remain mandatory. None created here.
- **Synthetic monoculture:** 128 scenes from one family have highly correlated visual structure; future approved training must add independent source families and measure family-diverse DEV/never-exposed blind OOD separately. No holdout drawn from this public source is blind after exposure.
- **Leakage:** avoid writing samples, git blobs of screenshots, task gold, hidden coords, model outputs or even per-scene detailed traces; no private receipt/Kaggle download in this PR; close source contexts and clear transient buffers.
- **CI scalability:** 128 scenes * two screenshot stages + unchanged checks can be heavy; allocate 25 minutes each in free source-only CI; do not shorten validation or sample only first cases to pass. Model-time/LoRA stays off.
- **Oracle contamination:** source DOM/target coordinates are visible to trusted extractor ONLY, never policy. This factory does no model inference.
- **EGM truth:** authentic Goldfish isolated zero-context reviewer needs native subagents and read evidence; absent in this environment, `REVIEW_BLOCKED` not PASS.

## 6. Phases, preconditions, and pass/fail criteria

1. Pin and review exact source HTML bytes and original Git introduction separately. No source-rights grant inferred.
2. Validate 128 native source tasks / 256 action stages with unique decoded RGB, exact source visual-state transition and all-or-nothing handoff on both browsers; if any scene fails, **no output**.
3. Publish only a stacked Draft PR with exact-head free GitHub Actions, aggregate counters and links to source, zero screenshots.
4. Request independent EGM Goldfish review (Zoom Out → Comprehension → Design Critic → Implementation Readiness) through a fresh supported agent and separately source/copyright rights review; this is not automatically completed.
5. Only **after** explicit rights, source-family split, signed owner decisions, H199 official TRAIN admission, FREE GPU quota and source-independent blind evaluation protocol, consider genuine model training and BASE/ADAPTED paired OOD comparison. Never silently advance stages 4–5.

**Success for this Draft** = 128 verified original source browser scenes, 256 unique decoded RGB action-stage candidate images, 256 valid native action labels, zero persisted candidate media, all exact-head CI green; marked **CANDIDATES ONLY**. **Hala product success remains blocked** until independent legal TRAIN admission, blinded generalization and trained-model gain.
