# H62 — MiniMax Live Training Frontier and Decision Map

**Date: 2026-10-22.** This is the verified current handoff from signed Draft PR [H61 #45](https://github.com/TheHalfMoon/Hala/pull/45), exact head 5838f0b757693a7622dc6a40818be44928eec3b3. H61 is stacked on H60 #44, on H59 #43. H61 has four GitHub Python CI jobs all SUCCESS. H62 changes are in a separate worktree and will not mutate a concurrent H61/H60 runner or local scratch.

## Facts, not targets

| Measurement | Observed at H61 | Meaning |
|---|---:|---|
| Full original Hcompany/DragOn upstream TRAIN | 286,012 screenshots; 3,560,142 drag actions; four pinned TARs totalling 61,195,154,432 bytes | Upstream advertised; NOT completely materialized |
| H61 Stage C genuine exported TRAIN | 62,300 screenshots; 830,524 original action records | Measured export; 340 rows rejected for credential-like text or invalid source boxes |
| H61 Stage C source-bound training curriculum | 64,300 rows over 64,300 DISTINCT screens | Five drag capability families; 1,000 Arabic + 63,300 English |
| Curriculum native directions | 35,757 ordered; 28,543 unordered | MUST preserve H Company semantics |
| First-party bilingual synthetic | 2,000 verified scenes; 1,000 Arabic + 1,000 English | Actual source authoring and pixel checks, not real model improvement |
| Real Qwen3 tokenizer coordinate focus | 28.91% unweighted to 76.42% coordinate-weighted | Measured supervision density; does not itself prove visual grounding |
| H60 REAL pinned-Qwen adapter measured | Strict bare JSON 0 to 17/48; GUI task completion 0/48 | A real NEGATIVE model result |
| H61 training + evaluation | No completed remote H61 checkpoint/optimizer run or neural results at checked GitHub head | NOT DONE |
| Upstream DragOn public eval source | 160 official static scenes, 40 per each domain | Static geometry third-party EVAL, not browser execution and NEVER TRAIN |
| H61 readiness source receipt | status READY, model_result_known false, training_qualified false | CPU and data loader ready, GPU entitlement and runtime training not qualified |
| Colab GPU allocation | T4 CLI 503 Service Unavailable, 0 active assignments; L4/G4 entitlement denied | NO accepted provider GPU session; cause cannot be proved capacity-only |
| Vercel durable Workflow deployment | No verified deployed workflow/credential | NOT deployed; source-code orchestration alone is not cloud hosting |

**Historical correction:** The older H60 slider whole-archive-verification claim was contradicted by H61 review: only 6 of 14 windows had receipts and 57,797 screenshots / 506,857 actions were in the earlier slider shards. Do not double count those data with Stage C or assert full LFS verification. H59's 116-window map contains official source LFS SHA targets, but zero completed windows when authored. H61's 830,524 source actions supersede old imported-only 2,346 count; NOT the full 3.56M upstream.

## Critical runtime software defect fixed in H62

The inherited H61 remote_entry.py previously recorded a child training pipeline returncode but always returned process success 0. In a durable runner this can falsely mark a failed GPU training job as successful. H62 explicitly rejects a nonzero or invalid child exit and writes a failure receipt before propagating the error. H62 also requires an explicit HALA_PINNED_SOURCE_SHA (40-character git SHA) and verifies the real remote checkout HEAD BEFORE any training; an optional HALA_REMOTE_SOURCE_BRANCH selects the single approved source branch. Missing/wrong source pins fail closed. The test tests/test_h62_remote_entry_exit.py exercises both the actual mocked parent main failure exit 137 and mocked success, and invalid return codes. This CPU mock does NOT create a GPU checkpoint. **H62 cannot retroactively fix already-running H61 branch sessions; MiniMax must deliberately adopt this source and verify the exact checkout SHA.**

## Colab access and bounded orchestration

An advertised 36-hour wait is a **local client watcher**, not a provider-accepted queue or running GPU. The reported 325 T4 requests require a throttle review. Google documents dynamic free GPU availability, changing quotas, and usage limits; 400/412/503 may have different causes. Label failures by HTTP code and actual provider response, never claim 503 proves capacity rather than all entitlement/usage causes. A single official Colab CLI request at a time with finite attempts, exponential backoff, jitter, Retry-After when supplied and hard stop at configured 36h is the maximum safe continuation. Do not create duplicate watchers, run idle keepalives, rotate accounts, use paid compute or bypass limits. Stop on disallowed entitlement or quota. No public Vercel deployment means there is no deployed Vercel durability guarantee.

The existing Kaggle training guard stays in force until explicit provider commercial-usage permission has been established; donor copy permission is not permission to use a particular hosted GPU product. Private Kaggle data storage is distinct from GPU training entitlement. A different legitimate $0 provider is acceptable only if it actually permits this particular use.

## Source pin, checkpoint integrity, next phases

1. Inspect live GitHub H61/H62 PR SHA and the single waiting Colab client; record whether a session ID has actually been allocated; do not touch other agents' H60 private corpus publication job or scratch. Use source H61 receipts, not the obsolete H59 progress percentages.
2. Pin the exact git checkout SHA for the remote run BEFORE it starts; reject mutable branch drift, missing source media and absence of measured CUDA. The H61 config on Stage C is a TRAIN **candidate**, not a qualification certificate: source manifest and actual train admission must independently pass.
3. Once a permitted FREE GPU exists, perform actual gradient-bearing H61 run on the balanced bilingual 10k diagnostic first. Save actual loss/gradients/adapter weights plus optimizer/RNG/step and model/source hashes in PRIVATE remote storage before session expiry. Verify remote readback. Never write checkpoints to PUBLIC GitHub.
4. Matched comparison on the **same raw model prompts and inputs**: frozen pinned Qwen3-VL-2B base versus H60 versus H61, H55/H58 48 actual real Edge GUI cases with language/domain breakdown, and official public third-party DragOn eval split for static native ordered-aware geometry. DEV is not independently blind; keep sealed TEST separate.
5. Only if real task completion improves should MiniMax advance 10k => 64.3k => 100k => 1m => 3.56m original annotation scales as source availability, permitted GPU budget and unique screenshot counts actually support. Multi-million annotations are correlated to fewer screenshots; balance source image IDs and do not present raw action count as unique visual skills.
6. Hala requires further **click/type/scroll/navigation/multistep/error recovery/abstention** work to claim broad computer use. AgentNet and GUI Odyssey have other source modalities; neither is fully indexed into currently verified private Kaggle training.

**Strong gate:** no actual checkpoint means no H61 performance claim. H61 "READY" CPU gate plus a 36h watcher does NOT mean the GPU training started. The project goal is repeated independent real task success, not merely larger data counts.

Reference: h61/MINIMAX_PROGRESS.md; reviews/h61-readiness-20261022/readiness.json; h61/remote_entry.py; h59/EXECUTION_MAP.md historical data source map.

## Local software evidence (not GPU training)

- Pre-source-pin addition full Windows CPU suite: 828 tests total; 816 PASS and 12 preexisting SKIP, zero failure/error, 154.300 sec. Raw log reviews/h62-full-suite-20261022.log SHA256 60a0e8c4f3088700152652d2707c98a8507c9d635010f86a6b73660a98af70d0. This full suite was run before the subsequent exact-source-pin guard and fifth test; do not misstate it as validating the final edit. GitHub hosted exact-new-head CI still must be checked.
- Final exact-source-pin + nonzero-exit targeted suite: 5/5 PASS, source file reviews/h62-failclosed-remote-entry-focused-20261022.log SHA256 d51a87cd061b640ec72a259991e23762bbe4cfd390e2923eecf8bea82014400b. It mocks Colab; no checkpoint, no neural gradients.
- Final code Graft wiring-only build and check PASS: 325 files, 2,707 nodes, 7,012 edges; deep semantic meaning tier NOT built. Check receipt reviews/h62-graft-check-final-20261022.log SHA256 54b8ce98f67600fe6bebfc9cc956a7d530d84bc41b292a9cf268d726f6ea08bc.
- Source PR and model training remain Draft/NOT DONE; no cloud provider GPU was started by this handoff.
