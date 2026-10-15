# PR #1 immutable-frontier combined review

**Verdict: no correctness, parsimony or product must-fix identified for the engineering candidate at this head. This is source/helper qualification evidence, not permission to merge or a model release approval.**

Reviewed base `00f6692383ca9e9cc147380b062676dd74932e83` to head `9f4c26dbb4d4cefe15c4177e10f3e6976145af14`. Captured engineering diff SHA-256 `0b37ab7ccd72d4a7a0af7af98d2d83870f3f9ded292ea75fc4030aa7d9c96b63`. Review used an independent `git archive` of that head, AGENTS.md/CLAUDE.md/PRODUCT.md, applicable contracts/evaluation/adaptation/service/release specifications, and existing immutable review reports. Later H14 builder files and the mutable checkout were excluded. This judge made no product source edits, commits, pushes, model downloads or remote/paid actions.

## Findings as data

```json
{"must_fix":[],"worth_considering":[],"skip_it":[{"location":"training/scripts/train_adapter.py; specs/08-adaptation.md","issue":"Actual processor/LoRA/runtime/resume quality remains unexecuted at this head, but source and status explicitly label the pipeline a candidate.","fix":"No engineering-PR source fix; require real bounded remote evidence before upgrading its qualification claims."},{"location":"experiments/runtime/colab-calibration-fp16-20261014.json; docs/CURRENT_STATUS.md","issue":"7/8 reuses the eight observed strict-run fixtures and measures geometry only.","fix":"Preserve the existing calibration limitation; obtain frozen held-out full-task evidence for model qualification."}]}
```

## Bars and evidence

| Bar | Result | Concrete evidence |
|---|---|---|
| Correctness, contracts/service | Pass within CPU mock scope | Immutable proposal/request bindings, bounded action vocabulary and geometry, cancelled/exhausted behavior, backend-output reparsing, authenticated loopback service, admission/deadline tests and fail-closed real backend remain covered. All 91 head tests passed independently. No execution authority or real-backend readiness is inferred. |
| Correctness, data/evaluation | Pass within original synthetic scope | Detached checker labels and budgets, original-content train binding and normalized semantic deduplication remain present. Independent execution produced oracle 180/180 and wrong-target/stale/invalid/false-done each 0/180. These are 180 variants in 12 dependent designed families, not 180 independent real-model trials. |
| Correctness, trainer repairs | Pass for inspected source/helpers | Every image and JSONL is audited before runtime work; completion labels reject prefix/token-budget mismatches; immutable checkpoint versions hash the complete adapter/resume pair and publish one pointer; completed resume finalizes before model imports. Current integrity tests cover media substitution, corrupt/extra/missing artifacts and publication interruptions. CPU checkpoint fixtures contain synthetic bytes, not actual optimizer or adapter tensors. |
| Correctness, diagnostics/economics | Pass | Independently recomputed every saved calibration decoded point and geometric hit, checked strict rejection and equal IDs/goals/image hashes/target boxes. Tiny-denominator economic overflow probes reject instead of emitting infinities. Notebook canonical baseline/source hashes match and stored outputs are empty. |
| Parsimony | Pass | Separate artifact, masking and checkpoint modules each enforce a concrete boundary. The custom transport admission lease and framing reader preserve tested bounded-work behavior. No additional framework, provider abstraction or resident model is justified by this diff. |
| Product | Pass for the proposed engineering scope | Contracts, mock service, synthetic environment and research satisfy PRODUCT.md's independently testable first stage. H8/H10/H13 and native Aladdin/model qualification remain incomplete in status/specs. English-only initial-state SFT, geometric diagnostics and proposed two-microbatch smoke are expressly limited. |

## Review coverage and freshness

- The original foundation combined report's C1-C4 are superseded by training-delta-1 and foundation-delta-2. The latter cost-only delta expressly did not re-review trainer source. Current source/helper checks and this head suite support the resolved media, checkpoint, finite-output and strict-parser repairs; they do not establish actual numerical resume equivalence.
- The original evaluation combined report's four must-fixes are superseded by evaluation-delta-1, whose factory/engine blob IDs match this frontier's repaired implementations. This review independently reran the canonical execution and negative controls.
- The provider/font and coordinate calibration reports describe their exact narrow source/experiment deltas; they do not supply whole-trainer security coverage. The old `h2-h11-security.md` inspected contracts/inference/service and their tests only. A fresh separately coordinated security judge must cover the current training artifact/checkpoint/deserialization and research-fetch surfaces before the parent calls all engineering gates qualified. This combined report supplies no security approval.
- Primary Jev evidence has 19/21 source hashes matching this head. Its stale baseline and trainer entries are both superseded by matching 2/2 hashes in `jev-colab-provider.json`; all 21 listed files therefore have current typed evidence across the two records. The reported concern probabilities are advisory and were not interpreted as a gate approval.
- Captured Graft summary reports build/check/blast exit zero and a 413-node graph; deep context is absent. This judge did not rerun Graft or independently prove its graph freshness. Parent exact-frontier graph qualification and live CI/signing verification remain separate evidence.

## Independent checks and limits

Bundled Python 3.12.14 through RTK: `python -m unittest discover -s tests` passed **91/91** in **12.151 seconds**. The independent scratch probe loaded all **58 English SFT PNG records**, with **8 train and 4 held-out families**; reproduced canonical execution controls; rejected three tiny-denominator/overflow cost scenarios; checked notebook source parity; and audited Jev source hashes. Captured metadata integrity also passed **35 pinned file hashes across 9 candidates**, with four DragOn evaluation samples.

The archived tracked inventory contains **365 files, 5,721,947 bytes total**, largest **446,616 bytes**; no multi-GB weights are present. The strict real diagnostic remains **0/8**. The separate normalized calibration recomputes to **7/8**, including **4/4 English and 3/4 Arabic**, on the same observed fixtures. Both records deny release qualification. Image hashes identify generated pixel bytes; this review did not independently witness the recorded remote inference.

No actual processor training invocation, LoRA gradient step, adapter quality, continuous-versus-resumed numerical comparison, model-fit qualification, held-out complete-task execution, quantization, native Aladdin integration or serving economics was established by these CPU checks. Research hashes establish captured provenance/integrity, not independent commercial-rights clearance. Founder merge and production decisions remain separate.

panel: fresh combined session-model judge — correctness PASS (source/helper scope) · parsimony PASS (inline bar) · product PASS (candidate scope) · security — separate parent-coordinated fresh judge required; no CPU-derived model/release approval
