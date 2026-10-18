# H27 — H26 measured failure forensics and prospective Hala research gate

**Date:** 2026-10-18. **Status:** source-based post-hoc DEV analysis, **NO-GO** for model adoption, native actions, main merge, or release. Hala-only; no changes to neighboring repositories.

## Provenance and scope

Original Kaggle H26 DEV version: https://www.kaggle.com/code/abdulazizshehri/hala-h26-choice-dev-20261018. Frozen source: `1a171faf5e824b9f21ef7a38ac8c5bccbdaf8542`. Evidence branch: `ps/h26-dev-v1-evidence` at `a71e719ec34ececb9d0a7be8df9710386c91dcac`. Original Kaggle `result.json` SHA256: `789b3bb04e095334765774472fca9264fa7e094efd95465762ec3714bebc4aed`.

`tools/analyze_h26_failure.py` first invokes the original source/hash-bound `tools.verify_h26_dev.inspect` before decoding/rechecking the raw bytes and recalculating descriptive distributions. The retained output is `docs/reviews/h27-h26-failure-forensics.json`. This is **not** a new model run, independent external evaluator, prospective TEST, confidence calibration or causal ablation. Test-generated example rows are explicitly synthetic machinery fixtures, not observations.

Reproduction (on the H27 source checkout and with the original evidence file available):

```powershell
python -m tools.analyze_h26_failure "..\Hala-h26-v1-evidence\h26-choice-dev-v1\result.json" --expected-sha256 789b3bb04e095334765774472fca9264fa7e094efd95465762ec3714bebc4aed --source-commit 1a171faf5e824b9f21ef7a38ac8c5bccbdaf8542 --output NEW_H26_FORENSICS.json
```

The output path MUST be new. Never modify the evidence branch or use the public H26 DEV as newly unseen TEST.

## Observed failures, not speculations

| Property | English | Arabic | Combined |
| --- | ---: | ---: | ---: |
| Public synthetic DEV cases | 21 | 21 | 42 |
| Choice C1 selected | 14 | 17 | **31** |
| Correct C1 among C1 selections | 3 | 3 | **6/31** |
| Choice C0 withheld | 4 | 4 | **8** |
| Correct decisions, all codes | 5 | 3 | **8/42** |
| Correct among complete non-C0 proposals | 5/17 | 3/17 | **8/34** |
| Unsafe-user cases explicitly rejected C7 | 1/3 | 0/3 | **1/6** |
| Unsafe-user cases withheld C0 | 2/3 | 2/3 | **4/6** |
| Unsafe-user cases routed to non-refusal | 0/3 | 1/3 | **1/6** |

- Of the 31 C1 selections, only 6 matched the source-authored intended label. This is **19.35% conditional C1 precision on correlated public DEV**, not representative precision or a population statistic. The overall C1 selection rate is **73.81%** (Arabic **80.95%**, English **66.67%**).
- Six of six genuinely ready tasks selected C1, but so did all six stale, all six conflicting-authority, all six saved-result verification tasks. Five of six missing-parameter tasks also chose C1. These are concrete failure modes in task/state discrimination.
- No instance selected C2 (refresh), C3 (hostile page), C4 (persisted-state verification), or C6 (conflict clarification). One selected C5 and two selected C7. On hostile pages, C7 appeared once, which is still the **wrong source-oracle decision**: refusing the user's benign inspection is not the same as ignoring hostile page instructions.
- C0 is a non-proposal. It may prevent direct exposure to a bad recommendation in the host, but **C0 alone is not an explicit appropriate C7 safety refusal** and does not prove safety comprehension.
- The grammar gate structurally guarantees C0-C7 formatting; all 42 generated valid codes prove only constrained serialization. Its selection frequencies and semantic correctness cannot be compared as an improvement over H25, whose tasks and contract differ.

## Root-cause hypotheses and non-findings

**Established symptom:** an extreme C1 response-preference collapse coexists with poor discrimination of stale frames, conflicts, hostile-page commands and required postconditions. Semantic Arabic success is 3/21; the one unambiguously unsafe proposed action occurs in Arabic.

**Not established by this dataset:** whether the failure is driven primarily by token priors, first-choice/default bias, model language ability, prompting, serialization, choice-label order, quantization, incomplete observation semantics or a tokenizer problem. Raw chosen codes provide no full per-option logits, log-probability calibration, counterbalanced codebooks, blinded randomized interventions, equivalent control prompts or independently bilingual-adjudicated gold labels. A causal explanation would currently be speculation.

The synthetic user goals themselves often name the right state ("refresh", "ignore", "verify", "clarify"), yet the model still returns C1; these examples do **not** isolate visual reasoning. The observation is symbolic and does not supply actual screenshot pixels. The prompt is English even when the task goal is Arabic. Test both linguistic and format hypotheses prospectively, not with DEV retuning masquerading as evidence.

## Proposed independent, preregistered research phases (not executed)

1. **Evaluation design before model edits.** Author independently bilingual-checked scenarios from new genuinely disjoint domains (e.g., public-health form intake, parcel chain of custody, municipal notice handling, research-lab instrument booking). Obtain human domain and Arabic review of every oracle and adverse consequence. Keep held-out TEST inputs inaccessible to prompt authors/model tuning; publish only predeclared hashes, metrics and evaluation code as permitted. A public GitHub dataset is not a blind TEST.
2. **Counterbalanced causal DEV experiments.** On separate declared DEV partitions, vary codebook ordering/letter IDs, semantic label names, language of instructions (Arabic/English), scoring mode (finite-code and unconstrained typed projection), and task-state evidence without changing the goal. Record choice/logit distribution where supported, order sensitivity, coverage, latency and GPU memory on an identical permitted model/revision. Never claim a result until the free research run and receipts actually exist.
3. **Safety and provenance challenge families.** Independent hostile-page/role-spoofing injection, destructive user actions, missing/ambiguous permissions, stale-observation replay, contradictory trusted sources and separately saved postcondition mismatches. Add adversarial bilingual minimal pairs, Arabic diacritics/dialect and RTL screenshot/RAQM/OCR visual cases only after a qualified image path exists. Prompt material embedded in pages is data, not system authority.
4. **Selective risk and refusal measurements.** Report exact valid grammar separately from expected decision, independent verifier outcome, unsafe proposal rate, explicit safe refusal precision/recall, wrong-withhold rates, selective accuracy/coverage and language-by-state confusion. C0 is **not** a calibrated probability. Establish thresholds and confidence intervals prospectively before observing TEST, including an appropriate zero-observed-unsafe-proposals condition with statistically defensible sample sizes; zero on six cases is not proof of safety.
5. **Real closed-loop qualification.** Validate that native executor authorization, provenance-bound targets, fresh observations, bounded side effects and independent durable postconditions are checked externally; no command executes on model text. Qualify actual H23 CUDA in-flight cancellation, cleanup, restart and resource recovery on genuinely available permitted hardware. Record verified latency, memory, costs and exception rates separately from task success. Leave Orcel and Aladdin repositories untouched.
6. **Review/rights gates.** Real model-driven Alibaba Open Code Review (not rules preview), Jev typed advisory, actual Graft tier, independent PStack and E2E as applicable; record exact tool provenance and outcomes. Recheck checkpoint/data commercial and Kaggle non-commercial-use terms before commercial R&D use. Founder controls merge/release.

## HalaMind / HalaType / HalaReliance architecture decision

- **HalaMind:** keep a small bilingual intent/state selector as an *untrusted proposal generator* only. Evaluate semantic accuracy on independently reviewed labels before designing RL/finetuning; avoid adopting arbitrary C1 defaults or promoting a structural eight-code output as correct reasoning.
- **HalaType:** use a separate perception/grounding component with explicit screenshot/frame digest, target bounds, timestamp and fresh-target validity; do not conflate synthetic labels with pixels or let a coordinate imply permission. Verify true Arabic/RTL images, not just text strings.
- **HalaReliance:** deterministic host policy owns action type and permission checks; any learned uncertainty or selective classifier may only abstain/escalate and never upgrade authority, remove verification or override host deny. Require held-out risk-coverage calibration before making claims about low-cost model cascades.
- **Cost policy:** no paid APIs, overflow, Windows model weights or hidden cloud allocation. Free research capacity and terms must be checked separately; do not infer authorization merely from a free quota.

**Disposition:** preserve H24/H25/H26 immutable source and experimental artifacts. H27 adds only auditable diagnosis, CPU checks and research design, with **no changes to Hala runtime action authorizers**, no novel model performance claim and no automatic PR merge.
