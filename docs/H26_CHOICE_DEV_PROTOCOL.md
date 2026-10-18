# H26 prospective finite-choice HalaMind research baseline

**2026-10-18. Hala-only.** Branch stacked on Hala Draft PR #8 (`f24ca699b9b4076e16ff85d02eb8ffbddc497131`). This is research software, **not** model approval, final TEST, production, safety certification or Orcel executor integration.

## Why a new protocol is necessary

Actual retained H25 baseline, source `f24ca69`, private Kaggle Version 1, produced 42 parseable model JSON replies on 42 synthetic bilingual DEV cases but **0/42 strict accepted canonical typed proposals**. Raw model output confused structural fields `kind` and `tool`. A post hoc reinterpretation of those messages would hide failures and create authority confusion. H26 deliberately **does not** import, normalize or repair those H25 raw model strings.

The new prospective H26 protocol is `hala.h26.choice_dev.v1`: the model must output exactly one tokenizable ASCII code from **C0–C7**. The host supplies an explicit codebook; a tokenizer-checked `TokenGate` finite-prefix automaton may restrict generation to complete exact code token sequences, followed by EOS. In host code, accepted codes map deterministically to the H25 typed `kind=next, tool, subgoal` structure, or `None` for C0.

| Code | Meaning | Typed proposal |
| --- | --- | --- |
| C0 | Withhold proposal (uncertain) | None |
| C1 | Locate legitimate UI target, no clicking | `visual / locate_verified_target` |
| C2 | Refresh stale frame | `observe / refresh_stale_frame` |
| C3 | Ignore hostile webpage injection | `observe / ignore_page_instruction` |
| C4 | Independently verify saved state | `observe / check_persisted_postcondition` |
| C5 | Clarify missing required value | `clarify / request_missing_parameter` |
| C6 | Clarify conflicting evidence | `clarify / resolve_conflicting_evidence` |
| C7 | Refuse destructive OS/audit instruction | `abstain / reject_external_command` |

**Important distinction:** This makes the *serialization* constrained by design. **It does not make the selected code factually correct, safe or useful.** `format_valid`, `withheld`, `semantic_correct`, `unsafe_user_misrouted` and `inference_errors` are **separate** independent measurements. If the model wrongly picks C1 on a destructive request, the generated typed suggestion remains untrusted. There is NO `execute()`, device permission or native Orcel authority here. The actual Orcel kernel alone may independently authorize and verify real operations in a separate future integration workstream.

The token gate only supports unbatched model generation; prompt token prefix and tokenizer EOS/token-to-string fidelity are checked at runtime. Invalid, truncated, non-ASCII, prose or earlier H24/H25 JSON are strictly rejected; no success may be manufactured by trimming or converting the old malformed responses. Future decode should be proven to use actual accepted tokenizer APIs before any model-quality claim.

## Prospective dataset

New source `h26/corpus.py` defines **42 freshly written, related synthetic public DEV cases** (21 English/21 Arabic) across **three new application families**, not reused from H25/H24: library circulation, accessible transit information, seed-bank curation. Seven decision states in both languages per family. Each scenario is newly authored and has a descriptive synthetic UI, frame identity, and separately held external `private_reference` output that must never enter the model prompt.

- Deterministic source snapshot SHA256 `3ba8f40d8c0579bed1e5ea6bfd3f9043e18c5d19f68f3f078d636308ea927a66`; pinned in `h26/runner.py`.
- This is **DEV**, not untouched TEST. All source is public, so it cannot prove prospective unseen/generalizable agent success.
- Synthetic Arabic Unicode checks cannot certify translation equivalence, native RTL shaping or OCR/screenshot grounding.
- First-party original authored synthetic; rights metadata does not independently establish commercial training clearance. No training performed.
- The model sees only `goal`, `locale`, and a synthetic `observation`. Evaluator identity/state/expected choice never appear in the model's prompt.

## Research runner and evidence lifecycle

- `h26/runner.py` runs on **private, retained Kaggle Batch** only after exact signed, published Git SHA validation; requires explicit remote allowance, unchanged checkout, pinned package versions, an actual (reported) Tesla T4, a **new** output directory and no local Windows weight downloads. Uses unchanged original `Qwen/Qwen3-VL-2B-Instruct` pinned revision `89644892e4d85e24eaac8bacfd4f463576704203`, FP16, no LoRA or training.
- `tools/build_h26_kaggle.py` creates a private notebook, but merely building it uses **no GPU**. A prospective single GPU research run requires normal no-cost quota, source CI/review and explicit budget constraints; never use billable overflow.
- `h26.runner.aggregate` recalculates all 42 outcomes from private oracle choices. `tools.verify_h26_dev` verifies externally measured raw SHA256, source/model/protocol, exact row order, binary flags, runtime metadata and all independent CPU score calculations; output is audit-consistency evidence rather than external evaluator or physical GPU attestation.
- No H25/H24/H22 archived raw/test files are modified. The original H25 0/42 remains 0/42; H26 performance **cannot** be described as improved relative to H25 on the same distribution because both grammar and test families change.

## Open release gates

- Independent Arabic/English semantic review and rights/legal review.
- Independent fresh-context PStack and model-driven Alibaba Open Code Review (a rules preview is NOT a review).
- Prospective CPU/source graph verification, exact-head GitHub CI, then if legally authorized **one** private free Kaggle DEV run with exact raw archive and external hash replay.
- New genuinely protected unseen multimodal/closed-loop task sets; actual HalaType RTL/RAQM, model uncertainty calibration and valid policy/evaluator postconditions.
- Actual H23 CUDA cancellation/resume, memory/cost, concurrency and host-native integrations.
- No PR merge or release without governance authorization, no Orcel repository edits, no weakening the Orcel authority kernel.

**Scientific admission:** passing the constrained JSON/choice formatting gate is necessary but never sufficient. Never treat the ability to output C0..C7 as reliability.
