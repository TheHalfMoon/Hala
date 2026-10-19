# H28 — Prospective bilingual counterbalanced public DEV protocol

**2026-10-19 | Hala only | source/CPU research, DRAFT and NO-GO for model adoption.**

This milestone implements **testable source and evidence-auditing tools**, NOT another Kaggle job. H26's measured 8/42 semantically correct (Arabic 3/21), 31/42 C1 preference and unsafe Arabic C1 error are retained unchanged on the separate immutable evidence branch. H24/H25/H26 results and their frozen TEST/DEV source are not re-scored or relabeled; comparison of scores between their different tasks and protocols is invalid. H27 remains a descriptive **post-hoc** analysis, not a causal result.

## Research question and actual design

**Question:** Is the H26 C1 collapse primarily fixed-position/wire-code preference, instruction-language sensitivity, or genuine semantic task-state failure? The hypothesis is still unproven. H28 provides a prospective experiment capable of testing response differences under controlled label permutations; it does NOT claim the model has been run, logits recorded, or that any cause has been established.

- **64** freshly authored source-bound first-party synthetic **public DEV** cases; **4 domains** × **8 states** × **2 task languages**. Domains: civic permit desk, cold-chain handoff, community event registry, marine specimen log. Per-language count EN 32, AR 32; 8 per state. All examples are authored as text and symbolic observations, not pixels or real UI.
- The eight target outcomes are C1 read-only locate, C2 refresh stale view, C3 ignore hostile page, C4 independently verify persisted result, C5 clarify required missing parameter, C6 resolve contradictory trusted evidence, C7 explicitly refuse destructive host/audit command, and **C0 withhold because access authority cannot be established**. C0 is **not** proof of an explicit refusal, and no model prediction grants access.
- Fixed original case-data SHA256 **`63f67ad16d23fe19c178c6dd35ef2d06ad548c691be514d2b694b8440bd5955c`**. Dataset code `h28/corpus.py` validates original cases with exact source reconstruction, language/state/family counts and embedded SHA pin. Oracle data are held in `private_reference`, never in model messages.
- `h28/protocol.py` maps semantic C1–C7 choices over **7 cyclic code rotations** (C0 remains fixed). The system instruction is independently given in either English or Arabic, crossing each of the 64 source scenarios. This yields a CPU-planned balanced **64 × 7 × 2 = 896** prompt matrix; each code C1–C7 represents each non-C0 semantic class once per rotation group. Prompt-language and user-goal language are separate factors.
- The *codebook* is always explicitly included in a system instruction; native page text remains only in user `observation.untrusted_page_text`, never promoted to system authority. `h25.baseline.user_payload` limits model input to `goal`, `locale` and synthetic `observation`. The model does not see the original case ID, reference code, dataset state or choice scoring.
- `tools/build_h28_dev_plan.py` creates `model_messages.jsonl`, separate `evaluator_only.jsonl`, and `manifest.json` **only inside a NEW user-specified destination**. Source-only, create-only, no GPU/model calls or network. The opaque request key is envelope metadata, NOT model context. An evaluation runner MUST pass only each item's `messages` to the model, never the whole envelope or evaluator sidecar. The original source is published in Hala; **the evaluator sidecar is not an independently blinded secret**.
- `h28/replay.py` is an auditor for **hypothetical FUTURE externally supplied predictions**. It requires the previously observed raw file SHA256, exact model/source/plan envelope, complete and ordered 896 request keys, explicit inference-error records, and independent replay of each source label and rotated choice. It returns separate raw-wire and semantic-code frequencies, per-task and system-instruction language, per-state and per-rotation summaries, unsafe non-refusal suggestions and unverified-permission access-like suggestions. It cannot attest a physical GPU or independently prove that a claimed model run occurred.
- The auditor's synthetic unit fixtures include *all-C0*, *all-wire-C1* and *perfect-oracle* outputs to validate the scorer. **None are actual Hala model results.** Never quote fixture accuracy as model success. No H28 result files, model checkpoints, adapters or GPU evaluations have been produced.

## Reproduction (source/CPU, no billable service)

```powershell
python -m unittest tests.test_h28_counterbalanced tests.test_h28_plan_builder tests.test_h28_source_audit
python -m unittest discover -s tests

# Optional: create a NEW small, local development-plan bundle; do not publish evaluator
# sidecar as secret TEST. Do not send it to a model:
python -m tools.build_h28_dev_plan --output-directory .\new-h28-dev-plan
```

Deterministic full-matrix manifest, **without inference**:

| Item | SHA256 |
| --- | --- |
| Original source cases | `63f67ad16d23fe19c178c6dd35ef2d06ad548c691be514d2b694b8440bd5955c` |
| Proposed 896 model-message JSONL | `6027801cf42ab42529f085cea838b03ec4768185d10bfa383d28d2d9641748e3` |
| Separate 896-row evaluator JSONL | `6ab0f718d7495fd3f23934369746ac32fbea442fa4cb1a802394248afb525afa` |

These digests permit future checks that the source and plan were not silently edited. The plan is DEV, not an unseen TEST.

## Pre-registered metrics and prospective safeguards

For every run on appropriately authorized, non-billable resources, record **format validity vs semantic correctness**, explicit unsafe-user C7 recall and wrongful alternative recommendation rate, C0 withholding vs wrong refusal, unauthorized access-like proposals, validity/accuracy by language × state × rotation × instruction language, conditional wire-C1 frequency versus semantic-C1, coverage, per-case latency/inference exceptions, and verified independently saved task postconditions. Do not infer a calibrated probability from code choice or raw C0 frequency. A cross-rotational reversal may suggest wire-position sensitivity, but without replication, independent labels, causal controls and model logs does not establish generalizable cause.

Do not tune prompts or adapters on public H24/H25/H26/H28 and later call those same cases "unseen TEST". A **future genuinely disjoint, securely held, independently bilingual adjudicated TEST**, with preregistered protocols and blinded independent evaluators, remains a separate requirement. Synthetic scenarios involving destructive commands must never execute those commands. For native testing, first qualify external authorization, target provenance, fresh observations and independently persisted postconditions, and the H23 actual CUDA cancellation/cleanup/restart lifecycle.

## Hala architecture and governance boundaries

- **HalaMind:** bilingual intent/state classification remains strictly advisory and unqualified; bounded syntax does not imply correct choice.
- **HalaType:** genuine pixels, screenshot digests, Arabic RTL/RAQM, label geometry, stale view checks and safe target binding still require independent evaluation. H28 does not test screenshots.
- **HalaReliance:** uncertainty/cascade proposal can only abstain or escalate. Deterministic external host authorization owns permission and side effects; no learned model can grant access.
- Zero out-of-pocket cost; no paid GPU, Kaggle billable overflow, new large Windows weights, or external commercial APIs. Kaggle's previously documented service/rights limitations must be checked before any commercial use.
- Genuine Alibaba Open Code Review, independent PStack/Arabic QA, Jev typed review, source/deep Graft as applicable, and real E2E still require their own evidence; rule previews, fixtures and advisory probabilistic checks are **not release approvals**.
- Founder approval remains necessary for merging or release. Keep H28 Draft, stacked on H27; no edits to Orcel, Aladdin or any other repository.
